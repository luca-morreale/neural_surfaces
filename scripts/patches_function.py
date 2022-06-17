
from pathlib import Path
import gdist
import numpy as np
import os
import torch

from utils import clean_mesh
from utils import compute_curvature
from utils import compute_mesh_geo_measures
from utils import normalize
from utils import parametrize
from utils import read_mesh
from utils import remove_mesh_dangling_faces
from utils import upsample_mesh
from utils import write_mesh


## Set of function used to convert meshes into samples for NSM and NCS

def extract_patches(input_file, name, bnd_shape, patch_size, global_param=True, can_upsample=True, save=True, verbose=False):
    input_path =  Path(input_file)
    output_folder = input_path.parent.absolute()
    file_name = input_path.name


    global_sample = prepare_global_sample(input_file, name, global_param, bnd_shape, can_upsample=can_upsample, verbose=verbose)
    if verbose:
        print('DONE preparing global sample')

    points    = global_sample['points'].double().numpy()
    faces     = global_sample['faces'].numpy()
    normals   = global_sample['normals'].numpy()
    curvature = global_sample['curvature'].numpy()

    curvature_faces = curvature[faces].sum(axis=-1)

    write_mesh(os.path.join(output_folder, f'{file_name}_filtered.obj'), points, faces, None, None)


    if verbose:
        print('Start patch processing')

    original_threshold = patch_size
    current_threshold = original_threshold


    N = points.shape[0]
    included_points = np.zeros(N)
    source_indices = select_next_point(included_points)

    patches = []
    while stop_condition(included_points, N):

        # compute geodesic distance
        distances = gdist.compute_gdist(points, faces.astype(np.int32), source_indices.astype(np.int32), target_indices=None, max_distance=current_threshold)
        mask = (distances >= -1.0e-6) * (distances < 1.0)

        mask, stop = remove_mesh_dangling_faces(faces, mask)

        # remove dangling faces iteratively
        while stop and mask.sum() > 0:
            mask, stop = remove_mesh_dangling_faces(faces, mask)
            if int(mask.sum()) < 10:
                source_indices = select_next_point(included_points)
                # source_indices = np.array([source_indices])
                if verbose:
                    print('reset number')
                continue

        # check enough points are selected or restart
        if int(mask.sum()) < 10:
            source_indices = select_next_point(included_points)
            # source_indices = np.array([source_indices])
            if verbose:
                print('reset number')
            continue

        if verbose:
            print(int(mask.sum()))

        # compute face mask (which faces are selected)
        face_mask = mask[faces]
        face_mask = face_mask.sum(axis=-1)
        keep_faces = face_mask > 2.0

        # check enough faces are selected or restart
        if keep_faces.sum() < 3:
            source_indices = select_next_point(included_points)
            # source_indices = np.array([source_indices])
            if verbose:
                print('reset faces')
            continue


        # extract the patch
        V_local, F_local, _, V_idx, _ = clean_mesh(points, faces[keep_faces])
        F_idx = np.nonzero(keep_faces)[0]
        N_local = normals[V_idx]

        # check no full overlap
        if included_points[V_idx].sum() == V_idx.shape[0]:
            if verbose:
                print('no new points')
            source_indices = select_next_point(included_points)
            # source_indices = np.array([source_indices])
            continue

        # check genus of the patch
        C, genus = compute_mesh_geo_measures(V_local, F_local, 4.0 if bnd_shape=='square' else np.pi)
        if genus > 0: # skip, patch need to be genus 0
            current_threshold /= 2.0 # restart by halving the patch size
            if verbose:
                print('reset genus', genus)
            write_mesh(os.path.join(output_folder, f'debug.obj'), V_local, F_local, None, None)
            continue

        # parametrize the patch
        V_local = np.array(V_local.tolist())
        UV_local = parametrize(V_local*C, F_local, 'slim', it=100, bnd_shape=bnd_shape)

        # save the patch
        sample = {}
        sample['faces']   = torch.from_numpy(F_local).long()
        sample['param']   = torch.from_numpy(UV_local).float()
        sample['normals'] = torch.from_numpy(N_local).float()
        sample['points']  = torch.from_numpy(V_local).float()
        sample['idx']     = torch.from_numpy(V_idx).long()
        sample['F_idx']   = torch.from_numpy(F_idx).long()
        sample['C']       = C
        sample['F_curv']  = torch.from_numpy(curvature_faces[F_idx]).float()

        patches.append(sample)

        # remove a subset of points to increase overlap
        probs = np.random.uniform(size=mask.shape)
        mask_ = mask * (probs > 0.5)
        while mask_.sum() < 0:
            probs = np.random.uniform(size=mask.shape)
            mask_ = mask * (probs > 0.5)
        mask = mask

        # update selected vertices
        included_points[mask] = 1.0

        if verbose:
            print(len(patches), int(included_points.sum()), N)

        # check if there are any more points to select
        if not stop_condition(included_points, N):
            break

        # select next point
        source_indices = select_next_point(included_points)
        # source_indices = np.array([source_indices])
        current_threshold = original_threshold


    ### iterate over all patches, if one has nans, recompute param
    patches = fix_nans(patches, bnd_shape, verbose=verbose)

    # compose the data sample
    sample = {}
    sample['patches'] = patches
    sample['global']  = global_sample

    # compose file name
    output_file = os.path.join(output_folder, f'{name}_{original_threshold}_{len(patches)}.pth')

    if save:
        # save sample
        torch.save(sample, output_file)

    if verbose:
        print('Done')

    # debug prints
    if verbose:
        print(len(patches))

    return sample


def prepare_global_sample(input_file, shape_name, global_param, bnd_shape, can_upsample=True, normalize_mesh=True, verbose=False):
    points, faces, _, _ = read_mesh(input_file)
    if verbose:
        print('DONE loading mesh')
        print(f'{points.shape[0]} points')
        print(f'{faces.shape[0]} faces')


    if normalize_mesh:
        points, faces, _ = normalize(points, faces)
        points = np.array(points.tolist())
        faces  = np.array(faces.tolist())
        if verbose:
            print('DONE mesh normalization')
            print(f'{points.shape[0]} points')
            print(f'{faces.shape[0]} faces')


    if points.shape[0] < 80000 and can_upsample:
        if verbose:
            print('upsampling')
        points, faces, _ = upsample_mesh(points, faces, None, threshold=0.4)
        points = np.array(points.tolist())
        faces  = np.array(faces.tolist())
        if verbose:
            print(f'{points.shape[0]} points')
            print(f'{faces.shape[0]} faces')

    ### remove ears of the model
    ears_mask = np.ones(points.shape[0])
    ears_mask, _ = remove_mesh_dangling_faces(faces, ears_mask)

    face_mask = ears_mask[faces]
    face_mask = face_mask.sum(axis=-1)
    keep_faces = face_mask > 2.0


    # remove ears vertices
    points, faces, normals, V_idx, face_normals = clean_mesh(points, faces[keep_faces])
    if verbose:
        print('removed ', ears_mask.shape[0] - points.shape[0], ' vertices')
        print('DONE ear removal')
    ## DONE ear removal

    points = np.array(points.tolist())
    if global_param:
        C_global, genus = compute_mesh_geo_measures(points, faces, 4.0 if bnd_shape=='square' else (1.0 if bnd_shape=='free' else np.pi) )
        uv_slim_global = parametrize(points*C_global, faces, 'slim', it=100, bnd_shape=bnd_shape)

    curvature = compute_curvature(points, faces)
    # curvature = (curvature - curvature.min()) / (curvature.max() - curvature.min())
    curvature = np.array(curvature.tolist())
    curvature_faces = curvature[faces].sum(axis=-1)

    global_sample = {}
    if global_param:
        global_sample['param']      = torch.from_numpy(uv_slim_global).float()
    global_sample['points']         = torch.from_numpy(points).float()
    global_sample['faces']          = torch.from_numpy(faces).long()
    global_sample['normals']        = torch.from_numpy(normals).float()
    global_sample['face_normals']   = torch.from_numpy(face_normals).float()
    global_sample['curvature']      = torch.from_numpy(curvature).float()
    global_sample['name']           = shape_name
    global_sample['V_idx_original'] = torch.from_numpy(V_idx).long()
    if global_param:
        global_sample['C'] = C_global

    return global_sample




def stop_condition(included_points, num_points):
    return (included_points.sum() < num_points)

def select_next_point(included_points):
    source_indices = np.random.choice(np.nonzero(included_points == 0.0)[0], 1).astype(np.int64)[0]
    return np.array([source_indices])

def fix_nans(patches, bnd_shape, verbose=False):
    ### iterate over all patches, if one has nans, recompute param
    for i, patch in enumerate(patches):
        if torch.isnan(patch['param']).any():
            if verbose:
                print(f'Trying to fix a NaN {i}')
            UV_local = parametrize(np.array(patch['points'].tolist())*patch['C'], patch['faces'].numpy(), 'slim', it=100, bnd_shape=bnd_shape)
            patch['param'] = torch.from_numpy(UV_local).float()

            if verbose:
                if torch.isnan(patch['param']).any():
                    print('Failed!')
                else:
                    print('Succeded!')
    return patches
