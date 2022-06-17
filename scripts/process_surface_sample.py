
import numpy as np
import torch
from argparse import ArgumentParser
from pathlib import Path
from scipy import spatial
from tqdm import tqdm

from utils import clean_mesh
from utils import close_holes
from utils import compute_mesh_geo_measures
from utils import list_all_files
from utils import read_mesh
from utils import simplify_mesh
from utils import upsample_mesh
from utils import write_mesh

from .patches_function import prepare_global_sample

## convert mesh into NSM sample

def close_holes_mesh(path):
    v, f, _, _ = read_mesh(path)

    orig_faces = f.shape[0]
    _, genus = compute_mesh_geo_measures(v, f, target_area=1.0)

    if genus == 0:
        return None
    print(path)

    out = close_holes(v, f)
    if out is None:
        return None
    v, f = out

    num_faces = f.shape[0] - orig_faces
    print('Num inserted faces: ', num_faces)
    new_path = path[:-4] + '_holes.obj'
    write_mesh(new_path, v, f, None, None)
    return v, f, new_path


def is_mesh_file(path):
    ext = path[-3:]
    return ext == 'ply' or ext == 'obj' or ext == 'off'



parser = ArgumentParser(description='convert mesh into Neural Surface Maps sample')
parser.add_argument('--data', type=str, help='path to file or folder containing mesh to conver', required=True)
parser.add_argument('--square', action='store_true', default=False, help='use squared slim (default is circle)')
parser.add_argument('--free', action='store_true', default=False, help='slim with no boundary (default is circle)')
parser.add_argument('--rand_seed', type=int, default=42, help='random seed')
parser.add_argument('--verbose', action='store_true', default=False, help='verbose process')
args = parser.parse_args()


#############
np.random.seed(args.rand_seed)

bnd_shape = 'square' if args.square else ('free' if args.free else 'circle')

files_list = list_all_files(args.data)
files_list = [ file for file in files_list if '_slim' not in file and '_parametrized' not in file and '_freeslim' not in file and '_holes' not in file ]

### print files names
if args.verbose:
    for file in files_list:
        print(file)

### process files
for file in tqdm(files_list):

    if args.verbose:
        print('Start processing', file)

    input_path = Path(file)
    shape_name = str(input_path.stem)

    out = close_holes_mesh(file)
    if out is None:
        ## create sample data
        sample = prepare_global_sample(file, shape_name, global_param=True, bnd_shape=bnd_shape, can_upsample=False, normalize_mesh=False)
        V  = sample['points']
        F  = sample['faces']
        UV = sample['param']

    else:
        v, f, new_path = out
        v_orig, f_orig, _, _ = read_mesh(file)

        sample = prepare_global_sample(new_path, shape_name, global_param=True, bnd_shape=bnd_shape, can_upsample=False, normalize_mesh=False)
        V  = sample['points']
        F  = sample['faces']
        UV = sample['param']

        tree = spatial.KDTree(v_orig)
        distances, indices = tree.query(V.numpy())

        mask_verts = distances < 1.0e-4
        mask_faces = mask_verts[F].sum(axis=1) > 2.0

        trans_faces = indices[F][mask_faces]
        new_faces = []

        for f in trans_faces:
            if np.abs(f_orig-f).sum(axis=-1).min() == 0:
                new_faces.append(f)

        original_faces = np.vstack(new_faces)
        original_points = np.zeros_like(v_orig)
        original_points[indices] = V[mask_verts]

        original_param = np.zeros_like(v_orig)[:,:2]
        original_param[indices] = UV[mask_verts]

        V_small, F_small, N_small, V_idx, NF_small = clean_mesh(original_points, original_faces)

        sample['original_faces']  = torch.from_numpy(F_small).long()
        sample['original_param']  = torch.from_numpy(original_param[V_idx]).float()
        sample['original_points'] = torch.from_numpy(V_small).float()
        sample['V_idx_original']  = torch.from_numpy(indices[mask_verts]).long()

    ## simplify 2D domain for filtering / domain check
    v = torch.cat([UV, torch.zeros(UV.size(0),1)], dim=1).numpy() # 2D domain to 3D
    if out is not None:
        v = torch.cat([sample['original_param'], torch.zeros(sample['original_param'].size(0),1)], dim=1).numpy() # 2D domain to 3D
        f = F[f_orig.shape[0]:]

    target_num_faces = min(800, int(F.size(0)/2) )
    v, f = simplify_mesh(v, F.numpy(), target_num_faces=target_num_faces, preserve_boundary=True)
    if args.verbose:
        print('Domain size ', f.shape[0])

    sample['domain_vertices'] = torch.from_numpy(v).float()[:, :2] # remove z coordinate
    sample['domain_faces']    = torch.from_numpy(f).long()


    ## add visual grid and all
    if args.verbose:
        print('upsampling')
    points, faces, uvs = upsample_mesh(V.numpy(), F.numpy(), UV.numpy(), threshold=0.3)
    sample['oversampled_points'] = torch.from_numpy(points).float()
    sample['oversampled_faces']  = torch.from_numpy(faces).float()
    sample['oversampled_param']  = torch.from_numpy(uvs).float()

    path_prefix = str(input_path.parent.absolute() / input_path.stem)

    ## save parameterized mesh
    output_file = path_prefix + '_' + bnd_shape + '_parametrized.obj'
    if args.verbose:
        print('Saving mesh file', output_file)
    write_mesh(output_file, V * sample['C'], F, UV, None)


    ## save file as pth
    output_file = path_prefix + '_' + bnd_shape + '.pth'
    if args.verbose:
        print('Saving binary file', output_file)
    torch.save(sample, output_file)

    if args.verbose:
        print('Done processing', file)
