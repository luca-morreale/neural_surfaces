
import numpy as np
import torch
import trimesh

from .io_mesh import mesh_to_trimesh_object
from .io_mesh import write_mesh
from .tensor_move import tensor_to_numpy


def get_mesh_edges(V, F):
    mesh = mesh_to_trimesh_object(V, F)
    boundary              = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1) # edges which appears only once
    vertices_index        = mesh.edges[boundary]
    return vertices_index


def normalize(V, F):
    import pymeshlab

    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(V, F)
    ms.add_mesh(mesh)

    ms.transform_scale_normalize(unitflag=True)
    ms.transform_translate_center_set_origin(traslmethod='Center on Scene BBox')
    # ms.transform_rotate(angle=-90.0)
    ms.transform_rotate(angle=90.0, rotaxis='Z axis')

    mesh = ms.current_mesh()
    V_small  = mesh.vertex_matrix()
    F_small  = mesh.face_matrix()
    N_small  = mesh.vertex_normal_matrix()

    return V_small, F_small, N_small


def compute_curvature(V, F, type='Gaussian Curvature'):
    import pymeshlab

    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(V, F)
    ms.add_mesh(mesh)

    ms.compute_curvature_principal_directions(curvcolormethod=type)
    mesh = ms.current_mesh()
    curvature = mesh.vertex_quality_array()

    return curvature


## remove unreferenced vertices and compute normals
def clean_mesh(V, F):
    import pymeshlab

    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(V, F)
    mesh.add_vertex_custom_scalar_attribute(np.arange(V.shape[0]), 'idx')
    ms.add_mesh(mesh)
    ms.remove_unreferenced_vertices()
    ms.re_orient_all_faces_coherentely()
    ms.re_compute_vertex_normals()

    mesh = ms.current_mesh()

    V_small  = mesh.vertex_matrix()
    F_small  = mesh.face_matrix()
    N_small  = mesh.vertex_normal_matrix()
    NF_small = mesh.face_normal_matrix()
    V_idx = mesh.vertex_custom_scalar_attribute_array('idx').astype(np.int64)

    return V_small, F_small, N_small, V_idx, NF_small


### upsample the mesh
def upsample_mesh(V, F, uv, threshold=0.2):
    import pymeshlab

    write_mesh('/tmp/file.obj', V, F, uv, None)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh('/tmp/file.obj')

    p = pymeshlab.Percentage(threshold)
    ms.subdivision_surfaces_midpoint(threshold=p)
    ms.re_compute_vertex_normals()
    ms.re_compute_face_normals()

    mesh = ms.current_mesh()

    V_large  = mesh.vertex_matrix()
    F_large  = mesh.face_matrix()
    try:
        UV_large = mesh.vertex_tex_coord_matrix()
    except:
        UV_large = None

    return V_large, F_large, UV_large


## compute genus and area size
def compute_mesh_geo_measures(V, F, target_area=np.pi):
    import pymeshlab

    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(V, F)
    ms.add_mesh(mesh)

    out_dict = ms.compute_geometric_measures()
    A = out_dict['surface_area']
    C = np.sqrt( target_area / A )

    out_dict = ms.compute_topological_measures()

    return C, out_dict['number_holes']-1


def remove_mesh_dangling_faces(faces, points_selected):
    # remove points\faces that are not fully selected, onlu faces that have 3 vertices selected

    # identify which faces are completely selected (all 3 vertex)
    face_mask  = points_selected[faces]
    face_mask  = face_mask.sum(axis=-1)
    keep_faces = face_mask > 2.0

    # identify which points are selected by more than 1 face
    mask = np.zeros(points_selected.shape[0])
    for face_idx in np.nonzero(keep_faces)[0]:
        mask[faces[face_idx]] += 1.0

    mask_binary = mask >= 2.0 # remove points if they are not selected by at least 2 faces

    ### flap ears -> remove vertices that belong to only one face
    ### repeat (recursive)
    return mask_binary, (mask_binary != points_selected).sum() > 0


def faces_to_vertices(points, faces, scalar, to_torch=False):
    points = tensor_to_numpy(points)
    faces  = tensor_to_numpy(faces)
    scalar = tensor_to_numpy(scalar)

    # convert to numpy
    mesh = trimesh.Trimesh(points, faces, process=False)
    vertex = mesh.faces_sparse.dot(scalar.astype(np.float64))
    vertex_val = (vertex / mesh.vertex_degree.reshape(-1)).astype(np.float64)

    if to_torch:
        vertex_val = torch.from_numpy(vertex_val).float()

    return vertex_val


def close_holes(v, f):
    import pymeshlab
    holesize = 10
    _, genus = compute_mesh_geo_measures(v, f, target_area=1.0)

    while genus != 0 and holesize < 500:
        ## close holes
        ms = pymeshlab.MeshSet()
        mesh = pymeshlab.Mesh(v, f)
        ms.add_mesh(mesh)
        ms.re_orient_all_faces_coherentely()

        ms.close_holes(maxholesize=holesize, newfaceselected=False, selfintersection=False)

        ## get new faces and vertices
        mesh = ms.current_mesh()
        v = mesh.vertex_matrix()
        f = mesh.face_matrix()

        ## check genus
        v = np.array(v.tolist())
        _, genus = compute_mesh_geo_measures(v, f, target_area=1.0)

        holesize += 10

    if genus > 0:
        print('FAILED')
        return None

    return v, f


def simplify_mesh(v, f, target_num_faces=None, preserve_boundary=True):
    import pymeshlab

    if target_num_faces is None:
        target_num_faces = int(f.shape[0] / 2)

    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(v, f)
    ms.add_mesh(mesh)
    ms.simplification_quadric_edge_collapse_decimation(targetfacenum=target_num_faces,
                            preservetopology=True, preserveboundary=preserve_boundary)

    mesh = ms.current_mesh()
    v = mesh.vertex_matrix()
    f = mesh.face_matrix()

    v = np.array(v.tolist())
    f = np.array(f.tolist())

    return v, f
