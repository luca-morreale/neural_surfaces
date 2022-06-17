
import numpy as np
import trimesh

from .colors import sinebow
from .read_OBJ import readOBJ
from .tensor_move import tensor_to_numpy
from .write_OBJ import writeOBJ


def write_mesh(filename, V, F, UV, N, scalars=None, colors=None):
    ext = filename[filename.rfind('.')+1:]

    if ext == 'obj':
        writeOBJ(filename, V, F, UV, N)
    elif ext == 'ply':

        mesh = mesh_to_trimesh_object(V, F, colors)
        if scalars is not None:
            for k, v in scalars.items():
                mesh.vertex_attributes[k] = tensor_to_numpy(v)

        if UV is not None:
            UV = tensor_to_numpy(UV)
            mesh.vertex_attributes['texture_u'] = UV[:, 0]
            mesh.vertex_attributes['texture_v'] = UV[:, 1]
            mesh.vertex_attributes['s'] = UV[:, 0] # for blender visualization
            mesh.vertex_attributes['t'] = UV[:, 1] # for blender visualization

        # not sure how much this will affect later computations
        mesh.remove_unreferenced_vertices()

        mesh.export(filename, include_attributes=True)

    elif ext == 'off':
        mesh = mesh_to_trimesh_object(V, F)
        mesh.export(filename)


def read_mesh(filename):

    ext = filename[filename.rfind('.')+1:]

    if ext == 'obj':
        V, F, UV, _, N = readOBJ(filename)
    elif ext == 'ply':
        mesh = trimesh.load(filename, process=False)
        V = mesh.vertices
        F = mesh.faces
        if 'texture_u' in mesh.metadata['ply_raw']['vertex']['data']:
            u = mesh.metadata['ply_raw']['vertex']['data']['texture_u']
            v = mesh.metadata['ply_raw']['vertex']['data']['texture_v']
            UV = np.concat([u.reshape(-1,1), v.reshape(-1,1)], axis=1)
        else:
            UV = None
        N = mesh.vertex_normals
    else:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(filename)
        mesh = ms.current_mesh()

        V = mesh.vertex_matrix()
        F = mesh.face_matrix()
        N = mesh.vertex_normal_matrix()
        UV = None

    return V,F,UV,N


def save_mesh_with_correspondences(filename, points, faces, correspondences, vertices_colors=None):
    L = len(correspondences)
    correspondences_colors = [ np.concatenate((sinebow(i/L), [1]))*255 for i in range(L) ]

    mesh = mesh_to_trimesh_object(points, faces, vertices_colors)

    for i, point in enumerate(correspondences):
        sphere = generate_sphere(point, radius=0.015)
        mesh  += sphere
        mesh.visual.vertex_colors[-sphere.visual.vertex_colors.shape[0]:] = correspondences_colors[i]

    mesh.export(filename)


def mesh_to_trimesh_object(points, faces, vertices_colors=None):
    vertices = tensor_to_numpy(points)
    faces    = tensor_to_numpy(faces)
    if vertices_colors is None:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    else:
        vertices_colors = tensor_to_numpy(vertices_colors)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertices_colors, process=False)
    return mesh


def generate_sphere(center, radius=0.0025, color=None):
    center = tensor_to_numpy(center)
    if color is None:
        color = trimesh.visual.random_color()

    sphere = trimesh.primitives.Sphere(center=center, radius=radius)
    sphere.visual.vertex_colors = color

    return sphere
