
import torch

from .mixin import Mixin


class SurfaceCheckpointing(Mixin):

    def check_imports(self):
        ### check if library exists, if does not remove function
        try:
            import trimesh
        except ImportError as err:
            print('Error missing library ' + str(err))
            self.save_surface = self.empty_function
            self.interpolate_temperature = self.empty_function

    # ======================== Save Mesh ==================================== #
    def save_surfaces_from_list(self, points3D_list, faces_list, uvs_list, scalars, colors, names, checkpoint_dir):
        for points3D, faces, uvs, scalar, color, name in zip(points3D_list, faces_list, uvs_list, scalars, colors, names):
            self.save_surface(points3D, faces, uvs, scalar, color, checkpoint_dir, prefix=name)

    def save_surface(self, points3D, faces, uvs, scalars, colors, checkpoint_dir, prefix=''):
        ### Save surface as ply file with colors, scalars
        from utils import write_mesh

        out_folder = self.compose_out_folder(checkpoint_dir, ['mesh'])
        filename = '{}/{}mesh.ply'.format(out_folder, prefix)

        write_mesh(filename, points3D, faces, uvs, None, scalars, colors)

        self.save_edges_list(points3D, faces, checkpoint_dir, prefix)


    def save_edges_list(self, points3D, faces, checkpoint_dir, prefix=''):
        ### Save list of boundary edges
        from utils import get_mesh_edges

        out_folder = self.compose_out_folder(checkpoint_dir, ['mesh'])
        filename = '{}/{}mesh_edges.txt'.format(out_folder, prefix)

        ## extract boundary
        edges = get_mesh_edges(points3D, faces)
        with open(filename, 'w') as stream:
            for ed in edges:
                stream.write(f'{ed[0]} {ed[1]}\n')
            stream.write('--\n')


    # ======================== Save Vector Field ==================================== #
    def save_vectorfield(self, points3D, vectors, scalars, checkpoint_dir, prefix=''):
        ### Save vectorfield as points with arrows and scalars

        # write a CSV-like file containing all the infos
        out_folder = self.compose_out_folder(checkpoint_dir, ['vector'])
        filename = '{}/{}mesh.pts'.format(out_folder, prefix)

        V = self.move_to_numpy(points3D)
        header = 'x y z'

        scs  = {}
        vecs = {}
        for k in scalars.keys():
            header += ' ' + k
            scs[k] = self.move_to_numpy(scalars[k])
        for k in vectors.keys():
            header += ' vector_' + k + '_x'
            header += ' vector_' + k + '_y'
            header += ' vector_' + k + '_z'
            vecs[k] = self.move_to_numpy(vectors[k])

        with open(filename, 'w') as stream:
            stream.write(header + '\n')
            for i in range(V.shape[0]):
                stream.write('{} {} {}'.format(V[i,0],V[i,1],V[i,2]))

                for k in scs.keys():
                    stream.write(' {}'.format(scs[k][i]))

                for k in vecs.keys():
                    stream.write(' {} {} {}'.format(vecs[k][i,0], vecs[k][i,1], vecs[k][i,2]))

                stream.write('\n')

    # ======================== Interpolate face data ==================================== #
    def interpolate_temperature(self, vertices, faces, temperature):
        ### Interpolate values between faces and vertices
        import trimesh

        F = self.move_to_numpy(faces)
        V = self.move_to_numpy(vertices)
        D = self.move_to_numpy(temperature)

        mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)

        indices = mesh.vertex_faces
        mask    = indices >= 0

        interp = (D[indices] * mask).sum(axis=-1) / mask.sum(axis=-1)
        return torch.from_numpy(interp, dtype=torch.float)
