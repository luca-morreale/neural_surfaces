
import numpy as np
from argparse import ArgumentParser

from utils import read_mesh
from utils import write_mesh
from utils import parametrize
from utils import clean_mesh
from utils import compute_mesh_geo_measures
from utils import remove_mesh_dangling_faces


parser = ArgumentParser(description='parametrize a mesh')
parser.add_argument('--obj', type=str, help='shape', required=True)
parser.add_argument('--square', action='store_true', default=False, help='use squared slim (default disk)')
parser.add_argument('--free', action='store_true', default=False, help='use squared slim (default disk)')
args = parser.parse_args()

bnd_shape = 'square' if args.square else ('free' if args.free else 'circle')

v, f, _, _ = read_mesh(args.obj)

### remove ears of the model
ears_mask = np.ones(v.shape[0]).astype(bool)
ears_mask, stop = remove_mesh_dangling_faces(f, ears_mask)

face_mask = ears_mask[f]
face_mask = face_mask.sum(axis=-1)
keep_faces = face_mask > 2.0

# remove ears vertices
v, f, _, V_idx, _ = clean_mesh(v, f[keep_faces])
print('removed ', ears_mask.shape[0] - v.shape[0], ' vertices')
print('DONE ear removal')
## DONE ear removal

C, genus = compute_mesh_geo_measures(v, f, target_area=4.0 if args.square else (1.0 if args.free else np.pi) )

v = np.array(v.tolist())
f = np.array(f.tolist())

uva = parametrize(v * C, f, method='slim', it=100, bnd_shape=bnd_shape)

write_mesh(args.obj[:-4] + '_slim.obj', v * C, f, uva, None)
