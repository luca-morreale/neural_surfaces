
import torch
import trimesh

from argparse import ArgumentParser
from chamfer.chamfer3D import chamfer_3DDist
from utils import read_mesh


parser = ArgumentParser(description='Compute Chamfer Distance from meshes')
parser.add_argument('--gt', type=str, help='original shape')
parser.add_argument('--mesh', type=str, help='test shape')
args = parser.parse_args()


shape_file = args.source
test_file = args.test

V, F, _, _ = read_mesh(shape_file)
V_test, F_test, _, _ = read_mesh(test_file)


V = torch.from_numpy(V).float()
F = torch.from_numpy(F).long()

V_test = torch.from_numpy(V_test).float()
F_test = torch.from_numpy(F_test).long()


## compute double sided chamfer distance

V = V.cuda().unsqueeze(0)
# V.requires_grad = True
V_test = V_test.cuda().unsqueeze(0)
V_test.requires_grad = True

distChamfer = chamfer_3DDist()

dist1, dist2, idx1, idx2 = distChamfer(V, V_test)

dist1 = dist1.sqrt()
dist2 = dist2.sqrt()


print('Dist 1 no sampling', dist1.mean().item())
print('Dist 2 no sampling', dist2.mean().item())
print('AVG chamfer no sampling', (dist1.mean().item()+dist2.mean().item())/2.0)

samples_gt, _   = trimesh.sample.sample_surface(trimesh.Trimesh(V.cpu().squeeze().numpy(), F.cpu().numpy()), 100000)
samples_test, _ = trimesh.sample.sample_surface(trimesh.Trimesh(V_test.detach().cpu().squeeze().numpy(), F_test.cpu().numpy()), 100000)

dist1, dist2, idx1, idx2 = distChamfer(V, V_test)

dist1 = dist1.sqrt()
dist2 = dist2.sqrt()

print('Dist 1 sampling', dist1.mean().item())
print('Dist 2 sampling', dist2.mean().item())
print('AVG chamfer sampling', (dist1.mean().item()+dist2.mean().item())/2.0)
