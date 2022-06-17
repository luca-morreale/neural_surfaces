
import matplotlib as mpl
import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from .colors import sinebow
from .tensor_move import tensor_to_numpy

mpl.rcParams['agg.path.chunksize'] = 10000


def save_uv_correspondences(filename, pred_points, gt_points):
    correspondences_colors = [ np.concatenate((sinebow(i/7), [1]))*255 for i in range(7) ]

    plt.figure(figsize=(10, 10), dpi=60)

    pred_points = tensor_to_numpy(pred_points)
    gt_points   = tensor_to_numpy(gt_points)

    for i, pt in enumerate(pred_points):
        color = correspondences_colors[i][:-1].reshape(1,3) / 255.0
        plt.scatter(pt[0], pt[1], marker='x', c=color, alpha=0.75, s=2.0)
        plt.scatter(gt_points[i][0], gt_points[i][1], marker='o', c=color, alpha=0.75, s=2.0)

    plt.axis('equal')
    plt.savefig(filename)
    plt.close()


def save_uv_displacement(filename, uv_points, displacement, triangles=None):
    # draw image of conformal points given faces
    uv_points    = tensor_to_numpy(uv_points)
    displacement = tensor_to_numpy(displacement)

    plt.figure(figsize=(10, 10), dpi=60)
    plt.title('Grid displacement')
    if triangles is None:
        plt.scatter(uv_points[:,0], uv_points[:,1], alpha=0.75, s=2.0)
    else:
        triangles = tensor_to_numpy(triangles)
        plt.triplot(uv_points[:,0], uv_points[:,1], triangles, alpha=0.75, linewidth=0.5)

    for i in range(uv_points.shape[0]):
        plt.arrow(uv_points[i,0], uv_points[i,1], displacement[i,0], displacement[i,1], linewidth=0.8)

    plt.axis('equal')
    plt.savefig(filename)
    plt.close()


def save_uv_layout(filename, uv_points, triangles, landmarks=None):
    uv_points = tensor_to_numpy(uv_points)
    triangles = tensor_to_numpy(triangles)

    # draw image of conformal points given faces
    plt.figure(figsize=(10, 10), dpi=90)
    plt.title('Grid layout')

    if landmarks is not None:
        landmarks = [ tensor_to_numpy(land) for land in landmarks ]
        L = landmarks[0].shape[0]
        landmarks_colors = [ np.concatenate((sinebow(i/L), [1]))*255 for i in range(L) ]

        for i in range(L):
            color = landmarks_colors[i][:-1].reshape(1,3) / 255.0
            for l, lands in enumerate(landmarks):
                alpha = 1.0 if l == 0 else 0.5
                plt.scatter(lands[i,0], lands[i,1], marker='o', alpha=alpha, s=100.0, c=color)

    plt.triplot(uv_points[:,0], uv_points[:,1], triangles, linewidth=0.5, c='k')

    plt.axis('equal')
    plt.savefig(filename)
    plt.close()


def save_uv_layout_colormap(filename, uv_points, scalar):
    uv_points = tensor_to_numpy(uv_points)
    scalar    = tensor_to_numpy(scalar)

    # draw image of conformal points given faces
    plt.figure(figsize=(10, 10), dpi=60)
    plt.title('Distortion colormap')

    min_x = uv_points[:, 0].min()
    max_x = uv_points[:, 0].max()
    min_y = uv_points[:, 1].min()
    max_y = uv_points[:, 1].max()

    # interpolate the scalar values
    grid_x, grid_y = np.mgrid[min_x:max_x:1000j, min_y:max_y:1000j]
    grid_z = scipy.interpolate.griddata(uv_points, scalar, (grid_x, grid_y), method='linear')
    # plot color map
    plt.pcolormesh(grid_x, grid_y, np.ma.masked_invalid(grid_z), cmap='jet', vmin=0.0, shading='auto')
    plt.colorbar()

    plt.axis('equal')
    plt.savefig(filename)

    plt.scatter(uv_points[:, 0], uv_points[:, 1], marker='o', c='r', alpha=0.75, s=10.0)
    plt.savefig(filename+'_wpoints.png')

    plt.close()


def save_overlapping_uv_layout(filename, uv_points, triangles):
    colors = ['green', 'red']

    plt.figure(figsize=(10, 10), dpi=60)
    for i, grid in enumerate(uv_points):
        grid = tensor_to_numpy(grid)
        tris = tensor_to_numpy(triangles[i])
        plt.triplot(grid[:,0], grid[:,1], tris, color=colors[i], alpha=0.75, linewidth=1.0)
    plt.axis('equal')
    plt.savefig(filename)
    plt.close()


def save_mat_as_image(filename, mat):
    plt.figure(figsize=(10, 10), dpi=60)
    plt.matshow(tensor_to_numpy(mat))
    plt.axis('equal')
    plt.savefig(filename)
    plt.close()


def save_domain_image(filename, points):
    plt.figure(figsize=(10, 10), dpi=60)
    plt.title('Domain')
    ax = plt.gca()

    if len(points) == 0:
        rect = Circle((0.0, 0.0), 1.0, ec='g', linewidth=2.0, fill=False)
        ax.add_patch(rect)
    else:
        rect = Circle((0.0, 0.0), 1.0, ec='k', linewidth=2.0, fill=False)
        ax.add_patch(rect)
        points = tensor_to_numpy(points)
        if points.ndim < 2:
            points = points.reshape(1, -1)
        plt.scatter(points[:, 0], points[:, 1], marker='o', c='r', alpha=0.75, s=10.0)


    plt.axis('equal')
    plt.savefig(filename)
    plt.close()

def save_boundary_domain_image(filename, left, right, top, bottom):
    plt.figure(figsize=(10, 10), dpi=60)
    plt.title('Boundary')
    ax = plt.gca()

    rect = Circle((0.0, 0.0), 1.0, ec='k', linewidth=2.0, fill=False)
    ax.add_patch(rect)

    left   = tensor_to_numpy(left)
    right  = tensor_to_numpy(right)
    top    = tensor_to_numpy(top)
    bottom = tensor_to_numpy(bottom)

    left_out = left[:, 0] < -1.0
    left_in  = left[:, 0] > -1.0
    plt.scatter(left[left_out, 0], left[left_out, 1], marker='o', c='g', alpha=0.75, s=10.0)
    plt.scatter(left[left_in, 0],  left[left_in, 1], marker='o', c='m', alpha=0.75, s=10.0)

    right_out = right[:, 0] > 1.0
    right_in  = right[:, 0] < 1.0
    plt.scatter(right[right_out, 0], right[right_out, 1], marker='o', c='g', alpha=0.75, s=10.0)
    plt.scatter(right[right_in, 0],  right[right_in, 1], marker='o', c='m', alpha=0.75, s=10.0)

    top_out = top[:, 1] > 1.0
    top_in  = top[:, 1] < 1.0
    plt.scatter(top[top_out, 0], top[top_out, 1], marker='o', c='g', alpha=0.75, s=10.0)
    plt.scatter(top[top_in, 0],  top[top_in, 1], marker='o', c='m', alpha=0.75, s=10.0)

    bottom_out = bottom[:, 1] < -1.0
    bottom_in  = bottom[:, 1] > -1.0
    plt.scatter(bottom[bottom_out, 0], bottom[bottom_out, 1], marker='o', c='g', alpha=0.75, s=10.0)
    plt.scatter(bottom[bottom_in, 0],  bottom[bottom_in, 1], marker='o', c='m', alpha=0.75, s=10.0)

    plt.axis('equal')
    plt.savefig(filename)
    plt.close()


def save_scalar_histogram(filename, scalar):
    scalar = torch.min(scalar.cpu(), torch.tensor(20.0))
    scalar = tensor_to_numpy(scalar)
    plt.figure(figsize=(10, 10), dpi=60)
    n, bins, patches = plt.hist(scalar, 1000, range=(0,20), facecolor='g', alpha=0.75)
    plt.axvline(np.median(scalar), color='r', linewidth=1)
    plt.savefig(filename)
    plt.close()


def save_grad_barplot(filename, max_grads, avg_grads, layers):
    plt.figure(figsize=(10, 10), dpi=60)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(avg_grads)), avg_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(avg_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(avg_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(avg_grads))
    plt.ylim(bottom = -0.001) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(filename)
    plt.close()


def rreplace(s, old, new):
    li = s.rsplit(old, 1)
    return new.join(li)