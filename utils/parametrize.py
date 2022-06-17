
import numpy as np
import igl


def parametrize(V, F, method='slim', it=100, bnd_shape='circle'):
    ## Find the open boundary
    bnd = igl.boundary_loop(F)

    ## Map the boundary to a circle, preserving edge proportions
    bnd_uv = igl.map_vertices_to_circle(V, bnd)
    if bnd_shape == 'square':
        bnd_uv = map_cirlce_to_square(bnd_uv)

    ## Harmonic parametrization for the internal vertices
    uv_init = igl.harmonic_weights(V, F, bnd, bnd_uv, 1)

    if len(igl.flipped_triangles(uv_init, F).shape) > 0:
        uv_init = igl.harmonic_weights_uniform_laplacian(F, bnd, bnd_uv, 1) # use uniform laplacian

    bnd_constrain_weight = 1.0e35 if bnd_shape != 'free' else 0.0

    if method == 'slim':
        return slim(V, F, uv_init, bnd, bnd_uv, it, bnd_constrain_weight)
    elif method == 'arap':
        return arap(V, F, uv_init, bnd, bnd_uv, it)
    elif method == 'lscm':
        return lscm(V, F, uv_init, bnd, bnd_uv, it)


def slim(V, F, uv_init, bnd, bnd_uv, it, bnd_constrain_weight):
    slim = igl.SLIM(V, F, uv_init, bnd, bnd_uv, igl.SLIM_ENERGY_TYPE_SYMMETRIC_DIRICHLET, bnd_constrain_weight)
    print(f'SLIM initial energy {slim.energy()}')
    count = 0
    slim.solve(it)
    while slim.energy() > 100.0:
        slim.solve(it)
        count += 1
        if count > 200:
            break
    # slim.solve(it)
    uva = slim.vertices()
    print(f'SLIM final energy {slim.energy()}')
    return uva


def arap(V, F, uv_init, bnd, bnd_uv, it):
    arap = igl.ARAP(V, F, 2, np.zeros(0))
    uva = arap.solve(np.zeros((0, 0)), uv_init)
    return uva


def lscm(V, F, uv_init, bnd, bnd_uv, it):
    _, uva = igl.lscm(V, F, bnd, bnd_uv)
    return uva


def map_cirlce_to_square(bnd_uv):
    u = bnd_uv[:,0].reshape(-1,1)
    v = bnd_uv[:,1].reshape(-1,1)
    u2 = np.power(u, 2)
    v2 = np.power(v, 2)
    sqrt_2 = np.sqrt(2)
    x = 0.5 * np.sqrt(2.0+2.0*sqrt_2*u+u2-v2) - 0.5* np.sqrt(2-2*sqrt_2*u+u2-v2)
    y = 0.5 * np.sqrt(2.0+2.0*sqrt_2*v-u2+v2) - 0.5* np.sqrt(2-2*sqrt_2*v-u2+v2)
    xy = np.concatenate([x,y], axis=1)
    return xy
