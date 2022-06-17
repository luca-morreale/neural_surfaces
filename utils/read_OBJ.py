
import numpy as np


def readOBJ(filename):

    V = []
    F = []
    UV = []
    TF = []
    N = []

    with open(filename, 'r') as stream:
        for line in stream.readlines():
            els = line.split(' ')
            els = list(filter(None, els))
            if els[0] == 'v':
                V.append([float(els[1]), float(els[2]), float(els[3])])
            elif els[0] == 'vt':
                UV.append([float(els[1]), float(els[2])])
            elif els[0] == 'vn':
                N.append([float(els[1]), float(els[2]), float(els[3])])
            elif els[0] == 'f':
                face = []
                face_uv = []
                #face

                for Fi in els[1:]:
                    F_els = Fi.split('/')
                    face.append(int(F_els[0]))
                    if len(F_els) > 1:
                        if len(F_els[1]) > 0:
                            face_uv.append(int(F_els[1]))
                        else:
                            face_uv.append(int(F_els[0]))

                F.append(face)
                TF.append(face_uv)

    V = np.array(V)
    F = np.array(F) - 1
    UV = np.array(UV)
    TF = np.array(TF) - 1
    N = np.array(N)

    return V, F, UV, TF, N
