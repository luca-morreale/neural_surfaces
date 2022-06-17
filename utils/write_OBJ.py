
def writeOBJ(filename, V, F, UV, N):

    with open(filename, 'w') as stream:
        for i in range(V.shape[0]):
            stream.write('v {} {} {}\n'.format(V[i,0], V[i,1], V[i,2]))
            if UV is not None:
                stream.write('vt {} {}\n'.format(UV[i,0], UV[i,1]))
            if N is not None:
                stream.write('vn {} {} {}\n'.format(N[i,0], N[i,1], N[i,2]))

        for f in F:
            stream.write('f {}/{}/{} {}/{}/{} {}/{}/{}\n'.format(f[0]+1, f[0]+1, f[0]+1,
                                        f[1]+1, f[1]+1, f[1]+1, f[2]+1, f[2]+1, f[2]+1))
