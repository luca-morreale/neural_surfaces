
import os


def is_mesh_file(path):
    ext = path[-3:]
    return ext == 'ply' or ext == 'obj' or ext == 'off'

def is_pth_file(path):
    ext = path[-3:]
    return ext == 'pth'

def list_all_files(path, mesh=True, binary=False):

    if os.path.isfile(path):
        files_list = [ path ]
    else:
        files_list = [ os.path.join(path, file) for file in os.listdir(path) ]
    files_list.sort()

    if mesh:
        files_list = [file for file in files_list if is_mesh_file(file)]
    if binary:
        files_list = [file for file in files_list if is_pth_file(file)]

    return files_list
