
from pathlib import Path
from shutil import copyfile

from .mixin import Mixin


class CollectionVisualizationCheckpointing(Mixin):

    def create_visualization_page(self, name_filter, checkpoint_dir, prefix=''):

        ### List all files inside folder
        parent_path = Path(checkpoint_dir) / 'mesh'
        files_list  = parent_path.glob('*' + name_filter + '*.ply') # filter names
        files_list  = [ './'+str(el.relative_to(parent_path)) for el in files_list ]
        files_list.sort()

        ### Copy html visualization file to new location
        out_folder = self.compose_out_folder(checkpoint_dir, ['mesh'])
        filename = '{}/{}visualization.html'.format(out_folder, prefix)
        copyfile('./html/visualization.html', filename)

        ### Replace the key string in the file with the list of files
        with open(filename, 'r') as file :
            filedata = file.read()

        ### Replace the target string
        filedata = filedata.replace('replace_here', '"' + '","'.join(files_list) + '"')

        ### Write the file out again
        with open(filename, 'w') as file:
            file.write(filedata)
