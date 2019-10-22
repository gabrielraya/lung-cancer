
from settings import copy_all

# Tests

# Copy all svs files to new_folder removing folder afterwards
data_folder = '../../images/'
new_folder = '../../images_renamed/'
copy_all(data_folder, new_folder, ext='.svs', rm=True)