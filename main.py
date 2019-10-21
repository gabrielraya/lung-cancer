import os
import shutil


def delete_entire_directory(trash_dir):
    try:
        shutil.rmtree(trash_dir)
    except OSError as e:
        print(f'Error: {trash_dir} : {e.strerror}')


def copy_tree(src, dst, ext='', rm=False):
    """
    Copy All files in a given folder to dst
    Check if give folder exists otherwise creates it
    Dont copy files already in the folder
    :param src: folder path to be copied
    :param dst: destination folder path
    :param ext: type of files to be copied
    :param rm:  If True remove the source folder when copy is finished
    """
    if not os.path.exists(dst):
        os.makedirs(dst)
    for file_name in os.listdir(src):
        if file_name.endswith(ext):
            s = os.path.join(src, file_name)
            d = os.path.join(dst, file_name)
        if not os.path.exists(d):
            shutil.copy(s, dst)
        else:
            print("path exists already!")
            print(d)
    if rm:
        delete_entire_directory(src)


def copy_files(src, dst, ext=''):
    """
    :param src: folder path to be copied
    :param dst: destination folder path
    :param ext: type of files to be copied
    :return: Copies all the files from src to dst
    """
    folder_list = os.listdir(src)
    # Test using the function
    for i in range(len(folder_list)):
        # Update path
        src_folder = os.path.join(src, folder_list[i])
        print(src_folder )
        # Move all files in the given folder to dst
        copy_tree(src_folder, dst, ".svs")

# Test
source = '../data/103302/'
#source = '../data/100147/'
destination = '../new_images/'
copy_tree(source, destination, ".svs")

src = '../new_images/'
dst = '../new_images2/'
copy_tree(src, dst, ".svs", rm=True)

# Test deleting entire folder
trash = '../new_images'
delete_entire_directory(trash)

# Copy all data
source = '../data/'
destination = '../new_images/'
copy_files(source, destination, '.svs')



