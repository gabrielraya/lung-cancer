import os
import shutil


def check_file_exists(filename):
    """ Test whether a path exists.  Returns False for broken symbolic links
    Replace os.path.exits functionality as it does not work in the cluster.
    """
    try:
        f = open(filename, 'r')
        f.close()
        return True
    except IOError:
        return False


def get_file_list(path, ext=''):
    return sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)])


def delete_entire_directory(trash_dir):
    try:
        shutil.rmtree(trash_dir)
    except OSError as e:
        print(f'Error: {trash_dir} : {e.strerror}')


def copy_files(src, dst, folder_id, ext='', rm=False):
    """
    Copy All files in a given folder to dst renamed as folderId_fileName
    Dont copy files already in the folder
    :param src: folder path to be copied
    :param dst: destination folder path
    :param folder_id: patient id folder
    :param ext: type of files to be copied
    :param rm:  If True remove the source folder when copy is finished
    :return:
    """
    # Set file path for each patient
    pid_path = os.path.join(src, folder_id)
    # Read files for each patient folder
    pid_files = os.listdir(pid_path)
    # Iterates over each file in folderId
    for file_name in pid_files:
        if file_name.endswith(ext):
            # renames rename as folderId_fileName
            folder_path = os.path.join(src, folder_id)
            s = os.path.join(folder_path, file_name)
            d = os.path.join(dst, '_'.join([folder_id, file_name]))

        if not os.path.exists(d):
            shutil.copy(s, d)
            print("Copying File: " + s)
    if rm:
        delete_entire_directory(folder_path)


def copy_all(src, dst, ext='', rm=False):
    """
    Check if give folder exists otherwise creates it
    :param src: folder path to be copied
    :param dst: destination folder path
    :param ext: type of files to be copied
    :return: Copies all the files from src to dst
    """
    if not os.path.exists(dst):
        os.makedirs(dst)
    # read patient id (pid) folders
    folder_list = os.listdir(src)
    # Iterates over patient id folder list
    for i in range(len(folder_list)):
        folder_id = folder_list[i]
        print("Copying PatientID: " + folder_id)
        copy_files(src, dst, folder_id, ext='.svs', rm=rm)


if __name__ == '__main__':

    # Tests

    # Copy all svs files to new_folder removing folder afterwards
    data_folder = '../../images/'
    new_folder = '../../images_renamed/'
    copy_all(data_folder, new_folder, ext='.svs', rm=True)
