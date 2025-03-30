import glob, os, sys
sys.path.insert(0, os.getcwd())
import utils.src.data_utils as data_utils


VOID1500_ROOT_DIRPATH = os.path.join('training', 'void_1500', 'random_patches')

VOID1500_TRAIN_IMAGE_DIRPATH = os.path.join(VOID1500_ROOT_DIRPATH, 'images')
VOID1500_TRAIN_DEPTH_DIRPATH = os.path.join(VOID1500_ROOT_DIRPATH, 'depth')

TRAIN_REF_DIRPATH = os.path.join('training', 'void_1500', 'random_patches')

VOID1500_TRAIN_IMAGE_FILEPATH = os.path.join(TRAIN_REF_DIRPATH, 'void_image_patches.txt')
VOID1500_TRAIN_DEPTH_FILEPATH = os.path.join(TRAIN_REF_DIRPATH, 'void_depth_patches.txt')


def setup_dataset():
    '''
    Fetches all file paths to training and validation images and writes them to text files
    '''

    # Use glob to grab all image file paths in training image directory
    train_image_paths = glob.glob(f"{VOID1500_TRAIN_IMAGE_DIRPATH}/*.png")
    train_depth_paths = glob.glob(f"{VOID1500_TRAIN_DEPTH_DIRPATH}/*.png")
    

    # Sort training and validation all paths
    train_image_paths = sorted(train_image_paths)
    train_depth_paths = sorted(train_depth_paths)

    # Create directories for TRAIN_REF_DIRPATH and VAL_REF_DIRPATH
    os.makedirs(TRAIN_REF_DIRPATH, exist_ok=True)

    # Write training and validation image paths to file
    print('Storing {} training image file paths into: {}'.format(
        len(train_image_paths),
        VOID1500_TRAIN_IMAGE_FILEPATH))

    # Write training paths to file (VOID1500_TRAIN_IMAGE_FILEPATH) using data_utils.write_paths
    data_utils.write_paths(VOID1500_TRAIN_IMAGE_FILEPATH, train_image_paths)

    print('Storing {} depth image file paths into: {}'.format(
        len(train_depth_paths),
        VOID1500_TRAIN_DEPTH_FILEPATH))

    # Write validation paths to file (VOID1500_VAL_IMAGE_FILEPATH) using data_utils.write_paths
    data_utils.write_paths(VOID1500_TRAIN_DEPTH_FILEPATH, train_depth_paths)


if __name__ == '__main__':

    setup_dataset()
