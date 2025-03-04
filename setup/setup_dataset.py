import glob, os, sys
sys.path.insert(0, os.getcwd())
import utils.src.data_utils as data_utils

TRAIN_REF_DIRPATH = os.path.join('training', 'void_1500')
VAL_REF_DIRPATH = os.path.join('validation', 'void_1500')

VOID_1500_TRAIN_IMAGE_FILEPATH = os.path.join(TRAIN_REF_DIRPATH, 'void_1500_train_image.txt')
VOID_1500_TRAIN_GROUND_TRUTH_FILEPATH = os.path.join(TRAIN_REF_DIRPATH, 'void_1500_train_ground_truth.txt')


def setup_dataset():
    train_image_filepath = 'data/void_1500/train_image.txt'
    train_sparse_depth_filepath = 'data/void_1500/train_sparse_depth.txt'
    train_ground_truth_filepath = 'data/void_1500/train_ground_truth.txt'
    train_validity_map_filepath = 'data/void_1500/train_validity_map.txt'
    
    train_sparse_depth_paths = data_utils.read_paths(train_sparse_depth_filepath, prefix='data/')
    train_validity_map_paths = data_utils.read_paths(train_validity_map_filepath, prefix='data/')
    train_ground_truth_paths = data_utils.read_paths(train_ground_truth_filepath, prefix='data/')
    train_image_paths = data_utils.read_paths(train_image_filepath, prefix='data/')
    
    os.makedirs(TRAIN_REF_DIRPATH, exist_ok=True)
    os.makedirs(VAL_REF_DIRPATH, exist_ok=True)
    
    print('Storing {} training image file paths into: {}'.format(
        len(train_image_paths),
        VOID_1500_TRAIN_IMAGE_FILEPATH))
    
    data_utils.write_paths(VOID_1500_TRAIN_IMAGE_FILEPATH, train_image_paths)
    
    print('Storing {} training image file paths into: {}'.format(
        len(train_ground_truth_paths),
        VOID_1500_TRAIN_GROUND_TRUTH_FILEPATH))
    
    data_utils.write_paths(VOID_1500_TRAIN_GROUND_TRUTH_FILEPATH, train_ground_truth_paths)
    
if __name__ == '__main__':
    setup_dataset()