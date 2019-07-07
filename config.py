import tensorflow as tf


# Pathes
IMAGE_PATH = '/kaggle_data'
TFRECORD_PATH = '/kaggle_data/tfrecords/'
TRAIN_DATA_PATH = '/kaggle_data/train_split/'
VALID_DATA_PATH = '/kaggle_data/valid_split/'
TEST_DATA_PATH = '/kaggle_data/test_split/'

# Data params
BATCH_SIZE = 12
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15
IMAGE_SIZE = 128

CATEGORIES_NAME = [
    "Charlock",
    "Maize",
    "Black-grass",
    "Fat Hen",
    "Loose Silky-bent",
    "Cleavers",
    "Common Chickweed",
    "Small-flowered Cranesbill",
    "Sugar beet",
    "Scentless Mayweed",
    "Common wheat",
    "Shepherds Purse"
]
CATEGORIES_NUM = len(CATEGORIES_NAME)

path_params = tf.contrib.training.HParams(
    image_path = IMAGE_PATH,
    tfrecord_path = TFRECORD_PATH,
    train_data_path = TRAIN_DATA_PATH,
    valid_data_path = VALID_DATA_PATH,
    test_data_path = TEST_DATA_PATH
)

model_params = tf.contrib.training.HParams(
    batch_size = BATCH_SIZE,
    train_ratio = TRAIN_RATIO,
    valid_ratio = VALID_RATIO,
    test_ratio = TEST_RATIO,
    image_size = IMAGE_SIZE,
    categories_name = CATEGORIES_NAME,
    categories_num = CATEGORIES_NUM    
)
