import tensorflow as tf


# paths
DATA_DIR = '/Users/apple/Documents/git-repos/data/kaggle_plant_data/'
TFRECORD_PATH = DATA_DIR + 'tfrecords/'
TRAIN_DATA_PATH = DATA_DIR + 'training/'
VALID_DATA_PATH = DATA_DIR + 'validation/'
TEST_DATA_PATH = DATA_DIR + 'testing/'

# data params
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
    data_dir = DATA_DIR,
    tfrecord_path = TFRECORD_PATH,
    train_data_path = TRAIN_DATA_PATH,
    valid_data_path = VALID_DATA_PATH,
    test_data_path = TEST_DATA_PATH
)

train_params = tf.contrib.training.HParams(
    batch_size = BATCH_SIZE,
    train_ratio = TRAIN_RATIO,
    valid_ratio = VALID_RATIO,
    test_ratio = TEST_RATIO,
    image_size = IMAGE_SIZE,
    categories_name = CATEGORIES_NAME,
    categories_num = CATEGORIES_NUM,
    seed = 8888,
    epochs = 1
)

model_params = tf.contrib.training.HParams(
    image_size = train_params.image_size,
    learning_rate = 0.0001
)

