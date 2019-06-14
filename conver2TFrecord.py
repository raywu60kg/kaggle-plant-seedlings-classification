import tensorflow as tf
import os
import logging
from config import IMAGE_PATH, TFRECORD_PATH


def converter(file_path):
    """conver training file"""
    os.mkdir(TFRECORD_PATH)
    for categories in os.listdir(file_path + '/train'):
        category_path = file_path+'/train'+'/'+str(categories)
        for images in os.listdir(category_path):
            # logging.info(categories)
            # logging.info(images)
            image_path = category_path+'/'+str(images)
            # print(imagePath)
            filename_queue = tf.train.string_input_producer(
                tf.train.match_filenames_once(image_path))
            

    logging.info('conver successful')


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    converter(IMAGE_PATH)
    os._exit(0)
