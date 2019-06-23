import tensorflow as tf
import os
import logging
from config import IMAGE_PATH, IMAGE_SIZE, TFRECORD_PATH, CATEGORIES_NAME, CATEGORIES_NUM
from PIL import Image


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def labelMapping(category, category_name, category_num):
    labels = [0]*category_num
    counter = 0
    for cat in category_name:
        if cat == category:
            labels[counter] = 1
        counter += 1
    return labels


# def encode_to_tfrecords(data_path, name, rows=24, cols=16):
#     folders = os.listdir(data_path)
#     folders.sort()
#     numclass = len(folders)
#     i = 0
#     npic = 0
#     writer = tf.python_io.TFRecordWriter(name)
#     for floder in folders:
#         path = data_path+"/"+floder
#         img_names = glob.glob(os.path.join(path, "*.bmp"))
#         for img_name in img_names:
#             img_path = img_name
#             img = Image.open(img_path).convert('P')
#             img = img.resize((cols, rows))
#             img_raw = img.tobytes()
#             labels = [0]*34
#             labels[i] = 1
#             example = tf.train.Example(features=tf.train.Features(feature={
#                 'image_raw': _byte_feature(img_raw),
#                 'label': _int64_feature(labels)}))
#             writer.write(example.SerializeToString())
#             npic = npic+1
#         i = i+1
#     writer.close()
#     print(npic)


def converter(file_path):
    """conver training file"""
    try:
        os.mkdir(TFRECORD_PATH)
        logging.info("Create out put path at {0}".format(TFRECORD_PATH))
    except:
        logging.info("Out put path already exist at {0}".format(TFRECORD_PATH))
    os.chdir(TFRECORD_PATH)
    writer = tf.python_io.TFRecordWriter('test.tfrecords')
    for categories in os.listdir(file_path + '/train'):
        category_path = file_path+'/train'+'/'+str(categories)       
        numclass = len(categories)
        counter = 0
        for images in os.listdir(category_path):
            img = Image.open(category_path+'/'+images).resize((128,128))
            img_raw = img.tobytes()
            labels = labelMapping(categories, CATEGORIES_NAME, CATEGORIES_NUM)
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _byte_feature(img_raw),
                'label': _int64_feature(labels)}))
            writer.write(example.SerializeToString())
            logging.info('convered {0}'.format(images))
    writer.close()
    logging.info('conver successful')


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    converter(IMAGE_PATH)
    os._exit(0)
