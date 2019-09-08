import os
import logging
import numpy as np
from shutil import copyfile
from config import path_params

def train_validation_test_split(file_path):
    try:
        os.mkdir(file_path.train_data_path)
        os.mkdir(file_path.valid_data_path)
        os.mkdir(file_path.test_data_path)
    except:
        logging.info('Path already exist')

    for categories in os.listdir(file_path.data_dir + 'train'):
        category_path = file_path.data_dir + 'train/' + str(categories)
        
        try:
            os.mkdir(file_path.train_data_path + str(categories))
            os.mkdir(file_path.valid_data_path + str(categories))
            os.mkdir(file_path.test_data_path + str(categories))
        except:
            logging.info('Path already exist')

        for images in os.listdir(category_path):
            # Ramdomly decide where an image will belong: train, valid, or test
            data_path = np.random.choice([file_path.train_data_path,
                                          file_path.valid_data_path, 
                                          file_path.test_data_path])
            copyfile(category_path+'/'+images, data_path+str(categories)+'/'+str(images))
            
            logging.info('From: {0}'.format(category_path+'/'+images))
            logging.info('To: {0}'.format(data_path+'/'+str(categories)+str(images)))

    logging.info('Split successful')


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    train_validation_test_split(path_params)
    os._exit(0)
