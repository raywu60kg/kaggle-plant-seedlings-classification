import os
import logging
import numpy as np
from shutil import copyfile
from config import path_params, model_params


np.random.seed(0)
path_order = [
    path_params.train_data_path, 
    path_params.valid_data_path, 
    path_params.test_data_path
]


def dataSplitor(file_path):
    try:
        os.mkdir(TRAIN_DATA_PATH)
        os.mkdir(VALID_DATA_PATH)
        os.mkdir(TEST_DATA_PATH)
    except:
        logging.info('Path already exist')
    
    for categories in os.listdir(file_path + '/train'):
        category_path = file_path+'/train'+'/'+str(categories)       
        try:
            os.mkdir(TRAIN_DATA_PATH+'/'+str(categories))
            os.mkdir(VALID_DATA_PATH+'/'+str(categories))
            os.mkdir(TEST_DATA_PATH+'/'+str(categories))
        except:
            logging.info('Path already exist')

        for images in os.listdir(category_path):
            # Ramdomly decide train, valid and test
            ramdom_list = np.random.multinomial(1, 
                [path_params.train_data_path,path_params.valid_data_path, path_params.test_data_path], 
                size=1).reshape(-1)
            data_path = path_order[np.argmax(ramdom_list)]
            copyfile(category_path+'/'+images, data_path+'/'+str(categories)+str(images))


            logging.info('From: {0}'.format(category_path+'/'+images))
            logging.info('To: {0}'.format(data_path+'/'+str(categories)+str(images)))            
            
    logging.info('Split successful')


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    dataSplitor(model_params.image_path)
    os._exit(0)