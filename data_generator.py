#trial: edit here
import tensorflow as tf
import logging
from config import path_params, model_params
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class InputPipeline():
    def __init__(self, path_params, data_params):
        self.train_path = path_params.train_path
        self.valid_path = path_params.valid_path
        self.test_path = path_params.test_path
        self.image_size = model_params.image_size
        self.batch_size = model_params.batch_size


    def buildTrainData(self):
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)
        self.train_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            class_mode='categorical')


    def buildValidData(self):
        valid_datagen = ImageDataGenerator(rescale=1./255)
        self.valid_generator = valid_datagen.flow_from_directory(
            self.valid_path,
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            class_mode='categorical')


    def buildTestData(self):
        test_datagen = ImageDataGenerator(rescale=1./255)
        self.test_generator = test_datagen.flow_from_directory(
            self.test_path,
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            class_mode='categorical')

