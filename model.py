import tensorflow as tf
import os
import logging
from config import model_params

class Models():
    def __init__(self):
        self.image_shape = (model_params.image_size, model_params.image_size, 3)
        self.batch_size = model_params.batch_size
        self.model_name = 'MobileNetV2'
        self.learning_rate = 0.05

    def preTrainingModel(self):
        if self.model_name == 'MobileNetV2':
            self.base_model = tf.keras.applications.MobileNetV2(input_shape=self.image_shape,
                                                        include_top=False,
                                                        weights='imagenet')
        else:
            logging.info('Plz select models we have')
            os._exit(0)


        self.base_model.trainable = False
    

    def core(self):
        feature_batch = self.base_model(self.batch_size)

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)

        prediction_layer = tf.keras.layers.Dense(1)
        prediction_batch = prediction_layer(feature_batch_average)  


        self.model = tf.keras.Sequential([
            self.base_model,
            global_average_layer,
            prediction_layer
        ])

        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=self.learning_rate),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

