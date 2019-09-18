from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_generator import InputPipeline
import tensorflow
from config import path_params, model_params
import model

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    rescale=1./255)

image_generator = tensorflow.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
mask_generator = tensorflow.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

image_generator = image_generator.flow_from_directory(path_params.train_data_path,
        color_mode="grayscale",
        classes=['image'],
        class_mode=None,
        batch_size=5)

mask_generator = mask_generator.flow_from_directory(path_params.train_data_path,
        color_mode="grayscale",
        classes=['label'],
        class_mode=None,
        batch_size=5)


train_generator = zip(image_generator, mask_generator)
model = model.Models()

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)
