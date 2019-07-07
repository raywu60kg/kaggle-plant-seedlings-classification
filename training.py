from tf.keras.preprocessing.image import ImageDataGenerator
from data_generator import InputPipeline

train_generator = zip(image_generator, mask_generator)
model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)