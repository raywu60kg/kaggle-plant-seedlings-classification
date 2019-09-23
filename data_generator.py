from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataPipeline():
    def __init__(self, path_params, train_params):
        self.train_path = path_params.train_data_path
        self.valid_path = path_params.valid_data_path
        self.test_path  = path_params.test_data_path
        self.image_size = train_params.image_size
        self.batch_size = train_params.batch_size
        self.seed       = train_params.seed


    def build_training_data(self):
        train_datagen = ImageDataGenerator(
                rescale = 1./255,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True)
        return train_datagen.flow_from_directory(
                self.train_path,
                target_size = (self.image_size, self.image_size),
                batch_size = self.batch_size,
                seed = self.seed,
                class_mode = 'categorical',
                shuffle = True)

    def build_validation_data(self):
        validation_datagen = ImageDataGenerator(rescale=1./255)
        return validation_datagen.flow_from_directory(
                self.valid_path,
                target_size = (self.image_size, self.image_size),
                batch_size = self.batch_size,
                seed = self.seed,
                class_mode = 'categorical',
                shuffle = True)


    def build_testing_data(self):
        test_datagen = ImageDataGenerator(rescale=1./255)
        return test_datagen.flow_from_directory(
                self.test_path,
                target_size = (self.image_size, self.image_size),
                seed  = self.seed,
                batch_size = self.batch_size, #1
                class_mode = 'categorical',#None,#'categorical',
                shuffle = False)
