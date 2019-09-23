from tensorflow.keras import applications, optimizers
from tensorflow.keras.layers import Flatten, Dense, Activation, Conv2D, Dropout
from tensorflow.keras.models import Model

from config import model_params

class DNN():
    def __init__(self, model_params):
        self.image_shape = (model_params.image_size, model_params.image_size, 3)
        self.learning_rate = model_params.learning_rate

    def build_model(self):

        inception = applications.InceptionResNetV2(
            weights = 'imagenet', 
            include_top=False, 
            input_shape = self.image_shape )

        # Freeze layers
        for layer in inception.layers:
            layer.trainable = False

        # Create output
        x = inception.output
        x = Flatten()(x)
        x = Dense(512, activation="tanh")(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.5)(x)
        predictions = Dense(12, activation="softmax")(x)

        # creating the final model
        self.model = Model(inputs= inception.input,
                            outputs= predictions)
        # compile the model
        self.model.compile(loss = "categorical_crossentropy",
                            optimizer = optimizers.Adam(lr=self.learning_rate),
                            metrics=["accuracy"])
