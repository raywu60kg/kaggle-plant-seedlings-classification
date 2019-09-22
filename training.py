import time
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import model
from data_generator import DataPipeline
from config import path_params, train_params, model_params

# Save the model according to the conditions  
filepath = "model_weight-{epoch:02d}-{loss:.4f}-m1.hdf5"

checkpoint = ModelCheckpoint(filepath, 
                             monitor = 'val_acc', 
                             verbose = 1, 
                             save_best_only = True, 
                             save_weights_only = False, 
                             mode = 'auto', 
                             period = 1)

early = EarlyStopping(monitor = 'val_acc', 
                      min_delta = 5, 
                      patience = 10, 
                      verbose = 1, 
                      mode = 'auto')

time_start = time.time()

data_pipeline = DataPipeline(path_params, train_params)
train_generator = data_pipeline.build_training_data()
validation_generator = data_pipeline.build_validation_data()
test_generator = data_pipeline.build_testing_data()

STEP_SIZE_TRAIN = train_generator.n//data_pipeline.batch_size
STEP_SIZE_VALID = validation_generator.n//data_pipeline.batch_size
STEP_SIZE_TEST = test_generator.n//data_pipeline.batch_size #1

#model = model.Models(model_params)
final_model = model.final_model
#final_model = model.build_model()

final_model.fit_generator(
        train_generator,
        epochs = train_params.epochs,
        steps_per_epoch = STEP_SIZE_TRAIN,
        validation_data = validation_generator,
        validation_steps = STEP_SIZE_VALID)

final_model.evaluate_generator(
        generator=validation_generator,
        steps=STEP_SIZE_VALID)


#test_generator.reset()
final_model.evaluate_generator(
        generator=test_generator,
        steps=STEP_SIZE_TEST)

"""
test_generator.reset()
pred=model.predict_generator(
        test_generator,
        steps=STEP_SIZE_TEST,
        verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
"""

time_elapsed = time.time() - time_start

# Record time used
print(time_elapsed)