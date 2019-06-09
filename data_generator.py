from plant-seedlings-classification.config import NUM_CORES, IMAGE_PATH, BATCH_SIZE, VALID_RATIO, TRAIN_RATIOi, IMAGE_SIZE
import tensorfow as tf
import logging
import random
from PTL import Image


class inputPipeline():
    def __init__(self, input_pipeline_hps):
        self.graph = tf.Graph()
        self.num_cores = NUM_CORES
        sess_config = tf.ConfigProto(device_count={"GPU":self.num_cores},
                                 inter_op_parallelism_threads=8,
                                 intra_op_parallelism_threads=8)
        sess_config.gpu_options.allow_growth = True
        self.image_path = IMAGE_PATH
        self.batch_size = BATCH_SIZE
        self.valid_ratio = VALID_RATIO
        self.train_ratio = TRAIN_RATIO
        self.sess = tf.Session(config=sess_config, graph=self.graph)
        
        # Build the full_data tensor
        with self.graph.as_default():
            self.__build_data_tensor()
        

    def buildTrainData(self):
        with self.graph.as_default():
            train_dataset = self.train_dataset.repeat()
            train_dataset = train_dataset.batch(self.batch_size)
            train_dataset = train_dataset.prefetch(self.batch_size)

            train_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
            train_init_op = train_iterator.make_initializer(train_dataset)
            self.train_next_batch = train_iterator.get_next()

            self.sess.run(train_init_op)
            
    def buildValidData(self):
        with self.graph.as_default():
            valid_dataset = self.valid_dataset.repeat()
            valid_dataset = valid_dataset.batch(self.batch_size) 
            valid_dataset = valid_dataset.prefetch(self.batch_size)

            valid_iterator = tf.data.Iterator.from_structure(valid_dataset.output_types, valid_dataset.output_shapes)
            valid_init_op = valid_iterator.make_initializer(valid_dataset)
            self.valid_next_batch = valid_iterator.get_next()

            self.sess.run(valid_init_op)

    def extractImage(self):
        

    def __buildData(self):

        full_dataset = tf.data.TFRecordDataset(self.input_pipeline_hps.data_file)
        
        train_size = int(self.train_ratio * DATASET_SIZE)
        valid_size = DATASET_SIZE - train_size
        
        full_dataset = full_dataset.shuffle(buffer_size=DATASET_SIZE)
        full_dataset = full_dataset.map(extractImage, num_parallel_calls=self.input_pipeline_hps.num_cores)
        self.train_dataset = full_dataset.take(train_size)
        self.valid_dataset = full_dataset.skip(train_size)

            
    def getNextTrain(self):
        with self.graph.as_default():
            return self.sess.run(self.train_next_batch)
    
    def getNextValid(self):
        with self.graph.as_default():
            return self.sess.run(self.valid_next_batch)
        
    def __del__(self):
        self.close()
        
    def close(self):
        self.sess.close()

