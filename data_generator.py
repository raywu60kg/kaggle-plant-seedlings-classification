from plant-seedlings-classification.config import IMAGE_PATH
import tensorfow as tf

class inputPipeline():
    def __init__(self, input_pipeline_hps):
        self.input_pipeline_hps = input_pipeline_hps
        self.graph = tf.Graph()

        sess_config = tf.ConfigProto(device_count={"GPU":self.input_pipeline_hps.num_cores},
                                 inter_op_parallelism_threads=8,
                                 intra_op_parallelism_threads=8)
        logging.info('Controlling the use of GPU')
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config, graph=self.graph)
        
        # Build the full_data tensor
        with self.graph.as_default():
            self.__build_data_tensor()
        

    def build_train_data_tensor(self):
        with self.graph.as_default():
            train_dataset = self.train_dataset.repeat()
            train_dataset = train_dataset.batch(self.input_pipeline_hps.batch_size)
            train_dataset = train_dataset.prefetch(self.input_pipeline_hps.prefetch_size)

            train_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
            train_init_op = train_iterator.make_initializer(train_dataset)
            self.train_next_batch = train_iterator.get_next()

            self.sess.run(train_init_op)
            
    def build_valid_data_tensor(self):
        with self.graph.as_default():
            valid_dataset = self.valid_dataset.repeat()
            valid_dataset = valid_dataset.batch(self.input_pipeline_hps.batch_size) 
            valid_dataset = valid_dataset.prefetch(self.input_pipeline_hps.prefetch_size)

            valid_iterator = tf.data.Iterator.from_structure(valid_dataset.output_types, valid_dataset.output_shapes)
            valid_init_op = valid_iterator.make_initializer(valid_dataset)
            self.valid_next_batch = valid_iterator.get_next()

            self.sess.run(valid_init_op)
            
    def build_test_data_tensor(self):
        with self.graph.as_default():
            test_dataset = self.test_dataset.repeat()
            test_dataset = test_dataset.batch(self.input_pipeline_hps.batch_size) 
            test_dataset = test_dataset.prefetch(self.input_pipeline_hps.prefetch_size)

            test_iterator = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)
            test_init_op = test_iterator.make_initializer(test_dataset)
            self.test_next_batch = test_iterator.get_next()

            self.sess.run(test_init_op)           
            

    def __build_data_tensor(self):
        
        DATASET_SIZE = 0
        for fn in self.input_pipeline_hps.data_file:
            for _ in tf.python_io.tf_record_iterator(fn):
                 DATASET_SIZE += 1
        logging.info('TOTAL DATASET_SIZE = %d', DATASET_SIZE)
        
        random.seed(self.input_pipeline_hps.random_seed)
        random.shuffle(self.input_pipeline_hps.data_file)

        def extract_tfrecords(data_record):
            features = {
                'label':tf.FixedLenFeature([1], tf.int64),
                'iab':tf.FixedLenFeature([1, IAB_NUM], tf.int64),
                'dmp':tf.FixedLenFeature([1, DMP_LABEL_NUM], tf.int64),
                'clientlabels':tf.FixedLenFeature([1, CLIENT_LABEL_NUM], tf.int64),
                'site':tf.FixedLenFeature([1], tf.int64),
                'whex':tf.FixedLenFeature([1], tf.int64),
                'hr':tf.FixedLenFeature([1], tf.int64),
                'make':tf.FixedLenFeature([1], tf.int64)
            }
            
            sample = tf.parse_single_example(data_record, features)
            tf.reshape(sample['iab'],[tf.shape(sample['iab'])[0],-1])
            tf.reshape(sample['dmp'],[tf.shape(sample['dmp'])[0],-1])
            y = sample['label']

            if self.input_pipeline_hps.to_ohe:
                site_ohe = tf.cast(tf.one_hot(sample['site'] , depth=NAMEINT_MAP_NUM,   on_value=1, axis=1), dtype=tf.int64)
                whex_ohe = tf.cast(tf.one_hot(sample['whex'] , depth=WHEX_NUM,          on_value=1, axis=1), dtype=tf.int64)
                hr_ohe   = tf.cast(tf.one_hot(sample['hr']   , depth=HOUR_NUM,          on_value=1, axis=1), dtype=tf.int64)
                make_ohe = tf.cast(tf.one_hot(sample['make'] , depth=MAKER_NUM,         on_value=1, axis=1), dtype=tf.int64)

                x_ohe = tf.concat([sample['iab'], sample['dmp'], site_ohe, whex_ohe, hr_ohe, make_ohe],axis=1)
                return (x_ohe, y)  # x_ohe: array, y:int
            
            else:
                return (sample, y) # sample: dict, y: int 
        
        full_dataset = tf.data.TFRecordDataset(self.input_pipeline_hps.data_file)
        
        if self.input_pipeline_hps.is_test:
            test_ratio = 1.0 - self.input_pipeline_hps.train_ratio - self.input_pipeline_hps.valid_ratio 
            train_size = int(self.input_pipeline_hps.train_ratio * DATASET_SIZE)
            valid_size = int(self.input_pipeline_hps.valid_ratio * DATASET_SIZE)
            test_size  = int(test_ratio * DATASET_SIZE)
            logging.info('train_size: %d valid_size: %d test_size: %d', train_size, valid_size, test_size)
            
            full_dataset = full_dataset.shuffle(buffer_size=DATASET_SIZE)
            full_dataset = full_dataset.map(extract_tfrecords,
                                num_parallel_calls=self.input_pipeline_hps.num_cores)
            self.train_dataset = full_dataset.take(train_size)
            self.test_dataset = full_dataset.skip(train_size)
            self.valid_dataset = self.test_dataset.skip(valid_size)
            self.test_dataset = self.test_dataset.take(test_size)
            
        else:
            train_size = int(self.input_pipeline_hps.train_ratio * DATASET_SIZE)
            valid_size = DATASET_SIZE - train_size
            logging.info('train_size: %d valid_size: %d', train_size, valid_size)
            
            full_dataset = full_dataset.shuffle(buffer_size=DATASET_SIZE)
            full_dataset = full_dataset.map(extract_tfrecords, num_parallel_calls=self.input_pipeline_hps.num_cores)
            self.train_dataset = full_dataset.take(train_size)
            self.valid_dataset = full_dataset.skip(train_size)

            
    def get_train_next(self):
        with self.graph.as_default():
            return self.sess.run(self.train_next_batch)
    
    def get_valid_next(self):
        with self.graph.as_default():
            return self.sess.run(self.valid_next_batch)
        
    def get_test_next(self):
        with self.graph.as_default():
            return self.sess.run(self.test_next_batch)
        
    def __del__(self):
        self.close()
        
    def close(self):
        self.sess.close()

