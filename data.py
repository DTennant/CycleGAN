import tensorflow as tf

class data_reader(object):
    def __init__(self, args, image_paths, channels=3, shuffle=True, num_threads=4, min_after_dequeue=100,
                allow_smaller_final_batch=True):
        self.args = args
        self.batch_size = args.batch_size
        self.load_size = args.load_size
        self.crop_size = args.crop_size
        self.channels = channels
        self.paths = image_paths
        self.img_batch, self.img_num = self._make_batch_reader(shuffle, num_threads, 
                                                                min_after_dequeue,
                                                                allow_smaller_final_batch)

    def __len__(self):
        return self.img_num

    def read_batch(self, sess):
        return sess.run(self.img_batch)

    def _make_batch_reader(self, shuffle, num_threads, min_after_dequeue, allow_smaller_final_batch):
        img_queue = tf.train.string_input_producer(self.paths, shuffle=shuffle)
        reader = tf.WholeFileReader()
        # preprocessing
        _, img = reader.read(img_queue)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.resize_images(img, [self.load_size, self.load_size])
        img = tf.random_crop(img, [self.crop_size, self.crop_size, self.channels])
        img = tf.cast(img, tf.float32) / 127.5 - 1
        # batch
        if shuffle:
            capacity = min_after_dequeue + (num_threads + 1) * self.batch_size
            img_batch = tf.train.shuffle_batch([img],
                                               batch_size=self.batch_size,
                                               capacity=capacity,
                                               min_after_dequeue=min_after_dequeue,
                                               num_threads=num_threads,
                                               allow_smaller_final_batch=allow_smaller_final_batch)
        else:
            img_batch = tf.train.batch([img],
                                       batch_size=self.batch_size,
                                       allow_smaller_final_batch=allow_smaller_final_batch)
        return img_batch, len(self.paths)