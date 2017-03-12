import tensorflow as tf
from time import sleep
import numpy as np

fig = None

TRAIN_IMAGE_SIZE = 64
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2000

def read_dataset(format="jpg"):

   is_training = True

   def replace_extension(filename): 
      return filename.decode('utf8').replace('.pts', '.' + format).encode('utf8')

   def features_parser(features):
      components = features.decode('utf8').split("\n")
      points = []      
      for p in components[3:-1]:
         points += list(map(float, p.strip().split(" ")))
      return [points]

   def correct_landmarks(image, landmarks):
      bbx = boundaries(image, landmarks)
      width = TRAIN_IMAGE_SIZE / (bbx[3] - bbx[1])
      height = TRAIN_IMAGE_SIZE / (bbx[2] - bbx[0])  
      landmarks = landmarks.reshape(-1, 2)
      landmarks = [[(l[0] - bbx[1]) * width, (l[1] - bbx[0]) * height ] for l in landmarks] 
      return [landmarks]

   def crop_image(image, landmarks):
      width = image.shape[1]
      height = image.shape[0]  
      bbx = boundaries(image, landmarks)
      return [[bbx[0] / height, bbx[1] / width, bbx[2] / height, bbx[3] / width]]

   def boundaries(image, landmarks):
      landmarks = landmarks.reshape(-1, 2)
      x_min = int(min([a[0] for a in landmarks]))
      x_max = int(max([a[0] for a in landmarks]))
      y_min = int(min([a[1] for a in landmarks]))
      y_max = int(max([a[1] for a in landmarks]))
      width = image.shape[1]
      height = image.shape[0]  
      x_margin = (x_max - x_min) * 0.1
      y_margin = (y_max - y_min) * 0.1
      bbox = [(y_min-y_margin), (x_min-x_margin), (y_max+y_margin), (x_max+x_margin)]
      return bbox 

   class DatasetRecord(object):
      pass
   
   result = DatasetRecord()

   files = tf.matching_files('datasets/helen-trainset/*.pts')
   filename_queue = tf.train.string_input_producer(files, shuffle=is_training)

   reader = tf.WholeFileReader()
   result.filename, features_file = reader.read(filename_queue)

   image_path = tf.py_func(replace_extension, [result.filename], tf.string)

   if format == 'jpg':   
      image = tf.image.decode_jpeg(tf.read_file(image_path), channels=1)
   else: 
      image = tf.image.decode_png(tf.read_file(image_path), channels=1)
   
   if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
   
   landmarks = tf.py_func(features_parser, [features_file], tf.double)   
   landmarks_crop = tf.py_func(correct_landmarks, [image, landmarks], tf.double)   
   
   bbx = tf.py_func(crop_image, [image, landmarks], tf.double)   
   resized_image = tf.image.crop_and_resize([image], tf.expand_dims(tf.cast(bbx, tf.float32), 0), [0], [TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE])

   float_image = tf.image.per_image_standardization(resized_image[0])
   float_image.set_shape([TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 1])
   
   result.landmarks = landmarks_crop
   result.image = float_image

   return result

def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
   num_preprocess_threads = 4
   if shuffle:
      images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples,
        )
   else:
      images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

   tf.summary.image('images', images)

   return images, tf.reshape(label_batch, [batch_size, 68, 2])

def distorted_inputs(batch_size):

   min_fraction_of_examples_in_queue = 1 
   min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)

   read_input = read_dataset()
   distorted_image = read_input.image
   #distorted_image = tf.image.random_brightness(distorted_image, max_delta=0.2)
   #distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
  
   distorted_image.set_shape([TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 1])
   read_input.landmarks.set_shape([68, 2])   

   return _generate_image_and_label_batch(distorted_image, read_input.landmarks,
                                         min_queue_examples, batch_size,
                                         shuffle=True)
  
if __name__ == "__main__":
   read_input = distorted_inputs(64)
   with tf.Session() as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      for i in range(120):
         image_, landmarks_ = sess.run([read_input[0], read_input[1]])
         print(image_)
         show_image(image_[0], landmarks_[0])
         sleep(0.5)
      coord.request_stop()
      coord.join(threads) 

