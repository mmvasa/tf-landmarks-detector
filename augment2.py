
import tensorflow as tf
from time import sleep

TRAIN_IMAGE_SIZE = 96

fig = None 

def read_dataset():

   is_training = False

   def replace_extension(filename): 
      return filename.decode('utf8').replace('.pts', '.png').encode('utf8')

   def features_parser(features):
      components = features.decode('utf8').split("\n")
      points = []      
      for p in components[3:-1]:
         points += list(map(float, p.strip().split(" ")))
      return [points]

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
      print(x_max - x_min, y_max - y_min)

      if  x_max - x_min < (y_max - y_min)/2:
         print("WARNING")
      bbox = [(y_min-y_margin) / height, (x_min-x_margin) / width, (y_max+y_margin) / height, (x_max+x_margin) / width]
      return [bbox]

   class DatasetRecord(object):
      pass
   
   result = DatasetRecord()

   files = tf.matching_files('datasets/lfpw-trainset/*.pts')
   filename_queue = tf.train.string_input_producer(files, shuffle=is_training)
   reader = tf.WholeFileReader()
   result.filename, features_file = reader.read(filename_queue)

   image_path = tf.py_func(replace_extension, [result.filename], tf.string)
   image = tf.image.decode_png(tf.read_file(image_path), channels=3)
   
   if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
   
   landmarks = tf.py_func(features_parser, [features_file], tf.double)   
   bbx = tf.py_func(boundaries, [image, landmarks], tf.double)
   
    #result.original = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), tf.cast(bbx, tf.float32))
   result.landmarks = landmarks
   resized_image = tf.image.crop_and_resize([image], tf.expand_dims(tf.cast(bbx, tf.float32), 0), [0], [TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE])
   
   float_image = tf.image.per_image_standardization(resized_image[0])
   float_image.set_shape([TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 3])
   
   result.image = resized_image[0]

   return result


def distorted_inputs():
   read_input = read_dataset()

   reshaped_image = read_input.image #tf.cast(read_input.image, tf.float32)

   #distorted_image = tf.random_crop(reshaped_image, [TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 1])

   distorted_image = tf.image.random_flip_left_right(reshaped_image)

   distorted_image = tf.image.random_brightness(distorted_image, max_delta=0.1)

   #distorted_image = tf.image.random_contrast(distorted_image,
   #                                         lower=0.2, upper=1.8)

   read_input.image = distorted_image

   return read_input
  

def show_image(img):
   import matplotlib.pyplot as plt
   
   global fig
   

   if fig is None:
      fig = plt.figure()
      plt.show(block=False)
   plt.clf()
   plt.imshow(img)
   #import matplotlib.pyplot as plt
   #plt.imshow(img)
   #points = label.reshape([-1,int(TRAIN_IMAGE_SIZE/2),2])
   #for p in points[0]:
   #   plt.plot(p[0] * 64, p[1] * 64, 'g.')
   #   points = pred.reshape([-1,int(Y_SIZE/2),2])
   #for p in points[0]:
   #   plt.plot(p[0] * 64, p[1] * 64, 'r.')
   fig.canvas.draw()


read_input = distorted_inputs()

with tf.Session() as sess:
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(coord=coord)

   for i in range(1200):
      filename_, landmarks_, image_ = sess.run([read_input.filename, read_input.landmarks, read_input.image])

      print(filename_)
      show_image(image_)
      sleep(1.5)
   coord.request_stop()
   coord.join(threads) 

