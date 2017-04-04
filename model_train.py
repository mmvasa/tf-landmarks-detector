import tensorflow as tf
import numpy as np
import os
from input import input_pipeline
from libs.batch_norm import batch_norm
from libs import utils
from numpy.linalg import norm


Y_SIZE = 136
IMAGE_SIZE = 64

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', 'models/',
                           """Directory where to read model checkpoints.""")

def deepID(input_shape,
        n_filters,
        filter_sizes,
        activation=tf.nn.relu,
        dropout=False):

   x = tf.placeholder(tf.float32, input_shape, 'x')
   y = tf.placeholder(tf.float32, [None, Y_SIZE], 'y')
   phase_train = tf.placeholder(tf.bool, name='phase_train')
   keep_prob = tf.placeholder(tf.float32, name='keep_prob')

   # 2d -> 4d if convolution
   x_tensor = utils.to_tensor(x)
   current_input = x_tensor

   Ws = []
   shapes = []

   # Build the encoder
   shapes.append(current_input.get_shape().as_list())
   conv1, W = utils.conv2d(x=x_tensor,
                     n_output=n_filters[0],
                     k_h=filter_sizes[0],
                     k_w=filter_sizes[0],
                     d_w=1,
                     d_h=1,
                     name='conv1')
   Ws.append(W)
   # conv1 = activation(batch_norm(conv1, phase_train, 'bn1'))
   conv1 = activation(conv1)


   pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

   conv2, W = utils.conv2d(x=pool1,
                     n_output=n_filters[1],
                     k_h=filter_sizes[1],
                     k_w=filter_sizes[1],
                     d_w=1,
                     d_h=1,
                     name='conv2')
   Ws.append(W)
   # conv2 = activation(batch_norm(conv2, phase_train, 'bn2'))
   conv2 = activation(conv2)

   pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

   conv3, W = utils.conv2d(x=pool2,
                     n_output=n_filters[2],
                     k_h=filter_sizes[2],
                     k_w=filter_sizes[2],
                     d_w=1,
                     d_h=1,
                     name='conv3')
   Ws.append(W)
   # conv3 = activation(batch_norm(conv3, phase_train, 'bn3'))
   conv3 = activation(conv3)

   pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

   conv4, W = utils.conv2d(x=pool3,
                     n_output=n_filters[3],
                     k_h=filter_sizes[3],
                     k_w=filter_sizes[3],
                     d_w=1,
                     d_h=1,
                     name='conv4')
   Ws.append(W)
   # conv4 = activation(batch_norm(conv4, phase_train, 'bn4'))
   conv4 = activation(conv4)

   pool3_flat = utils.flatten(pool3)
   conv4_flat = utils.flatten(conv4)
   concat = tf.concat(1, [pool3_flat, conv4_flat],  name='concat')

   ip1, W = utils.linear(concat, 120, name='ip1')
   Ws.append(W)
   ip1 = activation(ip1)
   if dropout:
     ip1 = tf.nn.dropout(ip1, keep_prob)

   ip2, W = utils.linear(ip1, Y_SIZE, name='ip2')
   Ws.append(W)
   # ip2 = activation(ip2)

   p_flat = utils.flatten(ip2)
   y_flat = utils.flatten(y)

   regularizers = 1e-4 *(tf.nn.l2_loss(Ws[-1]) + tf.nn.l2_loss(Ws[-2]))
   # l2 loss
   loss_x = tf.reduce_sum(tf.squared_difference(p_flat, y_flat), 1)
   cost = tf.reduce_mean(loss_x) + regularizers
   prediction = tf.reshape(p_flat, (-1, int(Y_SIZE/2), 2))

   return {'cost': cost, 'Ws': Ws,
         'x': x, 'y': y, 'pred': prediction,
         'keep_prob': keep_prob,
         'train': phase_train}

def train_deepid(input_shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1],
                n_filters=[30, 60, 60, 90],
                filter_sizes=[3, 6, 6, 4],
                activation=tf.nn.relu,
                dropout=False,
                batch_size=64):
   with tf.device('/cpu:0'): 
      batch_x, label_x,_ = input_pipeline(['train_new.txt'], batch_size=batch_size, shape=[IMAGE_SIZE, IMAGE_SIZE, 1], is_training=True)

   deepid = deepID(input_shape=input_shape, n_filters=n_filters, filter_sizes=filter_sizes, activation=activation,
     dropout=dropout)

   batch = tf.Variable(0, dtype=tf.int32)
   learning_rate = tf.train.exponential_decay(0.003, batch * batch_size, 150000, 0.95, staircase=True)
   optimizer = tf.train.AdamOptimizer(
     learning_rate).minimize(deepid['cost'], global_step=batch)
   save_step = 100
   saver = tf.train.Saver()
 
   with tf.Session() as sess:
       ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
       if ckpt and ckpt.model_checkpoint_path:
           saver.restore(sess, ckpt.model_checkpoint_path)
           global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
           print('Succesfully loaded model from %s at step=%s.' %
               (ckpt.model_checkpoint_path, global_step))
       else:
           saver = tf.train.Saver(max_to_keep=5)
           sess.run(tf.initialize_all_variables())

       coord = tf.train.Coordinator()
       tf.get_default_graph().finalize()
       threads = tf.train.start_queue_runners(sess=sess, coord=coord)

       batch_i = 0
       run_metadata = tf.RunMetadata()
       for i in range(1000000):
           batch_i += 1
           batch_xs, batch_label = sess.run([batch_x, label_x], options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
           run_metadata=run_metadata)
           train_cost, pred = sess.run([deepid['cost'], deepid['pred'], optimizer], feed_dict={
                      deepid['x']: batch_xs, deepid['y']: batch_label, deepid['train']: True,
                      deepid['keep_prob']: 0.5})[:2]
                  
           if batch_i % 60 == 0:
               lr = sess.run(learning_rate)
               print("batch_i: {}, learning-rate: {:0.10f} train_cost: {:0.10f}".format(batch_i, lr, train_cost))
               id = np.random.randint(10)
               batch_label = batch_label.reshape([-1,int(Y_SIZE/2),2])
               
           if batch_i % save_step == 0:
               saver.save(sess, "./models/" + 'deepid.ckpt', 
                           global_step=batch_i,
                           write_meta_graph=False)

       coord.request_stop()
       coord.join(threads)

if __name__ == '__main__':
    train_deepid()
