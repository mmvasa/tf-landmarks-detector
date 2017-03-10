
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from pathlib import Path

import math
import matplotlib
import numpy as np
import os.path
import tensorflow as tf
import time
from model_train import deepID
from input import distorted_inputs, TRAIN_IMAGE_SIZE
from libs.utils import show_image_with_pred

Y_SIZE = 136

FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string('checkpoint_dir', 'models/',
#                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")

tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 3000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Number of examples per batch.""")
tf.app.flags.DEFINE_string('data_txt', 'test_new.txt',
                           """The text file containing test data path and annotations.""")
tf.app.flags.DEFINE_string('device', '/cpu:0', 'the device to eval on.')
tf.app.flags.DEFINE_string('use_tk2', True,
                           """Directory where to read model checkpoints.""")


def normalized_rmse(pred, gt_truth):
    # TODO: assert shapes
    #       remove 5
    norm = tf.sqrt(tf.reduce_sum(((gt_truth[:, 0, :] - gt_truth[:, 1, :])**2), 1))

    return (pred, tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(pred - gt_truth), 2)), 1) / (norm * Y_SIZE))




def _eval_once(saver, rmse_op, network):
  """Runs Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    rmse_op: rmse_op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    print(ckpt.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:

      saver.restore(sess, ckpt.model_checkpoint_path)


      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/imagenet_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('Succesfully loaded model from %s at step=%s.' %
            (ckpt.model_checkpoint_path, global_step))
    else:
      print('No checkpoint file found')
      return

    test_x, test_label = distorted_inputs(batch_size=4)
    test_label = tf.reshape(test_label, [-1,136])      


    #test_x, test_label, filename = input_pipeline([FLAGS.data_txt], batch_size=FLAGS.batch_size, shape=[64, 64, 1], is_training=False)
    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      # Counts the number of correct predictions.
      errors = []

      total_sample_count = num_iter * FLAGS.batch_size
      step = 0

      print('%s: starting evaluation on (%s).' % (datetime.now(), 'tf/'))
      start_time = time.time()
      
      while step < num_iter and not coord.should_stop():
        test_xs, label = sess.run([test_x, test_label])
        pred, rmse = sess.run(rmse_op, feed_dict={network['x']: test_xs, network['y']: label, network['train']: False,
                network['keep_prob']: 0.5})
        errors.append(rmse)
        step += 1
        if step % 10 == 0:
          duration = time.time() - start_time
          sec_per_batch = duration / 20.0
          examples_per_sec = FLAGS.batch_size / sec_per_batch
        
          print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
             'sec/batch) error = %.3f' % (datetime.now(), step, num_iter,
                             examples_per_sec, sec_per_batch, rmse[0]))

          start_time = time.time()

          if FLAGS.use_tk2 is True:
             show_image_with_pred(test_xs[0], label.reshape([-1,int(Y_SIZE/2),2])[0], pred.reshape([-1,int(Y_SIZE/2),2])[0])
             time.sleep(1)


      errors = np.vstack(errors).ravel()
      mean_rmse = errors.mean()
      auc_at_08 = (errors < .08).mean()
      auc_at_05 = (errors < .05).mean()

      print('Errors', errors.shape)
      print('%s: mean_rmse = %.4f, auc @ 0.05 = %.4f, auc @ 0.08 = %.4f [%d examples]' %
            (datetime.now(), errors.mean(), auc_at_05, auc_at_08, total_sample_count))

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
   



def evaluate(shape=[64, 64, 1]):
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    print('Loading model...')

    with tf.device(FLAGS.device):
        deepid = deepID(input_shape=[None, TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 1], n_filters=[26, 52, 52, 80], 
            filter_sizes=[3, 6, 6, 4], activation=tf.nn.relu, dropout=False)

        tf.get_variable_scope().reuse_variables()

    avg_pred = deepid['pred']
    gt_truth = deepid['y']
    gt_truth = tf.reshape(gt_truth, (-1, int(Y_SIZE/2), 2))
    norm_error = normalized_rmse(avg_pred, gt_truth)
    saver = tf.train.Saver()

    while True:
      _eval_once(saver, norm_error, deepid)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)

if __name__ == '__main__':
    evaluate()
