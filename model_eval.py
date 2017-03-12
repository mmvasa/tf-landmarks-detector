
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
from input import distorted_inputs, TRAIN_IMAGE_SIZE
from libs.utils import show_image_with_pred
from model import inference, MOVING_AVERAGE_DECAY
Y_SIZE = 136

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', 'models/',
                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")

tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 3000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_string('data_txt', 'test_new.txt',
                           """The text file containing test data path and annotations.""")
tf.app.flags.DEFINE_string('device', '/cpu:0', 'the device to eval on.')
tf.app.flags.DEFINE_string('use_tk2', True,
                           """Directory where to read model checkpoints.""")


def normalized_rmse(pred, gt_truth):
   pred = tf.reshape(pred, [-1, 68,2])
   
   gt_truth = tf.cast(gt_truth, tf.float32)

   norm = tf.sqrt(tf.reduce_sum(((gt_truth[:, 0, :] - gt_truth[:, 1, :])**2), 1))

   return (pred, tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(pred - gt_truth), 2)), 1) / (norm * Y_SIZE))




def _eval_once(saver, image, lm_pred, landmark_truth):
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
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('Succesfully loaded model from %s at step=%s.' %
            (ckpt.model_checkpoint_path, global_step))
    else:
      print('No checkpoint file found')
      return
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
  
      gt_truth = tf.reshape(landmark_truth, (-1, int(Y_SIZE/2), 2))
      norm_error = normalized_rmse(lm_pred, gt_truth)

      while step < num_iter and not coord.should_stop():
        test_xs, label, lm_pred_ = sess.run([image, landmark_truth, lm_pred])
        pred, rmse = sess.run(norm_error)
        errors.append(rmse)
        step += 1
        if step % 1 == 0:
          duration = time.time() - start_time
          sec_per_batch = duration / 20.0
          examples_per_sec = FLAGS.batch_size / sec_per_batch
        
          print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
             'sec/batch) error = %.3f' % (datetime.now(), step, num_iter,
                             examples_per_sec, sec_per_batch, rmse[0]))

          start_time = time.time()

          if FLAGS.use_tk2 is True:
             show_image_with_pred(test_xs[0], label.reshape([-1,int(Y_SIZE/2),2])[0], lm_pred_.reshape([-1,int(Y_SIZE/2),2])[0])
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
  
   with tf.Graph().as_default() as g:
      print('Loading model...')
   
      image, landmark_truth = distorted_inputs(batch_size=4)
      landmark_truth = tf.reshape(landmark_truth, [-1,136])      
  
      _, landmark = inference(image, landmark_truth) 


      # Restore the moving average version of the learned variables for eval.
      variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
      variables_to_restore = variable_averages.variables_to_restore()
      saver = tf.train.Saver()
  
      while True:
         _eval_once(saver, image, landmark, landmark_truth)
         if FLAGS.run_once:
            break
         time.sleep(FLAGS.eval_interval_secs)

if __name__ == '__main__':
    evaluate()
