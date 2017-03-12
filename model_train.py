import tensorflow as tf
import numpy as np
import os
from input import distorted_inputs, TRAIN_IMAGE_SIZE
from libs.batch_norm import batch_norm
from libs import utils
from numpy.linalg import norm
import model
import time 
from datetime import datetime
from libs.utils import show_image_with_pred

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', 'models/',
                           """Directory where to read model checkpoints.""")

tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

def train():

   with tf.Graph().as_default():
      global_step = tf.contrib.framework.get_or_create_global_step()

      images, landmarks = distorted_inputs(64)
      
      cost, pred = model.inference(images, landmarks)

      train_op = model.train(cost, global_step)

      class _LoggerHook(tf.train.SessionRunHook):
         """Logs loss and runtime."""

         def begin(self):
            self._step = -1

         def before_run(self, run_context):
            self._step += 1
            self._start_time = time.time()
            return tf.train.SessionRunArgs(cost)  # Asks for loss value.

         def after_run(self, run_context, run_values):
            duration = time.time() - self._start_time
            loss_value = run_values.results

            if self._step % 10 == 0:
               num_examples_per_step = FLAGS.batch_size
               examples_per_sec = num_examples_per_step / duration
               sec_per_batch = float(duration)
               format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
               print (format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))

      with tf.train.MonitoredTrainingSession(
               checkpoint_dir="models/",
               hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                      tf.train.NanTensorHook(cost),
                      _LoggerHook()],
               save_checkpoint_secs=6,
               save_summaries_steps=2,
               config=tf.ConfigProto(
                   log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            
         while not mon_sess.should_stop():
            [_, image_, lm_, pred_] = mon_sess.run([train_op, images, landmarks, pred])
            #print('Pred=', pred_[0])
            #print('Y=', lm_[0])
            show_image_with_pred(image_[0], pred_.reshape([-1,64,2])[0], lm_.reshape([-1,64,2])[0])
            time.sleep(0.1)


def main(argv=None):
   train()

if __name__ == '__main__':
   tf.app.run()

