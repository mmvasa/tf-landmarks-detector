import tensorflow as tf
import libs.utils as utils
from input import NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 96,
                            """Number of images to process in a batch.""")

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 3.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.9  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.003       # Initial learning rate.


def inference(image, y,
        n_filters=[26, 52, 52, 80],
        filter_sizes=[3, 6, 6, 4],
        activation=tf.nn.relu,
        dropout=False):

    phase_train = tf.placeholder(tf.bool, name='phase_train')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    current_input = utils.to_tensor(image) #x_tensor
    tf.summary.image('current', current_input)
    #tf.summary.scalar('y[0]', tf.unstack(tf.unstack(y)))
    Ws = []
    shapes = []

    # Build the encoder
    shapes.append(current_input.get_shape().as_list())

    conv1, W = utils.conv2d(x=current_input,
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

    ip2, W = utils.linear(ip1, 136, name='ip2')
    Ws.append(W)
    # ip2 = activation(ip2)

    p_flat = utils.flatten(ip2)
    y_flat = tf.reshape(tf.cast(y, tf.float32), [-1,1,136]) #utils.flatten(y)

    regularizers = 5e-4 *(tf.nn.l2_loss(Ws[-1]) + tf.nn.l2_loss(Ws[-2]))
    # l2 loss
    loss_x = tf.reduce_sum(tf.squared_difference(p_flat, y_flat), 1)
    cost = tf.reduce_mean(loss_x) + regularizers
    tf.add_to_collection('losses', tf.reduce_mean(loss_x))

    return cost, p_flat


def _add_loss_summaries(total_loss):
   """Add summaries for losses in the model.
   Generates moving average for all losses and associated summaries for
   visualizing the performance of the network.
   Args:
      total_loss: Total loss from loss().
   Returns:
      loss_averages_op: op for generating moving averages of losses.
   """
   # Compute the moving average of all individual losses and the total loss.
   loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
   losses = tf.get_collection('losses')
   loss_averages_op = loss_averages.apply(losses + [total_loss])

   # Attach a scalar summary to all individual losses and the total loss; do the
   # same for the averaged version of the losses.
   for l in losses + [total_loss]:
      # Name each loss as '(raw)' and name the moving average version of the loss
      # as the original loss name.
      tf.summary.scalar(l.op.name + ' (raw)', l)
      tf.summary.scalar(l.op.name, loss_averages.average(l))

   return loss_averages_op



def train(total_cost, global_step):
   
   # Variables that affect learning rate.
   num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
   decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

   # Decay the learning rate exponentially based on the number of steps.
   lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
   tf.summary.scalar('learning_rate', lr)

   loss_averages_op = _add_loss_summaries(total_cost)

   with tf.control_dependencies([loss_averages_op]): 
      optimizer = tf.train.AdamOptimizer(lr).minimize(total_cost, global_step=global_step)
  
   with tf.control_dependencies([optimizer]):
      train_op = tf.no_op(name='train')

   return optimizer  


