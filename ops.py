import re
import functools
import tensorflow as tf
import tensorflow.contrib.slim as slim

conv = functools.partial(slim.conv2d, activation_fn=None)
deconv = functools.partial(slim.conv2d_transpose, activation_fn=None)
relu = tf.nn.relu
bn = functools.partial(slim.batch_norm, scale=True, decay=0.9, 
                       epsilon=1e-5, updates_collections=None)

def lrelu(x, leak=0.2, scope=None):
    with tf.name_scope(scope, 'lrelu', [x, leak]):
        y = tf.maximum(x, leak * x)
    return y

# replace with tf.nn.l1_loss
def l1_loss(x, y, weight=1.0, scope=None):
    with tf.name_scope(scope, 'l1_loss', [x, y, weight]):
        loss = tf.reduce_mean(tf.abs(x - y)) * weight
    return loss

# replace with tf.nn.l2_loss
def l2_loss(x, y, weight=1.0, scope=None):
    with tf.name_scope(scope, 'l2_loss', [x, y, weight]):
        loss = tf.reduce_mean((x - y) ** 2) * weight
    return loss


def summary(tensor, summary_type=['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram']):
    tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', tensor.name)
    tensor_name = re.sub(':', '-', tensor_name)

    with tf.name_scope('summary_' + tensor_name):
        summaries = []
        if len(tensor.shape) == 0:
            summaries.append(tf.summary.scalar(tensor_name, tensor))
        else:
            if 'mean' in summary_type:
                mean = tf.reduce_mean(tensor)
                summaries.append(tf.summary.scalar(tensor_name + '/mean', mean))
            if 'stddev' in summary_type:
                mean = tf.reduce_mean(tensor)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
                summaries.append(tf.summary.scalar(tensor_name + '/stddev', stddev))
            if 'max' in summary_type:
                summaries.append(tf.summary.scalar(tensor_name + '/max', tf.reduce_max(tensor)))
            if 'min' in summary_type:
                summaries.append(tf.summary.scalar(tensor_name + '/min', tf.reduce_min(tensor)))
            if 'sparsity' in summary_type:
                summaries.append(tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(tensor)))
            if 'histogram' in summary_type:
                summaries.append(tf.summary.histogram(tensor_name, tensor))
    return tf.summary.merge(summaries)

def summary_tensors(tensors, summary_type=['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram']):
    with tf.name_scope('summary_tensors'):
        summaries = []
        for tensor in tensors:
            summaries.append(summary(tensor, summary_type))
    return tf.summary.merge(summaries)


def load_checkpoint(checkpoint_dir, sess, saver):
    print(" [*] Loading checkpoint...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        saver.restore(sess, ckpt_path)
        print(" [*] Loading successful!")
        return ckpt_path
    else:
        print(" [*] No suitable checkpoint!")
        return None

def counter(scope='counter'):
    with tf.variable_scope(scope):
        counter = tf.Variable(0, dtype=tf.int32, name='counter')
        update_cnt = tf.assign(counter, tf.add(counter, 1))
    return counter, update_cnt
