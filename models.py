import functools
import tensorflow as tf
from ops import conv, deconv, lrelu, bn, relu

def d_net(img, scope, df_dim=64, is_training=True, reuse=False):
    global bn
    bn = functools.partial(bn, is_training=is_training)
    with tf.variable_scope(scope + '_d', reuse=reuse):
        n = lrelu(conv(img, df_dim, kernel_size=4, stride=2, scope='conv1'))
        n = lrelu(bn(conv(n, df_dim * 2, kernel_size=4, stride=2, scope='conv2'), scope='bn1'))
        # (64 x 64 x df_dim*2)
        n = lrelu(bn(conv(n, df_dim * 4, kernel_size=4, stride=2, scope='conv3'), scope='bn2'))
        # (32x 32 x df_dim*4)
        n = lrelu(bn(conv(n, df_dim * 8, kernel_size=4, stride=1, scope='conv4'), scope='bn3'))
        # (32 x 32 x df_dim*8)
        n = conv(n, 1, kernel_size=4, stride=1, scope='conv5') 
        # (32 x 32 x 1)
    return n

def g_net(img, scope, gf_dim=64, is_training=True, reuse=False):
    global bn
    bn = functools.partial(bn, is_training=is_training)
    def res_block(x, dim, scope='res'):
        y = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        y = relu(bn(conv(y, dim, kernel_size=3, stride=1, padding='VALID', 
                        scope=scope + '_conv1'), scope=scope + '_bn1'))
        y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        y = bn(conv(y, dim, kernel_size=3, stride=1, padding='VALID', 
                    scope=scope + '_conv2'), scope=scope + '_bn2')
        return y + x

    with tf.variable_scope(scope + '_g', reuse=reuse):
        c0 = tf.pad(img, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = relu(bn(conv(c0, gf_dim, 7, 1, padding='VALID', scope='c1_conv'), scope='c1_bn'))
        c2 = relu(bn(conv(c1, gf_dim * 2, 3, 2, scope='c2_conv'), scope='c2_bn'))
        c3 = relu(bn(conv(c2, gf_dim * 4, 3, 2, scope='c3_conv'), scope='c3_bn'))

        r1 = res_block(c3, gf_dim * 4, scope='r1')
        r2 = res_block(r1, gf_dim * 4, scope='r2')
        r3 = res_block(r2, gf_dim * 4, scope='r3')
        r4 = res_block(r3, gf_dim * 4, scope='r4')
        r5 = res_block(r4, gf_dim * 4, scope='r5')
        r6 = res_block(r5, gf_dim * 4, scope='r6')
        r7 = res_block(r6, gf_dim * 4, scope='r7')
        r8 = res_block(r7, gf_dim * 4, scope='r8')
        r9 = res_block(r8, gf_dim * 4, scope='r9')

        d1 = relu(bn(deconv(r9, gf_dim * 2, 3, 2, scope='d1_dconv'), scope='d1_bn'))
        d2 = relu(bn(deconv(d1, gf_dim, 3, 2, scope='d2_dconv'), scope='d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = conv(d2, 3, 7, 1, padding='VALID', scope='pred_conv')
        pred = tf.nn.tanh(pred)

    return pred
