import ops
import data
import utils
import argparse
import numpy as np
from glob import glob
import tensorflow as tf
from models import d_net, g_net

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--batch_size', type=int, default=1, help='number of batch_size')
    parser.add_argument('--dataset', type=str, default='horse2zebra', help='the dataset to use')
    parser.add_argument('--load_size', type=int, default=286, help='resize the input img to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='crop the resized img')
    parser.add_argument('--epoch', type=int, default=200, help='the number of epoches')
    parser.add_argument('--lr', type=float, default=2e-4, help='init lr')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 in AdamOptimizer')
    parser.add_argument('--gpu', type=int, default=0, help='use which gpu to compute')
    parser.add_argument('--sample_interval', type=int, default=20, help='sample during # epoch')
    parser.add_argument('--save_interval', type=int, default=20, help='save during # epoch')
    return parser.parse_args()

def build_graph(args, a_r, b_r, a2b_s, b2a_s):
    with tf.device('/gpu:{}'.format(args.gpu)):
        a2b = g_net(a_r, 'a2b')
        b2a = g_net(b_r, 'b2a')
        a2b2a = g_net(a2b, 'b2a', reuse=True)
        b2a2b = g_net(b2a, 'a2b', reuse=True)
        cvt = (a2b, b2a, a2b2a, b2a2b)

        a_d = d_net(a_r, 'a')
        b2a_d = d_net(b2a, 'a', reuse=True)
        b2a_s_d = d_net(b2a_s, 'a', reuse=True)

        b_d = d_net(b_r, 'b')
        a2b_d = d_net(a2b, 'b', reuse=True)
        a2b_s_d = d_net(a2b_s, 'b', reuse=True)

        g_loss_a2b = tf.identity(ops.l2_loss(a2b_d, tf.ones_like(a2b_d)), name='g_loss_a2b')
        g_loss_b2a = tf.identity(ops.l2_loss(b2a_d, tf.ones_like(b2a_d)), name='g_loss_b2a')
        cyc_loss_a = tf.identity(ops.l1_loss(a_r, a2b2a) * 10.0, name='cyc_loss_a')
        cyc_loss_b = tf.identity(ops.l1_loss(b_r, b2a2b) * 10.0, name='cyc_loss_b')
        g_loss = g_loss_a2b + g_loss_b2a + cyc_loss_a + cyc_loss_b

        d_loss_a_r = ops.l2_loss(a_d, tf.ones_like(a_d))
        d_loss_b2a_s = ops.l2_loss(b2a_s_d, tf.zeros_like(b2a_s_d))
        d_loss_a = tf.identity((d_loss_a_r + d_loss_b2a_s) / 2., name='d_loss_a')

        d_loss_b_r = ops.l2_loss(b_d, tf.ones_like(b_d))
        d_loss_a2b_s = ops.l2_loss(a2b_s_d, tf.zeros_like(a2b_s_d))
        d_loss_b = tf.identity((d_loss_b_r + d_loss_a2b_s) / 2., name='d_loss_b')

        g_sum = ops.summary_tensors([g_loss_a2b, g_loss_b2a, cyc_loss_a, cyc_loss_b])
        d_sum_a = ops.summary(d_loss_a)
        d_sum_b = ops.summary(d_loss_b)
        sum_ = (g_sum, d_sum_a, d_sum_b)

        all_var = tf.trainable_variables()
        g_var = [var for var in all_var if 'a2b_g' in var.name or 'b2a_g' in var.name]
        d_a_var = [var for var in all_var if 'a_d' in var.name]
        d_b_var = [var for var in all_var if 'b_d' in var.name]

        g_tr_op = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(g_loss, var_list=g_var)
        d_tr_op_a = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(d_loss_a, var_list=d_a_var)
        d_tr_op_b = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(d_loss_b, var_list=d_b_var)
        tr_op = (g_tr_op, d_tr_op_a, d_tr_op_b)
    return cvt, sum_, tr_op

def get_data_reader(args):
    a_img_paths = glob('datasets/' + args.dataset + '/trainA/*.jpg')
    b_img_paths = glob('datasets/' + args.dataset + '/trainB/*.jpg')
    a_dr = data.data_reader(args, a_img_paths)
    b_dr = data.data_reader(args, b_img_paths)

    a_te_paths = glob('datasets/' + args.dataset + '/testA/*.jpg')
    b_te_paths = glob('datasets/' + args.dataset + '/testB/*.jpg')
    a_te_dr = data.data_reader(args, a_te_paths)
    b_te_dr = data.data_reader(args, b_te_paths)
    return a_dr, b_dr, a_te_dr, b_te_dr

def main():
    args = get_args()
    a_r = tf.placeholder(shape=[None, args.crop_size, args.crop_size, 3], dtype=tf.float32)
    b_r = tf.placeholder(shape=[None, args.crop_size, args.crop_size, 3], dtype=tf.float32)
    a2b_s = tf.placeholder(shape=[None, args.crop_size, args.crop_size, 3], dtype=tf.float32)
    b2a_s = tf.placeholder(shape=[None, args.crop_size, args.crop_size, 3], dtype=tf.float32)
    cvt, sum_, tr_op = build_graph(args, a_r, b_r, a2b_s, b2a_s)
    a2b, b2a, a2b2a, b2a2b = cvt
    g_sum, d_sum_a, d_sum_b = sum_
    g_tr_op, d_tr_op_a, d_tr_op_b = tr_op
    # create a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    it_cnt, update_cnt = ops.counter()
    # get data
    a_dr, b_dr, a_te_dr, b_te_dr = get_data_reader(args)
    a2b_pool = utils.item_pool()
    b2a_pool = utils.item_pool()
    # create summary writer
    summary_writer = tf.summary.FileWriter('summaries/' + args.dataset, sess.graph)
    # create saver
    ckpt_dir = 'checkpoints/' + args.dataset
    utils.mkdir(ckpt_dir + '/')
    saver = tf.train.Saver(max_to_keep=5)
    ckpt_path = ops.load_checkpoint(ckpt_dir, sess, saver)
    if ckpt_path is None:
        sess.run(tf.global_variables_initializer())
    else:
        print('Copy variables from {}'.format(ckpt_path))

    try:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        it_every_epoch = min(len(a_dr), len(b_dr)) // args.batch_size
        max_it = args.epoch * it_every_epoch
        start_it = sess.run(it_cnt)
        for it in range(start_it, max_it):
            sess.run(update_cnt)
            
            a_real = a_dr.read_batch(sess)
            b_real = b_dr.read_batch(sess)
            a2b_opt, b2a_opt = sess.run([a2b, b2a], feed_dict={a_r: a_real, b_r: b_real})
            a2b_sample_ipt = np.array(a2b_pool(list(a2b_opt)))
            b2a_sample_ipt = np.array(b2a_pool(list(b2a_opt)))

            # train G
            g_summary, _ = sess.run([g_sum, g_tr_op], feed_dict={a_r: a_real, b_r: b_real})
            summary_writer.add_summary(g_summary, it)
            # train D_b
            d_summary_b, _ = sess.run([d_sum_b, d_tr_op_b], feed_dict={b_r: b_real, a2b_s: a2b_sample_ipt})
            summary_writer.add_summary(d_summary_b, it)
            # train D_a
            d_summary_a, _ = sess.run([d_sum_a, d_tr_op_a], feed_dict={a_r: a_real, b2a_s: b2a_sample_ipt})
            summary_writer.add_summary(d_summary_a, it)

            # check epoch
            epoch = it // it_every_epoch
            it_epoch = it % it_every_epoch + 1
            print("Epoch[{}]: [{}/{}]".format(epoch, it_epoch, it_every_epoch))

            if (epoch + 1) % args.save_interval == 0 and it_epoch == 1:
                save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, it_every_epoch))
                print('[*] Model saved in file: %s' % save_path)

            if (epoch + 1) % args.sample_interval == 0 and it_epoch == 1:
                a_te_r = a_te_dr.read_batch(sess)
                b_te_r = b_te_dr.read_batch(sess)
                a2b_opt, a2b2a_opt, b2a_opt, b2a2b_opt = sess.run([a2b, a2b2a, b2a, b2a2b], feed_dict={a_r: a_te_r, b_r: b_te_r})
                sample_opt = np.concatenate((a_te_r, a2b_opt, a2b2a_opt, b_te_r, b2a_opt, b2a2b_opt), axis=0)
                save_dir = 'samples/' + args.dataset
                utils.mkdir(save_dir + '/')
                utils.imwrite(utils.immerge(sample_opt, 2, 3), '{}/Epoch{}.jpg'.format(save_dir, epoch))
                print('[*] Saved sample img to ' + '{}/Epoch{}.jpg'.format(save_dir, epoch))

    except Exception as e:
        coord.request_stop(e)
    finally:
        print("Stop threads and close session!")
        coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    main()