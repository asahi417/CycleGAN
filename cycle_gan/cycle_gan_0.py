import os
import tensorflow as tf
import numpy as np
from .dataset_tool import tfrecord_parser
from .util import create_log
from .util_tf import generator_resnet, discriminator_patch

DATA_TYPE = ['trainB', 'trainA', 'testA', 'testB']


class CycleGAN:

    """
    Only valid for images with shape [256, 256, 3]
    """

    def __init__(self,
                 tfrecord_dir: str,
                 checkpoint_dir: str,
                 image_shape: list,
                 cyclic_lambda_a: float,
                 cyclic_lambda_b: float,
                 identity_lambda: float,
                 learning_rate: float=None,
                 buffer_size: int = 50,
                 batch: int = 10,
                 optimizer: str = 'sgd',
                 debug: bool = True,
                 n_thread: int = 4,
                 log_img_size: int = 5
                 ):

        self.__ini_learning_rate = learning_rate
        self.__checkpoint_dir = checkpoint_dir
        self.__checkpoint = '%s/model.ckpt' % checkpoint_dir
        self.__log_img_size = log_img_size
        self.__identity_lambda = identity_lambda
        self.__buffer_size = buffer_size
        self.__image_shape = image_shape
        self.__cyclic_lambda_a = cyclic_lambda_a
        self.__cyclic_lambda_b = cyclic_lambda_b

        self.__base_batch = batch
        self.__optimizer = optimizer
        self.__logger = create_log('%s/log' % checkpoint_dir) if debug else None
        self.__n_thread = n_thread
        self.tfrecord_dir = tfrecord_dir

        self.__build_network()

        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.writer = tf.summary.FileWriter('%s/summary' % self.__checkpoint_dir, self.session.graph)

        # Load model
        if os.path.exists('%s.meta' % self.__checkpoint):
            self.__log('load variable from %s' % self.__checkpoint)
            self.__saver.restore(self.session, self.__checkpoint)
            self.__warm_start = True
        else:
            os.makedirs(self.__checkpoint_dir, exist_ok=True)
            self.session.run(tf.global_variables_initializer())
            self.__warm_start = False

    def __tfreocrd(self, record_name, batch, seed):
        data_set_api = tf.data.TFRecordDataset(record_name, compression_type='GZIP')
        # convert record to tensor
        data_set_api = data_set_api.map(tfrecord_parser(self.__image_shape), self.__n_thread)
        # set batch size
        data_set_api = data_set_api.shuffle(buffer_size=10000, seed=tf.cast(seed, tf.int64))
        data_set_api = data_set_api.batch(tf.cast(batch, tf.int64))
        # make iterator
        iterator = tf.data.Iterator.from_structure(data_set_api.output_types, data_set_api.output_shapes)
        iterator_ini = iterator.make_initializer(data_set_api)
        return iterator, iterator_ini

    def __build_network(self):

        with tf.name_scope('placeholders'):
            self.__tfrecord_a = tf.placeholder(tf.string, name='tfrecord_a')
            self.__tfrecord_b = tf.placeholder(tf.string, name='tfrecord_b')
            self.__batch = tf.placeholder_with_default(self.__base_batch, [], name='batch')
            self.__seed = tf.placeholder_with_default(1234, [], name='seed')
            self.__learning_rate = tf.placeholder_with_default(0.0, [], name='learning_rate')
            # will sampled from buffered generated images
            self.__fake_img_a_buffer = tf.placeholder(tf.float32,
                                                      [self.__base_batch] + self.__image_shape,
                                                      name='fake_image')
            self.__fake_img_b_buffer = tf.placeholder(tf.float32,
                                                      [self.__base_batch] + self.__image_shape,
                                                      name='fake_image')

        # tfrecord instance for training generators
        iterator_gen_a, self.iterator_gen_ini_a = self.__tfreocrd(self.__tfrecord_a, self.__batch, self.__seed)
        iterator_gen_b, self.iterator_gen_ini_b = self.__tfreocrd(self.__tfrecord_b, self.__batch, self.__seed)

        # tfrecord instance for training discriminators
        iterator_disc_a, self.iterator_disc_ini_a = self.__tfreocrd(self.__tfrecord_a, self.__batch, self.__seed)
        iterator_disc_b, self.iterator_disc_ini_b = self.__tfreocrd(self.__tfrecord_b, self.__batch, self.__seed)

        with tf.name_scope('model'):
            # generator
            with tf.name_scope('generators'):
                original_img_a = iterator_gen_a.get_next()
                original_img_a_norm = (original_img_a/255 - 0.5) * 2
                self.fake_img_b = generator_resnet(original_img_a_norm, scope='generator_a')
                cycle_img_a = generator_resnet(self.fake_img_b, scope='generator_b')

                original_img_b = iterator_gen_b.get_next()
                original_img_b_norm = (original_img_b / 255 - 0.5) * 2
                self.fake_img_a = generator_resnet(original_img_b_norm, scope='generator_b', reuse=True)
                cycle_img_b = generator_resnet(self.fake_img_a, scope='generator_a', reuse=True)

                if self.__identity_lambda != 0.0:
                    id_a = generator_resnet(original_img_a_norm, scope='generator_b', reuse=True)
                    self.id_loss_a = tf.reduce_mean(tf.abs(original_img_a_norm - id_a))

                    id_b = generator_resnet(original_img_b_norm, scope='generator_a', reuse=True)
                    self.id_loss_b = tf.reduce_mean(tf.abs(original_img_a_norm - id_b))
                else:
                    self.id_loss_a = self.id_loss_b = 0.0

            # discriminator (batch size would be varying due to random cropping)
            with tf.name_scope('discriminators'):

                # logit for update generator A
                logit_fake_a_generator = discriminator_patch(self.fake_img_a, scope='discriminator_a')
                # prob_fake_a_generator = tf.nn.softmax(logit_fake_a_generator)

                # logit for update generator B
                logit_fake_b_generator = discriminator_patch(self.fake_img_b, scope='discriminator_b')
                # prob_fake_b_generator = tf.nn.softmax(logit_fake_b_generator)

                # logit for update discriminator A
                original_img_disc_a = iterator_disc_a.get_next()
                original_img_disc_a_norm = (original_img_disc_a/255 - 0.5) * 2
                logit_fake_a = discriminator_patch(self.__fake_img_a_buffer, scope='discriminator_a', reuse=True)
                # prob_fake_a = tf.nn.softmax(logit_fake_a)
                logit_original_a = discriminator_patch(original_img_disc_a_norm, scope='discriminator_a', reuse=True)
                # prob_original_a = tf.nn.softmax(logit_original_a)

                # logit for update discriminator B
                original_img_disc_b = iterator_disc_b.get_next()
                original_img_disc_b_norm = (original_img_disc_b / 255 - 0.5) * 2
                logit_fake_b = discriminator_patch(self.__fake_img_b_buffer, scope='discriminator_b', reuse=True)
                # prob_fake_b = tf.nn.softmax(logit_fake_b)
                logit_original_b = discriminator_patch(original_img_disc_b_norm, scope='discriminator_b', reuse=True)
                # prob_original_b = tf.nn.softmax(logit_original_b)

        # loss
        with tf.name_scope('loss'):
            # adversarial loss (least square loss, known as LSGAN)
            self.gen_loss_a = tf.reduce_mean(tf.squared_difference(logit_fake_a_generator, 1))
            self.gen_loss_b = tf.reduce_mean(tf.squared_difference(logit_fake_b_generator, 1))

            self.disc_loss_a = (tf.reduce_mean(tf.squared_difference(logit_fake_a, 0)) +
                                tf.reduce_mean(tf.squared_difference(logit_original_a, 1))) * 0.5
            self.disc_loss_b = (tf.reduce_mean(tf.squared_difference(logit_fake_b, 0)) +
                                tf.reduce_mean(tf.squared_difference(logit_original_b, 1))) * 0.5

            # cycle consistency loss
            self.cycle_loss_a = tf.reduce_mean(tf.abs(original_img_a - cycle_img_a))
            self.cycle_loss_b = tf.reduce_mean(tf.abs(original_img_b - cycle_img_b))

        with tf.name_scope('optimization'):
            # optimizer
            if self.__optimizer == 'adam':
                optimizer_g_a = tf.train.AdamOptimizer(self.__learning_rate)
                optimizer_g_b = tf.train.AdamOptimizer(self.__learning_rate)
                optimizer_d_a = tf.train.AdamOptimizer(self.__learning_rate)
                optimizer_d_b = tf.train.AdamOptimizer(self.__learning_rate)
            else:
                raise ValueError('unknown optimizer !!')

            var_gen_a = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_a')
            var_gen_b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_b')
            # self.train_op_gen_a = optimizer.minimize(
            #     self.gen_loss_b +
            #     self.cycle_loss_a * self.__cyclic_lambda_a +
            #     self.cycle_loss_b * self.__cyclic_lambda_b +
            #     self.id_loss_b * self.__cyclic_lambda_a * self.__identity_lambda,
            #     var_list=var_gen_a)
            # self.train_op_gen_b = optimizer.minimize(
            #     self.gen_loss_a +
            #     self.cycle_loss_a * self.__cyclic_lambda_a +
            #     self.cycle_loss_b * self.__cyclic_lambda_b +
            #     self.id_loss_a * self.__cyclic_lambda_b * self.__identity_lambda,
            #     var_list=var_gen_b)

            var_disc_a = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_a')
            var_disc_b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_b')
            # self.train_op_disc_a = optimizer.minimize(
            #     self.disc_loss_a,
            #     var_list=var_disc_a)
            # self.train_op_disc_b = optimizer.minimize(
            #     self.disc_loss_b,
            #     var_list=var_disc_b)

            grad_gen_a = tf.gradients(
                self.gen_loss_b +
                self.cycle_loss_a * self.__cyclic_lambda_a +
                self.cycle_loss_b * self.__cyclic_lambda_b +
                self.id_loss_b * self.__cyclic_lambda_a * self.__identity_lambda,
                var_gen_a)
            grad_gen_b = tf.gradients(
                self.gen_loss_a +
                self.cycle_loss_a * self.__cyclic_lambda_a +
                self.cycle_loss_b * self.__cyclic_lambda_b +
                self.id_loss_a * self.__cyclic_lambda_b * self.__identity_lambda,
                var_gen_b)
            with tf.control_dependencies(grad_gen_a+grad_gen_b):
                self.train_op_gen_a = optimizer_g_a.apply_gradients(zip(grad_gen_a, var_gen_a))
                self.train_op_gen_b = optimizer_g_b.apply_gradients(zip(grad_gen_b, var_gen_b))

            grad_disc_a = tf.gradients(self.disc_loss_a, var_disc_a)
            grad_disc_b = tf.gradients(self.disc_loss_b, var_disc_b)
            with tf.control_dependencies(grad_disc_a+grad_disc_b):
                self.train_op_disc_a = optimizer_d_a.apply_gradients(zip(grad_disc_a, var_disc_a))
                self.train_op_disc_b = optimizer_d_b.apply_gradients(zip(grad_disc_b, var_disc_b))

        # logging
        n_var = 0
        for var in tf.trainable_variables():
            sh = var.get_shape().as_list()
            self.__log('%s: %s' % (var.name, str(sh)))
            n_var += np.prod(sh)
        self.__log('total variables: %i' % n_var)

        # saver
        self.__saver = tf.train.Saver()

        ##################
        # scalar summary #
        ##################

        def image_form(float_img):
            img_int = tf.floor((float_img + 1) * 255 / 2)
            return tf.cast(img_int, tf.uint8)

        self.summary_hyperparameter = tf.summary.merge([
            tf.summary.scalar('hyperparameter_cyclic_lambda_a', self.__cyclic_lambda_a),
            tf.summary.scalar('hyperparameter_cyclic_lambda_b', self.__cyclic_lambda_b),
            tf.summary.scalar('hyperparameter_image_shape_w', self.__image_shape[0]),
            tf.summary.scalar('hyperparameter_image_shape_h', self.__image_shape[1]),
            tf.summary.scalar('hyperparameter_image_shape_c', self.__image_shape[2]),
            tf.summary.scalar('hyperparameter_buffer_size', self.__buffer_size),
            tf.summary.scalar('hyperparameter_batch', self.__batch)
        ])

        self.summary_train_gen = tf.summary.merge([
            tf.summary.scalar('meta_learning_rate', self.__learning_rate),
            tf.summary.scalar('train_gen_loss_a', self.gen_loss_a),
            tf.summary.scalar('train_gen_loss_b', self.gen_loss_b),
            tf.summary.scalar('train_cycle_loss_a', self.cycle_loss_a),
            tf.summary.scalar('train_cycle_loss_b', self.cycle_loss_b)
        ])

        self.summary_train_disc = tf.summary.merge([
            tf.summary.scalar('train_disc_loss_a', self.disc_loss_a),
            tf.summary.scalar('train_disc_loss_b', self.disc_loss_b),
            tf.summary.image('buffer_A', image_form(self.__fake_img_a_buffer)),
            tf.summary.image('buffer_B', image_form(self.__fake_img_b_buffer))
        ])

        self.summary_valid = tf.summary.merge([
            tf.summary.scalar('valid_gen_loss_a', self.gen_loss_a),
            tf.summary.scalar('valid_gen_loss_b', self.gen_loss_b),
            tf.summary.scalar('valid_cycle_loss_a', self.cycle_loss_a),
            tf.summary.scalar('valid_cycle_loss_b', self.cycle_loss_b)
        ])

        self.summary_image = tf.summary.merge([
            tf.summary.image('original_A', original_img_a, self.__log_img_size),
            tf.summary.image('fake_B', image_form(self.fake_img_b), self.__log_img_size),
            tf.summary.image('cycled_A', image_form(cycle_img_a), self.__log_img_size),
            tf.summary.image('original_B', original_img_b, self.__log_img_size),
            tf.summary.image('fake_A', image_form(self.fake_img_a), self.__log_img_size),
            tf.summary.image('cycled_B', image_form(cycle_img_b), self.__log_img_size)
        ])

    def train(self,
              epoch: int,
              progress_interval: int = 1,
              validation_batch: int = 100):

        def shuffle_data(data, seed=None):
            """shuffle array along first axis"""
            np.random.seed(seed)
            np.random.shuffle(data)
            return data

        def learning_rate_scheduler(current_lr, current_epoch):
            """ heuristic scheduler used in original paper """
            bias = 2.0e-6
            if current_epoch > 100:
                return np.max([current_lr - bias, 0])
            else:
                return current_lr

        if self.__warm_start:
            meta = np.load('%s/meta.npz' % self.__checkpoint_dir)
            learning_rate = meta['learning_rate']
            buffer_a = meta['buffer_a']
            buffer_b = meta['buffer_b']
            buffer_ind = meta['buffer_ind']
            ini_epoch = meta['epoch']
            i_summary_train_gen = meta['i_summary_train_gen']
            i_summary_train_disc = meta['i_summary_train_disc']
            i_summary_valid = meta['i_summary_valid']
        else:
            learning_rate = self.__ini_learning_rate
            buffer_a = np.zeros(tuple([self.__buffer_size] + self.__image_shape))
            buffer_b = np.zeros(tuple([self.__buffer_size] + self.__image_shape))
            buffer_ind = 0
            ini_epoch = 0
            i_summary_train_gen = 0
            i_summary_train_disc = 0
            i_summary_valid = 0

            # write hyperparameters to tensorboad
            sums = self.session.run(self.summary_hyperparameter)
            self.writer.add_summary(sums, 0)

        e = -1
        for e in range(ini_epoch, ini_epoch + epoch):
            self.session.run([
                self.iterator_gen_ini_a,
                self.iterator_gen_ini_b,
                self.iterator_disc_ini_a,
                self.iterator_disc_ini_b],
                feed_dict={
                    self.__tfrecord_a: '%s/trainA.tfrecord' % self.tfrecord_dir,
                    self.__tfrecord_b: '%s/trainB.tfrecord' % self.tfrecord_dir,
                    self.__seed: np.random.randint(0, 10000)
                }
            )

            n = 0

            learning_rate = learning_rate_scheduler(learning_rate, e)

            # TRAIN
            while True:
                n += 1
                try:
                    # train generator
                    returns = self.session.run([
                        self.fake_img_a,
                        self.fake_img_b,
                        self.summary_train_gen,
                        self.train_op_gen_a,
                        self.train_op_gen_b
                    ],
                        feed_dict={self.__learning_rate: learning_rate}
                    )
                    self.writer.add_summary(returns[2], i_summary_train_gen)
                    i_summary_train_gen += 1

                    if buffer_ind > self.__buffer_size - 1:
                        # TODO: this works with only batch size `1`. Extend in general case
                        if np.random.rand() > 0.5:
                            sampled_fake_a = returns[0]
                            sampled_fake_b = returns[1]
                        else:
                            # sample from buffered a
                            buffer_a = shuffle_data(buffer_a)
                            sampled_fake_a = buffer_a[0:1, :, :, :]
                            buffer_a[0, :, :, :] = returns[0][0]
                            # sample from buffered b
                            buffer_b = shuffle_data(buffer_b)
                            sampled_fake_b = buffer_b[0:1, :, :, :]
                            buffer_b[0, :, :, :] = returns[1][0]
                    else:
                        sampled_fake_a = returns[0]
                        sampled_fake_b = returns[1]
                        buffer_a[buffer_ind, :, :, :] = sampled_fake_a
                        buffer_b[buffer_ind, :, :, :] = sampled_fake_b
                        buffer_ind += 1

                    # train discriminator
                    returns = self.session.run([
                        self.summary_train_disc,
                        self.train_op_disc_a,
                        self.train_op_disc_b
                    ],
                        feed_dict={
                            self.__fake_img_a_buffer: sampled_fake_a,
                            self.__fake_img_b_buffer: sampled_fake_b
                        }
                    )
                    self.writer.add_summary(returns[0], i_summary_train_disc)
                    i_summary_train_disc += 1

                    if progress_interval is not None and n % progress_interval == 0:
                        print('epoch %i-%i\r' % (e, n), end='', flush=True)

                except tf.errors.OutOfRangeError:
                    print()
                    self.__log('epoch %i:' % e)
                    break

            # VALID
            self.session.run([self.iterator_gen_ini_a, self.iterator_gen_ini_b],
                             feed_dict={
                                 self.__tfrecord_a: '%s/testA.tfrecord' % self.tfrecord_dir,
                                 self.__tfrecord_b: '%s/testB.tfrecord' % self.tfrecord_dir,
                                 self.__batch: validation_batch,
                                 self.__seed: np.random.randint(0, 10000)
                             })

            while True:
                try:
                    returns = self.session.run(self.summary_valid)
                    self.writer.add_summary(returns, i_summary_valid)
                    i_summary_valid += 1

                except tf.errors.OutOfRangeError:
                    break

            self.session.run([self.iterator_gen_ini_a, self.iterator_gen_ini_b],
                             feed_dict={
                                 self.__tfrecord_a: '%s/testA.tfrecord' % self.tfrecord_dir,
                                 self.__tfrecord_b: '%s/testB.tfrecord' % self.tfrecord_dir,
                                 self.__batch: self.__log_img_size,
                                 self.__seed: np.random.randint(0, 10000)
                             })
            returns = self.session.run(self.summary_image)
            self.writer.add_summary(returns, e)

        self.__saver.save(self.session, self.__checkpoint)
        np.savez('%s/meta.npz' % self.__checkpoint_dir,
                 learning_rate=learning_rate,
                 buffer_a=buffer_a,
                 buffer_b=buffer_b,
                 buffer_ind=buffer_ind,
                 i_summary_train_gen=i_summary_train_gen,
                 i_summary_train_disc=i_summary_train_disc,
                 i_summary_valid=i_summary_valid,
                 epoch=e + 1)

    def __log(self, statement):
        if self.__logger is not None:
            self.__logger.info(statement)

