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

    def __tfreocrd(self, record_name, batch, seed=None):
        seed = tf.cast(seed, tf.int64) if seed is not None else seed
        data_set_api = tf.data.TFRecordDataset(record_name, compression_type='GZIP')
        # convert record to tensor
        data_set_api = data_set_api.map(tfrecord_parser(self.__image_shape), self.__n_thread)
        # set batch size
        data_set_api = data_set_api.shuffle(buffer_size=10000, seed=seed)
        data_set_api = data_set_api.batch(tf.cast(batch, tf.int64))
        # make iterator
        iterator = tf.data.Iterator.from_structure(data_set_api.output_types, data_set_api.output_shapes)
        iterator_ini = iterator.make_initializer(data_set_api)
        return iterator, iterator_ini

    def __build_network(self):

        ##########
        # config #
        ##########
        self.__tfrecord_a = tf.placeholder(tf.string, name='tfrecord_a')
        self.__tfrecord_b = tf.placeholder(tf.string, name='tfrecord_b')
        self.__batch = tf.placeholder_with_default(self.__base_batch, [], name='batch')

        #########
        # input #
        #########
        img_iterator_a, self.iterator_ini_a = self.__tfreocrd(self.__tfrecord_a, self.__batch)
        img_iterator_b, self.iterator_ini_b = self.__tfreocrd(self.__tfrecord_b, self.__batch)
        self.img_a = img_iterator_a.get_next()
        self.img_b = img_iterator_b.get_next()

        ###############
        # placeholder #
        ###############
        img_shape = [None] + self.__image_shape
        # original images from domain A and B
        self.__original_img_a = tf.placeholder(tf.float32, img_shape, name='original_img_a')
        self.__original_img_b = tf.placeholder(tf.float32, img_shape, name='original_img_b')

        # will sampled from buffered generated images
        self.__fake_img_a_buffer = tf.placeholder(tf.float32, img_shape, name='fake_img_a_buffer')
        self.__fake_img_b_buffer = tf.placeholder(tf.float32, img_shape, name='fake_img_b_buffer')

        self.__learning_rate = tf.placeholder_with_default(0.0, [], name='learning_rate')

        original_img_a_norm = (self.__original_img_a / 255 - 0.5) * 2
        original_img_b_norm = (self.__original_img_b / 255 - 0.5) * 2

        #############
        # generator #
        #############
        with tf.name_scope('generators'):
            # generator from A to B
            self.fake_img_b = generator_resnet(original_img_a_norm, scope='generator_a')
            self.cycle_img_a = generator_resnet(self.fake_img_b, scope='generator_b')

            # generator from B to A
            self.fake_img_a = generator_resnet(original_img_b_norm, scope='generator_b', reuse=True)
            self.cycle_img_b = generator_resnet(self.fake_img_a, scope='generator_a', reuse=True)

            self.id_a = generator_resnet(original_img_a_norm, scope='generator_b', reuse=True)
            self.id_b = generator_resnet(original_img_b_norm, scope='generator_a', reuse=True)
            if self.__identity_lambda != 0.0:
                # self.id_a = generator_resnet(original_img_a_norm, scope='generator_b', reuse=True)
                id_loss_a = tf.reduce_mean(tf.abs(original_img_a_norm - self.id_a))

                # self.id_b = generator_resnet(original_img_b_norm, scope='generator_a', reuse=True)
                id_loss_b = tf.reduce_mean(tf.abs(original_img_b_norm - self.id_b))
            else:
                id_loss_a = id_loss_b = 0.0

        #################
        # discriminator #
        #################
        with tf.name_scope('discriminators'):
            # logit for update generator A
            logit_fake_a_generator = discriminator_patch(self.fake_img_a, scope='discriminator_a')
            # logit for update generator B
            logit_fake_b_generator = discriminator_patch(self.fake_img_b, scope='discriminator_b')

            # logit for update discriminator A
            logit_fake_a = discriminator_patch(self.__fake_img_a_buffer, scope='discriminator_a', reuse=True)
            logit_original_a = discriminator_patch(original_img_a_norm, scope='discriminator_a', reuse=True)

            # logit for update discriminator B
            logit_fake_b = discriminator_patch(self.__fake_img_b_buffer, scope='discriminator_b', reuse=True)
            logit_original_b = discriminator_patch(original_img_b_norm, scope='discriminator_b', reuse=True)

        ########
        # loss #
        ########
        with tf.name_scope('loss'):  # adversarial loss (least square loss, known as LSGAN)
            gen_loss_a = tf.reduce_mean(tf.squared_difference(logit_fake_a_generator, 1))
            gen_loss_b = tf.reduce_mean(tf.squared_difference(logit_fake_b_generator, 1))

            disc_loss_a = (tf.reduce_mean(tf.squared_difference(logit_fake_a, 0)) +
                           tf.reduce_mean(tf.squared_difference(logit_original_a, 1))) * 0.5
            disc_loss_b = (tf.reduce_mean(tf.squared_difference(logit_fake_b, 0)) +
                           tf.reduce_mean(tf.squared_difference(logit_original_b, 1))) * 0.5
            # cycle consistency loss
            cycle_loss_a = tf.reduce_mean(tf.abs(original_img_a_norm - self.cycle_img_a))
            cycle_loss_b = tf.reduce_mean(tf.abs(original_img_b_norm - self.cycle_img_b))

        ################
        # optimization #
        ################
        with tf.name_scope('optimization'):
            if self.__optimizer == 'adam':
                optimizer_g_a = tf.train.AdamOptimizer(self.__learning_rate)
                optimizer_g_b = tf.train.AdamOptimizer(self.__learning_rate)
                optimizer_d_a = tf.train.AdamOptimizer(self.__learning_rate)
                optimizer_d_b = tf.train.AdamOptimizer(self.__learning_rate)
            else:
                raise ValueError('unknown optimizer !!')

            var_gen_a = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_a')
            var_gen_b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_b')
            var_disc_a = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_a')
            var_disc_b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_b')

            cycle_loss = cycle_loss_a * self.__cyclic_lambda_a + cycle_loss_b * self.__cyclic_lambda_b
            self.train_op_gen_a = optimizer_g_a.minimize(
                gen_loss_b + cycle_loss + id_loss_b * self.__cyclic_lambda_a * self.__identity_lambda,
                var_list=var_gen_a)
            self.train_op_gen_b = optimizer_g_b.minimize(
                gen_loss_a + cycle_loss + id_loss_a * self.__cyclic_lambda_b * self.__identity_lambda,
                var_list=var_gen_b)
            self.train_op_disc_a = optimizer_d_a.minimize(
                disc_loss_a,
                var_list=var_disc_a)
            self.train_op_disc_b = optimizer_d_b.minimize(
                disc_loss_b,
                var_list=var_disc_b)

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
            tf.summary.scalar('hyperparameter_batch', self.__base_batch)
        ])

        self.summary_train_gen_a = tf.summary.merge([
            tf.summary.scalar('meta_learning_rate', self.__learning_rate),
            tf.summary.scalar('loss_gen_a', gen_loss_a),
            tf.summary.scalar('loss_cycle_a', cycle_loss_a),
            tf.summary.scalar('loss_id_a', id_loss_b)
        ])

        self.summary_train_gen_b = tf.summary.merge([
            tf.summary.scalar('loss_gen_b', gen_loss_b),
            tf.summary.scalar('loss_cycle_b', cycle_loss_b),
            tf.summary.scalar('loss_id_b', id_loss_a)
        ])

        self.summary_train_disc_a = tf.summary.merge([
            tf.summary.scalar('loss_disc_a', disc_loss_a),
            tf.summary.image('buffer_a', image_form(self.__fake_img_a_buffer))
        ])

        self.summary_train_disc_b = tf.summary.merge([
            tf.summary.scalar('loss_disc_b', disc_loss_b),
            tf.summary.image('buffer_b', image_form(self.__fake_img_b_buffer))
        ])

        self.summary_image = tf.summary.merge([
            tf.summary.image('original_a', self.__original_img_a, self.__log_img_size),
            tf.summary.image('fake_b', image_form(self.fake_img_b), self.__log_img_size),
            tf.summary.image('cycled_a', image_form(self.cycle_img_a), self.__log_img_size),
            tf.summary.image('original_b', self.__original_img_b, self.__log_img_size),
            tf.summary.image('fake_a', image_form(self.fake_img_a), self.__log_img_size),
            tf.summary.image('cycled_b', image_form(self.cycle_img_b), self.__log_img_size)
        ])

    def train(self, epoch: int, progress_interval: int = 1):

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
            i_summary = meta['i_summary']
        else:
            learning_rate = self.__ini_learning_rate
            buffer_a = np.zeros(tuple([self.__buffer_size] + self.__image_shape))
            buffer_b = np.zeros(tuple([self.__buffer_size] + self.__image_shape))
            buffer_ind = 0
            ini_epoch = 0
            i_summary = 0
            # write hyperparameters to tensorboad
            sums = self.session.run(self.summary_hyperparameter)
            self.writer.add_summary(sums, 0)

        e = -1
        for e in range(ini_epoch, ini_epoch + epoch):
            self.session.run([self.iterator_ini_a, self.iterator_ini_b],
                             feed_dict={
                                 self.__tfrecord_a: '%s/trainA.tfrecord' % self.tfrecord_dir,
                                 self.__tfrecord_b: '%s/trainB.tfrecord' % self.tfrecord_dir
                             })

            n = 0
            learning_rate = learning_rate_scheduler(learning_rate, e)

            # TRAIN
            while True:
                n += 1
                try:
                    # get input image
                    img_a, img_b = self.session.run(
                        [self.img_a, self.img_b],
                        feed_dict={self.__learning_rate: learning_rate})

                    # train generator A
                    summary, fake_img_a, _ = self.session.run([
                        self.summary_train_gen_a,
                        self.fake_img_a,
                        self.train_op_gen_a
                    ],
                        feed_dict={
                            self.__learning_rate: learning_rate,
                            self.__original_img_a: img_a,
                            self.__original_img_b: img_b}
                    )
                    self.writer.add_summary(summary, i_summary)
                    # train generator B
                    summary, fake_img_b, _ = self.session.run([
                        self.summary_train_gen_b,
                        self.fake_img_b,
                        self.train_op_gen_b
                    ],
                        feed_dict={
                            self.__learning_rate: learning_rate,
                            self.__original_img_a: img_a,
                            self.__original_img_b: img_b}
                    )
                    self.writer.add_summary(summary, i_summary)

                    # buffering generated images
                    if buffer_ind > self.__buffer_size - 1:
                        # TODO: this works with only batch size `1`. Extend in general case
                        if np.random.rand() > 0.5:
                            sampled_fake_a = fake_img_a
                            sampled_fake_b = fake_img_b
                        else:
                            # sample from buffered a
                            buffer_a = shuffle_data(buffer_a)
                            sampled_fake_a = buffer_a[0:1, :, :, :]
                            buffer_a[0, :, :, :] = fake_img_a[0]
                            # sample from buffered b
                            buffer_b = shuffle_data(buffer_b)
                            sampled_fake_b = buffer_b[0:1, :, :, :]
                            buffer_b[0, :, :, :] = fake_img_b[0]
                    else:
                        sampled_fake_a = fake_img_a
                        sampled_fake_b = fake_img_b
                        buffer_a[buffer_ind, :, :, :] = sampled_fake_a
                        buffer_b[buffer_ind, :, :, :] = sampled_fake_b
                        buffer_ind += 1

                    # train discriminator A
                    summary, _ = self.session.run([
                        self.summary_train_disc_a,
                        self.train_op_disc_a
                    ],
                        feed_dict={
                            self.__learning_rate: learning_rate,
                            self.__original_img_a: img_a,
                            self.__fake_img_a_buffer: sampled_fake_a}
                    )
                    self.writer.add_summary(summary, i_summary)
                    # train discriminator B
                    summary, _ = self.session.run([
                        self.summary_train_disc_b,
                        self.train_op_disc_b
                    ],
                        feed_dict={
                            self.__learning_rate: learning_rate,
                            self.__original_img_b: img_b,
                            self.__fake_img_b_buffer: sampled_fake_b}
                    )
                    self.writer.add_summary(summary, i_summary)

                    if progress_interval is not None and n % progress_interval == 0:
                        print('epoch %i-%i\r' % (e, n), end='', flush=True)

                    i_summary += 1

                except tf.errors.OutOfRangeError:
                    print()
                    self.__log('epoch %i:' % e)

                    # produce images from validation data
                    self.session.run([self.iterator_ini_a, self.iterator_ini_b],
                                     feed_dict={
                                         self.__tfrecord_a: '%s/testA.tfrecord' % self.tfrecord_dir,
                                         self.__tfrecord_b: '%s/testB.tfrecord' % self.tfrecord_dir,
                                         self.__batch: self.__log_img_size
                                     })
                    img_a, img_b = self.session.run([self.img_a, self.img_b])
                    summary = self.session.run(self.summary_image,
                                               feed_dict={self.__original_img_a: img_a, self.__original_img_b: img_b})
                    self.writer.add_summary(summary, e)
                    break

        self.__saver.save(self.session, self.__checkpoint)
        np.savez('%s/meta.npz' % self.__checkpoint_dir,
                 learning_rate=learning_rate,
                 buffer_a=buffer_a,
                 buffer_b=buffer_b,
                 buffer_ind=buffer_ind,
                 i_summary=i_summary,
                 epoch=e + 1)

    def generate_img(self, batch):
        """ Return generated img

        :param batch: number of img
        :return: 0~255, uint, numpy array
        [original_a, fake_from_a, cycle_a, identity_a, original_b, fake_from_b, cycle_b, identity_b]
        """

        def form_img(target_array):
            target_array = (target_array+1)/2*255
            target_array = target_array.astype(np.uint8)
            return target_array

        self.session.run([self.iterator_ini_a, self.iterator_ini_b],
                         feed_dict={
                             self.__tfrecord_a: '%s/testA.tfrecord' % self.tfrecord_dir,
                             self.__tfrecord_b: '%s/testB.tfrecord' % self.tfrecord_dir,
                             self.__batch: 1
                         })
        result = []
        for b in range(batch):
            img_a, img_b = self.session.run([self.img_a, self.img_b])
            imgs = self.session.run([
                self.fake_img_b, self.cycle_img_a, self.id_a,
                self.fake_img_a, self.cycle_img_b, self.id_b
            ],
                feed_dict={self.__original_img_a: img_a, self.__original_img_b: img_b}
            )
            result.append([
                img_a.astype(np.uint8), form_img(imgs[0]), form_img(imgs[1]), form_img(imgs[2]),
                img_b.astype(np.uint8), form_img(imgs[3]), form_img(imgs[4]), form_img(imgs[5])
            ])
        return result

    def __log(self, statement):
        if self.__logger is not None:
            self.__logger.info(statement)

