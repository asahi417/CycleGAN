import os
import logging
from glob import glob
import json
import tensorflow as tf


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    https://www.tensorflow.org/guide/summaries_and_tensorboard """
    with tf.name_scope('var_%s' % name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def raise_error(condition, msg):
    if condition:
        raise ValueError(msg)


def create_log(out_file_path=None):
    """ Logging
        If `out_file_path` is None, only show in terminal
        or else save log file in `out_file_path`
    Usage
    -------------------
    logger.info(message)
    logger.error(error)
    """

    # handler to record log to a log file
    if out_file_path is not None:
        if os.path.exists(out_file_path):
            os.remove(out_file_path)
        logger = logging.getLogger(out_file_path)

        if len(logger.handlers) > 1:  # if there are already handler, return it
            return logger
        else:
            logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s")

            handler = logging.FileHandler(out_file_path)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            logger_stream = logging.getLogger()
            # check if some stream handlers are already
            if len(logger_stream.handlers) > 0:
                return logger
            else:
                handler = logging.StreamHandler()
                handler.setFormatter(formatter)
                logger.addHandler(handler)

                return logger
    else:
        # handler to output
        handler = logging.StreamHandler()
        logger = logging.getLogger()

        if len(logger.handlers) > 0:  # if there are already handler, return it
            return logger
        else:  # in case of no, make new output handler
            logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            return logger


def checkpoint_version(checkpoint_dir: str,
                       config: dict = None,
                       version: int = None,
                       sub_version: int = None
                       ):
    """ Checkpoint versioner:
    - return checkpoint dir, which has same hyper parameter (config)
    - return checkpoint dir, which corresponds to the version
    - return new directory
    :param config:
    :param checkpoint_dir: `./checkpoint/lam
    :param version: number of checkpoint
    :return:
    """

    if version is not None:
        version = '%i' % version if sub_version is None else '%i.%i' % (version, sub_version)
        checkpoints = glob('%s/v%s/hyperparameters.json' % (checkpoint_dir, version))
        if len(checkpoints) == 0:
            raise ValueError('No checkpoint: %s, %s' % (checkpoint_dir, version))
        elif len(checkpoints) > 1:
            raise ValueError('Multiple checkpoint found: %s, %s' % (checkpoint_dir, version))
        else:
            parameter = json.load(open(checkpoints[0]))
            target_checkpoints_dir = checkpoints[0].replace('/hyperparameters.json', '')
            return target_checkpoints_dir, parameter

    elif config is not None:
        # check if there are any checkpoints with same hyperparameters
        target_checkpoints = []
        for i in glob('%s/*' % checkpoint_dir):
            json_dict = json.load(open('%s/hyperparameters.json' % i))
            if config == json_dict:
                target_checkpoints.append(i)
        if len(target_checkpoints) == 1:
            return target_checkpoints[0], config
        elif len(target_checkpoints) == 0:
            new_checkpoint_id = len(glob('%s/*' % checkpoint_dir))
            new_checkpoint_path = '%s/v%i' % (checkpoint_dir, new_checkpoint_id)
            os.makedirs(new_checkpoint_path, exist_ok=True)
            with open('%s/hyperparameters.json' % new_checkpoint_path, 'w') as outfile:
                json.dump(config, outfile)
            return new_checkpoint_path, config
        else:
            raise ValueError('Checkpoints are duplicated')