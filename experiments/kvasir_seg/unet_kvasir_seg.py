import numpy as np
import os
import tensorflow as tf

from data.preprocessing import get_segmentation_data
from models.unet import get_unet
from config.parser import ExperimentConfigParser
from sklearn.model_selection import KFold

CONFIG_FILE = '/config/files/ten_fold_config.yaml'
LOG_DIR = '/path/to/logdir'
DATA_PATH = '/path/to/Kvasir-SEG/'

NUM_FOLDS = 10


def normalize_fn(x, y, minimum_value=0., maximum_value=255.):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)

    x = tf.clip_by_value(x, clip_value_min=minimum_value, clip_value_max=maximum_value)
    y = tf.clip_by_value(y, clip_value_min=minimum_value, clip_value_max=maximum_value)

    return (x - minimum_value) / (maximum_value - minimum_value), (y - minimum_value) / (maximum_value - minimum_value)


def main():
    config_data = ExperimentConfigParser(name=f'kvasir-seg_pid{os.getpid()}',
                                         config_path=CONFIG_FILE,
                                         log_dir=LOG_DIR)
    dataset = get_segmentation_data(img_path=os.path.join(DATA_PATH, 'images'),
                                msk_path=os.path.join(DATA_PATH, 'masks'),
                                batch_size=1,
                                target_size=config_data.config.data.target_size)
    dataset = dataset.map(normalize_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset_np_x = np.array([x for (x, _) in dataset]).squeeze(axis=1)
    dataset_np_y = np.array([y for (_, y) in dataset]).squeeze(axis=1)

    kf = KFold(n_splits=NUM_FOLDS, shuffle=False)
    for fold, (train_index, val_index) in enumerate(kf.split(dataset_np_x)):
        print(f"Fold {fold} starting...")
        fold_data = ExperimentConfigParser(
            name=f'unet-{NUM_FOLDS}-folds-{fold}-kvasir-seg_pid{os.getpid()}',
            config_path=CONFIG_FILE,
            log_dir=LOG_DIR)
        model = get_unet(channels_per_level=fold_data.config.model.level_depth,
                         input_shape=fold_data.config.data.target_size + [3],
                         with_bn=False)
        model.compile(loss=fold_data.loss_object,
                      optimizer=fold_data.optimizer_obj,
                      metrics=fold_data.metrics)

        x_fold, y_fold = dataset_np_x[train_index], dataset_np_y[train_index]
        x_val, y_val = dataset_np_x[val_index], dataset_np_y[val_index]
        dev_size = int(len(x_fold) * 0.1)
        dev_idx = len(x_fold) - dev_size
        x_train, y_train = x_fold[:dev_idx], y_fold[:dev_idx]
        x_dev, y_dev = x_fold[dev_idx:], y_fold[dev_idx:]

        model.fit(x=x_train,
                  y=y_train,
                  epochs=fold_data.config.training.epochs,
                  batch_size=fold_data.config.training.batch_size,
                  validation_data=(x_dev, y_dev),
                  callbacks=fold_data.callbacks
                  )
        model.load_weights(fold_data.model_checkpoint_path)
        model.evaluate(x=x_val, y=y_val, callbacks=fold_data.test_callbacks)
        fold_data.dump_config(description=f'fold {fold}')


if __name__ == '__main__':
    main()
