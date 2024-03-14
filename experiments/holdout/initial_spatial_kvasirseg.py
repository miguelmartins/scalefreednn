import numpy as np
import os
import tensorflow as tf

from depracated.spatial_ssr import get_spatial_alpha_ols_encoders_unet
from etl.preprocessing import get_segmentation_data
from config.parser import ExperimentConfigParser
from sklearn.model_selection import KFold
import os
print(os.getcwd())
CONFIG_FILE = '/home/miguelmartins/Projects/FractalFCN/config/kvasir-seg_baseline.yaml'
LOG_DIR = '/home/miguelmartins/Projects/FractalFCN/logs_scale_free/holdout/kvasir'
DATA_PATH = '/home/miguelmartins/Datasets/kvasir-seg/Kvasir-SEG/'

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

    train_idx = 800
    dev_idx = 900
    x_train, y_train = dataset_np_x[:train_idx], dataset_np_y[:train_idx]
    x_dev, y_dev = dataset_np_x[train_idx:dev_idx], dataset_np_y[train_idx:dev_idx]
    x_val, y_val = dataset_np_x[dev_idx:], dataset_np_y[dev_idx:]

    model = get_spatial_alpha_ols_encoders_unet(channels_per_level=config_data.config.model.level_depth,
                                            input_shape=config_data.config.data.target_size + [3],
                                            with_bn=False)
    model.compile(loss=config_data.loss_object,
                  optimizer=config_data.optimizer_obj,
                  metrics=config_data.metrics)
    print("LOCAL!")
    model.fit(x=x_train,
              y=y_train,
              epochs=config_data.config.training.epochs,
              batch_size=config_data.config.training.batch_size,
              validation_data=(x_dev, y_dev),
              callbacks=config_data.callbacks
              )
    model.load_weights(config_data.model_checkpoint_path)
    model.evaluate(x=x_val, y=y_val, callbacks=config_data.test_callbacks)
    # config_data.dump_config(description=f'spatial holdout')


if __name__ == '__main__':
    main()
