import tensorflow as tf


def get_segmentation_data(*, img_path, msk_path, batch_size, target_size):
    image_gen = tf.keras.utils.image_dataset_from_directory(
        img_path,
        labels=None,
        label_mode=None,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=False,
        image_size=target_size,
        subset=None,
        interpolation='bilinear',
    )

    mask_gen = tf.keras.utils.image_dataset_from_directory(
        msk_path,
        labels=None,
        label_mode=None,
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=False,
        image_size=target_size,
        subset=None,
        interpolation='nearest',
    )

    return tf.data.Dataset.zip((image_gen, mask_gen))
