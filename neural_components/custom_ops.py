import tensorflow as tf

def sample_min_max_scaling(x, minimum_value=1, maximum_value=256):
    min_ = tf.reduce_min(x, axis=[0, 1, 2])[tf.newaxis, tf.newaxis, tf.newaxis, :]
    max_ = tf.reduce_max(x, axis=[0, 1, 2])[tf.newaxis, tf.newaxis, tf.newaxis, :]
    denominator_ = tf.clip_by_value(max_ - min_,
                                    clip_value_min=10e-8,
                                    clip_value_max=10e8)  # Following original implementation
    return minimum_value + (((x - min_) / denominator_) * (maximum_value - minimum_value))


def min_max_general(x, target_max, target_min):
    x_min = tf.reduce_min(x, axis=[0, 1, 2])[tf.newaxis, tf.newaxis, tf.newaxis, :]
    x_max = tf.reduce_max(x, axis=[0, 1, 2])[tf.newaxis, tf.newaxis, tf.newaxis, :]
    denominator = tf.clip_by_value(x_max - x_min,
                                    clip_value_min=10e-8,
                                    clip_value_max=10e8)
    x_standard = (x - x_min) / denominator
    return (x_standard * (target_max - target_min)) + target_max
