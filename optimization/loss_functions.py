import tensorflow as tf


def dice_loss(epsilon=1e-12):
    def soft_dice_coefficient(y_true, y_pred):
        _y_true = tf.keras.layers.Flatten()(y_true)
        _y_pred = tf.keras.layers.Flatten()(y_pred)
        intersection = tf.reduce_sum(_y_pred * _y_true)
        soft_dice = (2. * intersection + epsilon) / (tf.reduce_sum(_y_true, ) + tf.reduce_sum(_y_pred) + epsilon)
        return 1. - soft_dice
    return soft_dice_coefficient


def binary_weighted_loss(alpha, loss1, loss2):
    def loss(y_true, y_pred):
        return alpha * loss1(y_true, y_pred) + (1 - alpha) * loss2(y_true, y_pred)
    return loss
