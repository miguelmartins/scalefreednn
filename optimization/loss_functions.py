import tensorflow as tf


# def dice_loss(epsilon=1e-12):  # DOES NOT ACCOUNT FOR TF.SUM_BATCH REDUCTION
#     def soft_dice_coefficient(y_true, y_pred):
#         _y_true = tf.keras.layers.Flatten()(y_true)
#         _y_pred = tf.keras.layers.Flatten()(y_pred)
#         intersection = tf.reduce_sum(_y_pred * _y_true)
#         soft_dice = (2. * intersection + epsilon) / (tf.reduce_sum(_y_true) + tf.reduce_sum(_y_pred) + epsilon)
#         return 1. - soft_dice
#     return soft_dice_coefficient

def dice_loss(epsilon=1e-12):  # ASSUMES TF.SUM_OVER_BATCH REDUCTION
    def soft_dice_coefficient(y_true, y_pred):
        _y_true = tf.keras.layers.Flatten()(y_true)  # B x H x W x C -> B x (H.W.C)
        _y_pred = tf.keras.layers.Flatten()(y_pred)
        intersection = tf.reduce_sum(_y_pred * _y_true, axis=-1)  # B x (H.W.C) -> B
        soft_dice = (2. * intersection + epsilon) / (tf.reduce_sum(_y_true, axis=-1) + tf.reduce_sum(_y_pred, axis=-1) + epsilon)
        return tf.reduce_mean(1. - soft_dice)  # B -> 1  # AS IN TF.SUM_OVER_BATCH SIZE REDUCTION
    return soft_dice_coefficient


def dice_loss_multi(epsilon=1e-12):  # todo: CHECK https://github.com/Beckschen/TransUNet/blob/main/utils.py
    def soft_dice_coefficient_multi(y_true, y_pred):
        shape_ = y_true.shape
        _y_true = tf.keras.layers.Reshape([shape_[1] * shape_[2], shape_[3]])(y_true)
        _y_pred = tf.keras.layers.Reshape([shape_[1] * shape_[2], shape_[3]])(y_pred)
        intersection = tf.reduce_sum(_y_pred * _y_true, axis=1)
        soft_dice = (2. * intersection + epsilon) / (tf.reduce_sum(_y_true, axis=1) + tf.reduce_sum(_y_pred, axis=1) + epsilon)
        soft_dice_loss = 1. - soft_dice
        avg_soft_dice_loss = tf.reduce_mean(soft_dice_loss, axis=-1)
        return tf.reduce_sum(avg_soft_dice_loss, axis=0) / shape_[0] # SUM_OVER_BATCH REDUCTION
    return soft_dice_coefficient_multi


def weighted_loss(alpha, loss1, loss2):
    def loss(y_true, y_pred):
        return alpha * loss1(y_true, y_pred) + (1 - alpha) * loss2(y_true, y_pred)
    return loss
