import numpy as np
import tensorflow as tf
from optimization.loss_functions import dice_loss_multi, dice_loss, weighted_loss


def dice_coefficient(y_true, y_pred, epsilon=1e-12):
    # adapted from: https://stackoverflow.com/questions/72195156/correct-implementation-of-dice-loss-in-tensorflow-keras
    _y_true = tf.keras.layers.Flatten()(y_true)
    _y_pred = tf.where(tf.keras.layers.Flatten()(y_pred) >= 0.5, 1., 0.)
    intersection = tf.reduce_sum(_y_pred * _y_true, axis=-1)
    return (2. * intersection + epsilon) / (tf.reduce_sum(_y_true, axis=-1) + tf.reduce_sum(_y_pred, axis=-1) + epsilon)

def dice_wrong_coefficient(y_true, y_pred, epsilon=1e-12):
    # adapted from: https://stackoverflow.com/questions/72195156/correct-implementation-of-dice-loss-in-tensorflow-keras
    _y_true = tf.keras.layers.Flatten()(y_true)
    _y_pred = tf.where(tf.keras.layers.Flatten()(y_pred) >= 0.5, 1., 0.)
    intersection = tf.reduce_sum(_y_pred * _y_true, axis=-1)
    return (2. * intersection + epsilon) / (tf.reduce_sum(_y_true) + tf.reduce_sum(_y_pred) + epsilon)



if __name__ == '__main__':
    with tf.device('/cpu:0'):
        smooth = 10e-6
        y_pred = np.zeros((2, 128, 128, 1))
        # one pixel is set to 1
        y_pred[0, 1, 0, 0] = 1

        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        y_true = np.zeros_like(y_pred)
        y_true[0, 1, 0, 0] = 1
        y_true[0, 0, 1, 0] = 1
        print(tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])(np.zeros((3, 128, 128, 1)), np.zeros((3, 128, 128, 1))))
        iou = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])
        iou.update_state(y_pred=y_pred, y_true=y_true)
        print("iou", iou.result())
        iou.reset_states()

        dice_metric = tf.keras.metrics.MeanMetricWrapper(fn=dice_coefficient, name='dice_score')
        dice_metric.update_state(y_pred, y_true)
        print("dice metric wrapper", dice_metric.result())
        dice_metric.reset_states()

        dice_wrong = tf.keras.metrics.MeanMetricWrapper(fn=dice_wrong_coefficient, name='dice_wrong')
        dice_wrong.update_state(y_pred=y_pred, y_true=y_true)
        print("dice_wrong metric wrapper", dice_wrong.result())
        dice_wrong.reset_states()


        dice_loss = dice_loss()
        print("d", dice_loss(y_true, y_pred))
    #print("dice", dice2(y_true, y_pred))
