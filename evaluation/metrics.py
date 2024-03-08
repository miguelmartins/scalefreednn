import tensorflow as tf
from keras import backend as K


def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())


def dice_coefficient(y_true, y_pred, epsilon=1e-12):
    # adapted from: https://stackoverflow.com/questions/72195156/correct-implementation-of-dice-loss-in-tensorflow-keras
    _y_true = tf.keras.layers.Flatten()(y_true)
    _y_pred = tf.where(tf.keras.layers.Flatten()(y_pred) >= 0.5, 1., 0.)
    intersection = tf.reduce_sum(_y_pred * _y_true)
    return (2. * intersection + epsilon) / (tf.reduce_sum(_y_true, ) + tf.reduce_sum(_y_pred) + epsilon)


def get_baseline_segmentation_metrics():
    accuracy = tf.keras.metrics.BinaryAccuracy()
    auc = tf.keras.metrics.AUC()
    prec = tf.keras.metrics.Precision()
    sens = tf.keras.metrics.Recall(name='sensitivity')
    spec = specificity
    iou = tf.keras.metrics.BinaryIoU()
    # dice = tf.keras.metrics.F1Score(name='dice')
    return [accuracy, auc, prec, sens, spec, dice_coefficient, iou]
