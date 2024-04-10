import tensorflow as tf
from keras import backend as K


def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())


def dice_coefficient(y_true, y_pred, epsilon=1e-12):
    print("Shape of input", y_true.shape)
    print(y_true)
    # adapted from: https://stackoverflow.com/questions/72195156/correct-implementation-of-dice-loss-in-tensorflow-keras
    _y_true = tf.keras.layers.Flatten()(y_true)
    _y_pred = tf.where(tf.keras.layers.Flatten()(y_pred) >= 0.5, 1., 0.)
    intersection = tf.reduce_sum(_y_pred * _y_true, axis=-1)
    return (2. * intersection + epsilon) / (tf.reduce_sum(_y_true, axis=-1) + tf.reduce_sum(_y_pred, axis=-1) + epsilon)


def dice_coefficient_multi(y_true, y_pred, epsilon=1e-12):
    shape_ = y_true.shape
    _y_true = tf.keras.layers.Reshape([shape_[1] * shape_[2], shape_[3]])(y_true)
    _y_pred = tf.where(tf.keras.layers.Reshape([shape_[1] * shape_[2], shape_[3]])(y_pred) >= 0.5, 1., 0)
    intersection = tf.reduce_sum(_y_pred * _y_true, axis=1)
    return (2. * intersection + epsilon) / (
                tf.reduce_sum(_y_true, axis=1) + tf.reduce_sum(_y_pred, axis=1) + epsilon)


def avg_dice_coefficient_multi(y_true, y_pred, epsilon=1e-12):
    return tf.reduce_mean(dice_coefficient_multi(y_true, y_pred, epsilon), axis=-1)


def get_baseline_segmentation_metrics():
    accuracy = tf.keras.metrics.BinaryAccuracy()
    auc = tf.keras.metrics.AUC()
    prec = tf.keras.metrics.Precision()
    sens = tf.keras.metrics.Recall(name='sensitivity')
    spec = tf.keras.metrics.MeanMetricWrapper(specificity, name='specificity')
    dice_score = tf.keras.metrics.MeanMetricWrapper(fn=dice_coefficient, name='dice_score')
    iou = tf.keras.metrics.BinaryIoU(num_classes=2, target_class_ids=[1])
    return [accuracy, auc, prec, sens, spec, dice_score, iou]


def get_baseline_segmentation_metrics_multi():
    return [avg_dice_coefficient_multi]
