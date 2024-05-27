import tensorflow as tf
import numpy as np

from neural_components.custom_ops import sample_min_max_scaling, standardized_general_cosine_similarity
from neural_components.fractal_geometry import LocalSingularityStrength, OrdinaryLeastSquares, \
    WeightedLocalSingularityStrength, SoftHistogramLayer, SoftHistogramLayerL2, BinaryHistogramLayer, \
    LocalSingularityStrengthGN, SoftHistogramLayerGN


class SingularityStrengthRecalibration(tf.keras.layers.Layer):
    def __init__(self, r, max_scale, **kwargs):
        super(SingularityStrengthRecalibration, self).__init__(**kwargs)
        self.alpha_layer = LocalSingularityStrength(max_scale=max_scale)
        self.r = r

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        super(SingularityStrengthRecalibration, self).build(input_shape)

    def call(self, x):
        alphas = self.alpha_layer(x)
        squeeze = self.gap(alphas)
        excite = self.w2(self.w1(squeeze))
        return x * excite[:, tf.newaxis, tf.newaxis, :]


class SingularityStrengthRecalibrationGN(tf.keras.layers.Layer):
    def __init__(self, r, max_scale, **kwargs):
        super(SingularityStrengthRecalibrationGN, self).__init__(**kwargs)
        self.alpha_layer = LocalSingularityStrengthGN(max_scale=max_scale)
        self.r = r

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        super(SingularityStrengthRecalibrationGN, self).build(input_shape)

    def call(self, x):
        alphas = self.alpha_layer(x)
        squeeze = self.gap(alphas)
        excite = self.w2(self.w1(squeeze))
        return x * excite[:, tf.newaxis, tf.newaxis, :]


class WeightedSingularityStrengthRecalibration(tf.keras.layers.Layer):
    def __init__(self, r, max_scale, **kwargs):
        super(WeightedSingularityStrengthRecalibration, self).__init__(**kwargs)
        self.alpha_layer = WeightedLocalSingularityStrength(max_scale=max_scale)
        self.r = r

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        super(WeightedSingularityStrengthRecalibration, self).build(input_shape)

    def call(self, x):
        alphas = self.alpha_layer(x)
        squeeze = self.gap(alphas)
        excite = self.w2(self.w1(squeeze))
        return x * excite[:, tf.newaxis, tf.newaxis, :]


class SpatialSingularityStrengthRecalibration(tf.keras.layers.Layer):
    def __init__(self, r, max_scale, **kwargs):
        super(SpatialSingularityStrengthRecalibration, self).__init__(**kwargs)
        self.alpha_layer = LocalSingularityStrength(max_scale=max_scale)
        self.r = r

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        self.beta = self.add_weight(name='beta',
                                    shape=(1),
                                    initializer=tf.keras.initializers.Zeros(),
                                    dtype=tf.float32,
                                    trainable=True)
        super(SpatialSingularityStrengthRecalibration, self).build(input_shape)

    def call(self, x):
        alphas = self.alpha_layer(x)
        squeeze = self.gap(alphas)
        excite = self.w2(self.w1(squeeze))
        return (x * excite[:, tf.newaxis, tf.newaxis, :]) + (self.beta * (tf.nn.sigmoid(alphas)))


class HistogramSingularityStrengthRecalibration(tf.keras.layers.Layer):
    def __init__(self, max_scale, k, per_channel, **kwargs):
        super(HistogramSingularityStrengthRecalibration, self).__init__(**kwargs)
        self.alpha_layer = WeightedLocalSingularityStrength(max_scale=max_scale)
        self.soft_histogram = SoftHistogramLayer(num_anchors=k, per_channel=per_channel)

    def build(self, input_shape):
        num_channels = input_shape[-1]
        #self.beta = self.add_weight(name='beta',
        #                            shape=(1),
        #                            initializer=tf.keras.initializers.Zeros(),
        #                            dtype=tf.float32,
        #                            trainable=True)
        super(HistogramSingularityStrengthRecalibration, self).build(input_shape)

    def call(self, x):
        alphas = self.alpha_layer(x)
        soft_counts = tf.reduce_sum(tf.nn.relu(self.soft_histogram(alphas)),
                                    axis=-1)  # TODO: not sure about this ReLU; but we have bn before!
        return x + tf.nn.sigmoid(soft_counts)


class UnweightedFullSSR(tf.keras.layers.Layer):
    def __init__(self, max_scale, r, k, per_channel, **kwargs):
        super(UnweightedFullSSR, self).__init__(**kwargs)
        self.alpha_layer = LocalSingularityStrength(max_scale=max_scale)
        self.r = r
        self.soft_histogram = SoftHistogramLayer(num_anchors=k, per_channel=per_channel)
        self.concat_layer = tf.keras.layers.Concatenate(axis=-1)
        self.bn = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        super(UnweightedFullSSR, self).build(input_shape)

    def call(self, x, training=False):
        alphas = self.alpha_layer(x)
        soft_counts = tf.reduce_sum(tf.nn.relu(self.soft_histogram(alphas)),
                                    axis=-1)  # TODO: not sure about this ReLU; but we have bn before!
        squeeze = self.gap(alphas)
        excite = self.w2(self.w1(squeeze))
        # alternative (x + hist) * excite
        return self.concat_layer([x + tf.nn.sigmoid(soft_counts), x * excite[:, tf.newaxis, tf.newaxis, :]])


class UnweightedHistogramSingularityStrengthRecalibration(tf.keras.layers.Layer):
    def __init__(self, max_scale, k, per_channel, **kwargs):
        super(UnweightedHistogramSingularityStrengthRecalibration, self).__init__(**kwargs)
        self.alpha_layer = LocalSingularityStrength(max_scale=max_scale)
        self.soft_histogram = SoftHistogramLayer(num_anchors=k, per_channel=per_channel)
        self.bn = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        num_channels = input_shape[-1]
        super(UnweightedHistogramSingularityStrengthRecalibration, self).build(input_shape)

    def call(self, x, training=False):
        alphas = self.alpha_layer(x)
        soft_counts = tf.reduce_sum(tf.nn.relu(self.soft_histogram(alphas)),
                                    axis=-1)  # TODO: not sure about this ReLU; but we have bn before!
        #return x + tf.nn.sigmoid(soft_counts)
        # 1d beta sucked dick
        return x + tf.nn.sigmoid(soft_counts)


class UnweightedHistogramSingularityStrengthRecalibrationGN(tf.keras.layers.Layer):
    def __init__(self, max_scale, k, per_channel, **kwargs):
        super(UnweightedHistogramSingularityStrengthRecalibrationGN, self).__init__(**kwargs)
        self.alpha_layer = LocalSingularityStrengthGN(max_scale=max_scale)
        self.soft_histogram = SoftHistogramLayerGN(num_anchors=k, per_channel=per_channel)

    def build(self, input_shape):
        num_channels = input_shape[-2]
        super(UnweightedHistogramSingularityStrengthRecalibrationGN, self).build(input_shape)

    def call(self, x, training=False):
        alphas = self.alpha_layer(x)
        soft_counts = tf.reduce_sum(tf.nn.relu(self.soft_histogram(alphas)),
                                    axis=-1)  # TODO: not sure about this ReLU; but we have bn before!
        #return x + tf.nn.sigmoid(soft_counts)
        # 1d beta sucked dick
        return x + tf.nn.sigmoid(soft_counts)


class BinaryHistogramSingularityStrengthRecalibration(tf.keras.layers.Layer):
    def __init__(self, max_scale, per_channel, **kwargs):
        super(BinaryHistogramSingularityStrengthRecalibration, self).__init__(**kwargs)
        self.alpha_layer = LocalSingularityStrength(max_scale=max_scale)
        self.soft_histogram = BinaryHistogramLayer(per_channel=per_channel)
        self.bn = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        super(BinaryHistogramSingularityStrengthRecalibration, self).build(input_shape)

    def call(self, x, training=False):
        alphas = self.alpha_layer(x)
        return x + (x * tf.nn.sigmoid(self.soft_histogram(alphas)))


class NosigUnweightedHistogramSingularityStrengthRecalibration(tf.keras.layers.Layer):
    def __init__(self, max_scale, k, per_channel, **kwargs):
        super(NosigUnweightedHistogramSingularityStrengthRecalibration, self).__init__(**kwargs)
        self.alpha_layer = LocalSingularityStrength(max_scale=max_scale)
        self.soft_histogram = SoftHistogramLayer(num_anchors=k, per_channel=per_channel)
        self.bn = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        num_channels = input_shape[-1]
        super(NosigUnweightedHistogramSingularityStrengthRecalibration, self).build(input_shape)

    def call(self, x, training=False):
        alphas = self.alpha_layer(x)
        soft_counts = tf.reduce_sum(self.soft_histogram(alphas),
                                    axis=-1)  # TODO: not sure about this ReLU; but we have bn before!
        #return x + tf.nn.sigmoid(soft_counts)
        # 1d beta sucked dick
        return x + tf.nn.sigmoid(soft_counts)


class MSHistogramSingularityStrengthRecalibration(tf.keras.layers.Layer):
    def __init__(self, r, max_scale, k, per_channel, **kwargs):
        super(MSHistogramSingularityStrengthRecalibration, self).__init__(**kwargs)
        # used to be weighted
        self.alpha_layer = LocalSingularityStrength(max_scale=max_scale)
        self.soft_histogram = SoftHistogramLayer(num_anchors=k, per_channel=per_channel)
        self.r = r

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        super(MSHistogramSingularityStrengthRecalibration, self).build(input_shape)

    def call(self, x):
        alphas = self.alpha_layer(x)  # TODO: O SSR E WEIGHTED AQUI
        soft_counts = tf.reduce_sum(tf.nn.relu(self.soft_histogram(alphas)),
                                    axis=-1)  # TODO: not sure about this ReLU; but we have bn before!
        squeeze = self.gap(alphas)
        excite = self.w2(self.w1(squeeze))[:, tf.newaxis, tf.newaxis, :]
        return x * tf.nn.sigmoid(soft_counts + excite)


class MSHistogramNewSSRSingularityStrengthRecalibration(tf.keras.layers.Layer):
    def __init__(self, r, max_scale, k, per_channel, **kwargs):
        super(MSHistogramNewSSRSingularityStrengthRecalibration, self).__init__(**kwargs)
        self.alpha_layer = WeightedLocalSingularityStrength(max_scale=max_scale)
        self.soft_histogram = SoftHistogramLayer(num_anchors=k, per_channel=per_channel)
        self.r = r

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        super(MSHistogramNewSSRSingularityStrengthRecalibration, self).build(input_shape)

    def call(self, x):
        alphas = self.alpha_layer(x)  # TODO: O SSR E WEIGHTED AQUI
        soft_counts = tf.reduce_sum(tf.nn.relu(self.soft_histogram(alphas)),
                                    axis=-1)  # TODO: not sure about this ReLU; but we have bn before!
        squeeze = self.gap(soft_counts)
        excite = self.w2(self.w1(squeeze))[:, tf.newaxis, tf.newaxis, :]
        return x * tf.nn.sigmoid(soft_counts + excite)


class UnweightedMSHistogramSingularityStrengthRecalibration(tf.keras.layers.Layer):
    def __init__(self, r, max_scale, k, per_channel, **kwargs):
        super(UnweightedMSHistogramSingularityStrengthRecalibration, self).__init__(**kwargs)
        self.alpha_layer = LocalSingularityStrength(max_scale=max_scale)
        self.soft_histogram = SoftHistogramLayer(num_anchors=k, per_channel=per_channel)
        self.r = r

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        super(UnweightedMSHistogramSingularityStrengthRecalibration, self).build(input_shape)

    def call(self, x):
        alphas = self.alpha_layer(x)  # TODO: O SSR E WEIGHTED AQUI
        soft_counts = tf.reduce_sum(tf.nn.relu(self.soft_histogram(alphas)),
                                    axis=-1)  # TODO: not sure about this ReLU; but we have bn before!
        squeeze = self.gap(alphas)
        excite = self.w2(self.w1(squeeze))[:, tf.newaxis, tf.newaxis, :]
        return x * tf.nn.sigmoid(soft_counts + excite)


class L2HistogramSingularityStrengthRecalibration(tf.keras.layers.Layer):
    def __init__(self, max_scale, k, per_channel, **kwargs):
        super(L2HistogramSingularityStrengthRecalibration, self).__init__(**kwargs)
        self.alpha_layer = WeightedLocalSingularityStrength(max_scale=max_scale)
        self.soft_histogram = SoftHistogramLayerL2(num_anchors=k, per_channel=per_channel)

    def build(self, input_shape):
        num_channels = input_shape[-1]
        #self.beta = self.add_weight(name='beta',
        #                            shape=(1),
        #                            initializer=tf.keras.initializers.Zeros(),
        #                            dtype=tf.float32,
        #                            trainable=True)
        super(L2HistogramSingularityStrengthRecalibration, self).build(input_shape)

    def call(self, x):
        alphas = self.alpha_layer(x)
        soft_counts = tf.reduce_sum(tf.nn.relu(self.soft_histogram(alphas)),
                                    axis=-1)  # TODO: not sure about this ReLU; but we have bn before!
        return x + tf.nn.sigmoid(soft_counts)


class NoSigHistogramSingularityStrengthRecalibration(tf.keras.layers.Layer):
    def __init__(self, max_scale, k, per_channel, **kwargs):
        super(NoSigHistogramSingularityStrengthRecalibration, self).__init__(**kwargs)
        self.alpha_layer = WeightedLocalSingularityStrength(max_scale=max_scale)
        self.soft_histogram = SoftHistogramLayer(num_anchors=k, per_channel=per_channel)

    def build(self, input_shape):
        num_channels = input_shape[-1]
        #self.beta = self.add_weight(name='beta',
        #                            shape=(1),
        #                            initializer=tf.keras.initializers.Zeros(),
        #                            dtype=tf.float32,
        #                            trainable=True)
        super(NoSigHistogramSingularityStrengthRecalibration, self).build(input_shape)

    def call(self, x):
        alphas = self.alpha_layer(x)
        soft_counts = tf.reduce_sum(tf.nn.relu(self.soft_histogram(alphas)),
                                    axis=-1)  # TODO: not sure about this ReLU; but we have bn before!
        return x + soft_counts


class HistogramMultSingularityStrengthRecalibration(tf.keras.layers.Layer):
    def __init__(self, max_scale, k, per_channel, **kwargs):
        super(HistogramMultSingularityStrengthRecalibration, self).__init__(**kwargs)
        self.alpha_layer = WeightedLocalSingularityStrength(max_scale=max_scale)
        self.soft_histogram = SoftHistogramLayer(num_anchors=k, per_channel=per_channel)

    def build(self, input_shape):
        num_channels = input_shape[-1]
        #self.beta = self.add_weight(name='beta',
        #                            shape=(1),
        #                            initializer=tf.keras.initializers.Zeros(),
        #                            dtype=tf.float32,
        #                            trainable=True)
        super(HistogramMultSingularityStrengthRecalibration, self).build(input_shape)

    def call(self, x):
        alphas = self.alpha_layer(x)
        soft_counts = tf.reduce_sum(tf.nn.relu(self.soft_histogram(alphas)),
                                    axis=-1)  # TODO: not sure about this ReLU; but we have bn before!
        return x * tf.nn.sigmoid(soft_counts)


class SpatialWeightedSingularityStrengthRecalibration(tf.keras.layers.Layer):
    def __init__(self, r, max_scale, **kwargs):
        super(SpatialWeightedSingularityStrengthRecalibration, self).__init__(**kwargs)
        self.alpha_layer = WeightedLocalSingularityStrength(max_scale=max_scale)
        self.r = r

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        self.beta = self.add_weight(name='beta',
                                    shape=(1),
                                    initializer=tf.keras.initializers.Zeros(),
                                    dtype=tf.float32,
                                    trainable=True)
        super(SpatialWeightedSingularityStrengthRecalibration, self).build(input_shape)

    def call(self, x):
        alphas = self.alpha_layer(x)
        squeeze = self.gap(alphas)
        excite = self.w2(self.w1(squeeze))
        return (x * excite[:, tf.newaxis, tf.newaxis, :]) + (self.beta * (tf.nn.sigmoid(alphas)))


class SimpliefiedMSCAMSingularityStrengthRecalibration(tf.keras.layers.Layer):
    def __init__(self, r, max_scale, **kwargs):
        super(SimpliefiedMSCAMSingularityStrengthRecalibration, self).__init__(**kwargs)
        self.alpha_layer = LocalSingularityStrength(max_scale=max_scale)
        self.r = r

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        super(SimpliefiedMSCAMSingularityStrengthRecalibration, self).build(input_shape)

    def call(self, x):
        alphas = self.alpha_layer(x)
        squeeze = self.gap(alphas)
        excite = self.w2(self.w1(squeeze))
        return x * tf.nn.sigmoid(alphas + excite[:, tf.newaxis, tf.newaxis, :])  # TODO: Might need beta


class MSCAMSingularityStrengthRecalibration(tf.keras.layers.Layer):
    def __init__(self, r, max_scale, **kwargs):
        super(MSCAMSingularityStrengthRecalibration, self).__init__(**kwargs)
        self.alpha_layer = LocalSingularityStrength(max_scale=max_scale)
        self.r = r

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')

        self.c1 = tf.keras.layers.Conv2D(num_channels // self.r, kernel_size=1, activation='relu')
        self.c2 = tf.keras.layers.Conv2D(num_channels, kernel_size=1, activation='sigmoid')
        super(MSCAMSingularityStrengthRecalibration, self).build(input_shape)

    def call(self, x):
        alphas = self.alpha_layer(x)
        squeeze = self.gap(alphas)
        excite = self.w2(self.w1(squeeze))

        spatial_excite = self.c2(self.c1(alphas))
        return x * tf.nn.sigmoid(spatial_excite + excite[:, tf.newaxis, tf.newaxis, :])  # TODO: Might need beta


class AFFSingularityStrengthRecalibration(tf.keras.layers.Layer):
    def __init__(self, r, max_scale, **kwargs):
        super(AFFSingularityStrengthRecalibration, self).__init__(**kwargs)
        self.alpha_layer = LocalSingularityStrength(max_scale=max_scale)
        self.r = r

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')

        self.c1 = tf.keras.layers.Conv2D(num_channels // self.r, kernel_size=1, activation='relu')
        self.c2 = tf.keras.layers.Conv2D(num_channels, kernel_size=1, activation='sigmoid')
        super(AFFSingularityStrengthRecalibration, self).build(input_shape)

    def call(self, x):
        alphas = self.alpha_layer(x)

        fusion = x + alphas

        squeeze = self.gap(fusion)
        excite = self.w2(self.w1(squeeze))

        spatial_excite = self.c2(self.c1(fusion))
        z = fusion * tf.nn.sigmoid(spatial_excite + excite[:, tf.newaxis, tf.newaxis, :])
        return (z * x) + ((1 - z) * alphas)


class _SpatialSingularityRecalibration(tf.keras.layers.Layer):
    def __init__(self, r, max_scale, **kwargs):
        super(_SpatialSingularityRecalibration, self).__init__(**kwargs)
        self.max_scale = max_scale
        self.r = r
        self.scales = [2 ** i for i in range(1, max_scale + 1)]
        self.bn = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        self.conv_list = []
        for r in self.scales:
            self.conv_list.append(tf.keras.layers.DepthwiseConv2D(kernel_size=(r, r),
                                                                  depth_multiplier=1,
                                                                  trainable=False,
                                                                  activation=None,
                                                                  padding="SAME",
                                                                  depthwise_initializer=tf.keras.initializers.Ones()))
        self.scales = tf.cast(self.scales, dtype=tf.float32)

        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        super(_SpatialSingularityRecalibration, self).build(input_shape)

    def call(self, x, training=False):
        # TODO: debug sample_standard
        # x = sample_standard(x)  # this step can be removed if one ensures that x is non-negative
        x = sample_min_max_scaling(x)
        measures = tf.stack([conv(x) for conv in self.conv_list], axis=-1)
        alphas = OrdinaryLeastSquares(max_scale=self.max_scale)(measures)
        squeeze = self.gap(alphas)
        excite = self.w2(self.w1(squeeze))
        log_measures = tf.math.log(measures + tf.keras.backend.epsilon())
        log_r = tf.math.log(self.scales)
        ols_estimate = tf.tensordot(alphas, log_r, axes=0)
        sim = standardized_general_cosine_similarity(log_measures, ols_estimate)
        alpha_norm = self.bn(alphas, training=training)
        x = x * excite[:, tf.newaxis, tf.newaxis, :]
        return (sim * tf.nn.sigmoid(alpha_norm)) + ((1 - sim) * x)


class WeightedSpatialSingularityRecalibration(tf.keras.layers.Layer):
    def __init__(self, r, max_scale, **kwargs):
        super(WeightedSpatialSingularityRecalibration, self).__init__(**kwargs)
        self.max_scale = max_scale
        self.r = r
        self.scales = [2 ** i for i in range(1, max_scale + 1)]
        self.bn = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        self.conv_list = []
        for r in self.scales:
            self.conv_list.append(tf.keras.layers.DepthwiseConv2D(kernel_size=(r, r),
                                                                  depth_multiplier=1,
                                                                  trainable=False,
                                                                  activation=None,
                                                                  padding="SAME",
                                                                  depthwise_initializer=tf.keras.initializers.Ones()))
        self.scales = tf.cast(self.scales, dtype=tf.float32)
        self.k = self.add_weight(name='kappa',
                                 shape=[len(self.scales)],
                                 initializer=tf.keras.initializers.GlorotUniform(),
                                 dtype=tf.float32,
                                 trainable=True)
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        super(WeightedSpatialSingularityRecalibration, self).build(input_shape)

    def call(self, x, training=False):
        # TODO: debug sample_standard
        # x = sample_standard(x)  # this step can be removed if one ensures that x is non-negative
        x = sample_min_max_scaling(x)
        measures = tf.stack([conv(x) for conv in self.conv_list], axis=-1)
        alphas = OrdinaryLeastSquares(max_scale=self.max_scale)(measures)
        squeeze = self.gap(alphas)
        excite = self.w2(self.w1(squeeze))
        log_measures = tf.math.log(measures + tf.keras.backend.epsilon())
        log_r = tf.math.log(self.scales)
        k_ = tf.nn.sigmoid(self.k)
        log_k = tf.nn.sigmoid(tf.math.log(k_ + tf.keras.backend.epsilon()))
        # weighted_ols_estimate = tf.tensordot(alphas, log_k, axes=0) - log_k + tf.tensordot(alphas, log_r, axes=0)
        weighted_ols_estimate = tf.tensordot(alphas, log_k, axes=0) + tf.tensordot(alphas, log_r, axes=0)
        # weighted_ols_estimate = log_k + tf.tensordot(alphas, log_r, axes=0)
        sim = standardized_general_cosine_similarity(log_measures, weighted_ols_estimate)

        alpha_norm = self.bn(alphas, training=training)
        x = x * excite[:, tf.newaxis, tf.newaxis, :]
        return (sim * tf.nn.sigmoid(alpha_norm)) + ((1 - sim) * x)


class SSRSpatial(tf.keras.layers.Layer):
    def __init__(self, r, max_scale, **kwargs):
        super(SSRSpatial, self).__init__(**kwargs)
        self.alpha_layer = LocalSingularityStrength(max_scale=max_scale)
        self.r = r

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        self.beta = self.add_weight(name='beta',
                                    shape=(1),
                                    initializer=tf.keras.initializers.Zeros(),
                                    dtype=tf.float32,
                                    trainable=True)
        super(SSRSpatial, self).build(input_shape)

    def call(self, x):
        alphas = self.alpha_layer(x)
        squeeze = self.gap(alphas)
        excite = self.w2(self.w1(squeeze))
        return (x * excite[:, tf.newaxis, tf.newaxis, :]) + (self.beta * (tf.nn.sigmoid(alphas)))


class SqueezeExcite(tf.keras.layers.Layer):
    def __init__(self, r, **kwargs):
        super(SqueezeExcite, self).__init__(**kwargs)
        self.r = r

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        super(SqueezeExcite, self).build(input_shape)

    def call(self, x):
        squeeze = self.gap(x)
        excite = self.w2(self.w1(squeeze))
        return x * excite[:, tf.newaxis, tf.newaxis, :]

    def get_config(self):
        config = super(SqueezeExcite, self).get_config()
        config.update({
            "r": self.r
        })
        return config


class SpatialSqueezeExcite(tf.keras.layers.Layer):
    def __init__(self, r, **kwargs):
        super(SpatialSqueezeExcite, self).__init__(**kwargs)
        self.r = r

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        self.conv = tf.keras.layers.Conv2D(filters=1,
                                           kernel_size=(1, 1),
                                           activation='sigmoid')
        super(SpatialSqueezeExcite, self).build(input_shape)

    def call(self, x):
        squeeze = self.gap(x)
        channel_excite = x * self.w2(self.w1(squeeze))[:, tf.newaxis, tf.newaxis, :]
        spatial_excite = x * self.conv(x)
        return tf.reduce_max(tf.stack([channel_excite, spatial_excite], axis=-1), axis=-1)

    def get_config(self):
        config = super(SpatialSqueezeExcite, self).get_config()
        config.update({
            "r": self.r
        })
        return config


class StyleBasedRecalibration(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(StyleBasedRecalibration, self).__init__(**kwargs)
        self.bn = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w = tf.keras.layers.Dense(1, use_bias=False,
                                       activation=None)
        super(StyleBasedRecalibration, self).build(input_shape)

    def call(self, x, training=False):
        avg = self.gap(x)
        std = tf.math.sqrt(tf.math.reduce_variance(x, axis=[1, 2]) + 1e-12)
        t = tf.keras.layers.Concatenate(axis=-1)([avg[..., tf.newaxis], std[..., tf.newaxis]])
        z = tf.squeeze(self.w(t), axis=-1)
        return x * tf.nn.sigmoid(self.bn(z, training=training))[:, tf.newaxis, tf.newaxis, :]

    def get_config(self):
        config = super(StyleBasedRecalibration, self).get_config()
        return config

def fca_dct(spatial_resolution=224, channel_resolution=32, low_freq=None):
    x = tf.range(spatial_resolution)
    y = tf.range(channel_resolution)
    h, w = tf.meshgrid(x, y, indexing='ij')
    basis_support_h = tf.cast(tf.stack([h, w], axis=-1), dtype=tf.float32)
    pi = tf.cast(np.pi, dtype=tf.float32)
    left_cos = tf.math.cos((pi * basis_support_h[..., 0] / spatial_resolution) * (basis_support_h[..., 1] + 0.5))
    right_cos = tf.math.cos((pi * basis_support_h[..., 0] / spatial_resolution) * (basis_support_h[..., 1] + 0.5))
    dct_basis = np.zeros([spatial_resolution, spatial_resolution, channel_resolution, channel_resolution])
    for i in range(spatial_resolution):
        for j in range(spatial_resolution):
            dct_basis[i, j] = tf.linalg.matmul(left_cos[i, ...][..., tf.newaxis],
                                               right_cos[j, ...][tf.newaxis, ...])
    if low_freq is not None:
        dct_basis = dct_basis[:, :, :low_freq, :low_freq]
    return tf.cast(dct_basis[np.newaxis, ...], dtype=tf.float32)
