import tensorflow as tf

from neural_components.convolutional import ContractingLayer, UpsampleExpandingLayer
from neural_components.custom_ops import sample_min_max_scaling


class FastAlphaLayer(tf.keras.layers.Layer):
    def __init__(self, r1, r2, **kwargs):
        super(FastAlphaLayer, self).__init__(**kwargs)
        assert (r1 < r2) and (r2 % 2 == 0) and (r1 % 2 == 0)
        self.r1 = r1
        self.r2 = r2
        self.bn = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        self.r1_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(self.r1, self.r1),
                                                       depth_multiplier=1,
                                                       trainable=False,
                                                       activation=None,
                                                       padding="SAME",
                                                       depthwise_initializer=tf.keras.initializers.Ones())
        self.r2_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(self.r2, self.r2),
                                                       depth_multiplier=1,
                                                       trainable=False,
                                                       activation=None,
                                                       padding="SAME",
                                                       depthwise_initializer=tf.keras.initializers.Ones())
        self.r1 = tf.cast(self.r1, dtype=tf.float32)
        self.r2 = tf.cast(self.r2, dtype=tf.float32)
        super(FastAlphaLayer, self).build(input_shape)

    def call(self, x, training=False):
        x = sample_min_max_scaling(x)
        measures = tf.stack([self.r1_conv(x), self.r2_conv(x)], axis=-1)
        alphas = (tf.math.log(measures[..., -1]) - tf.math.log(measures[..., 0])) / (
                tf.math.log(self.r2) - tf.math.log(self.r1))
        return self.bn(alphas, training=training)


class LeastSquaresFittingLayer_(tf.keras.layers.Layer):
    def __init__(self, max_scale, **kwargs):
        super(LeastSquaresFittingLayer_, self).__init__(**kwargs)
        self.max_scale = max_scale

    def call(self, x):
        scales = 2 ** tf.range(1, self.max_scale + 1,
                               dtype=tf.float32)  # Inquiry if adding EPSILON here makes sense
        log_scales = tf.math.log(scales)
        log_measures = tf.math.log(x + tf.keras.backend.epsilon())
        mean_log_scales = tf.reduce_mean(log_scales)
        mean_log_measures = tf.reduce_mean(log_measures, axis=-1)[..., tf.newaxis]  # make it broadcastable
        numerator = (log_measures - mean_log_measures) * (log_scales - mean_log_scales)
        denominator = (log_scales - mean_log_scales) ** 2
        return tf.reduce_sum(numerator, axis=-1) / tf.reduce_sum(denominator, axis=-1)


class FastAlphaOLSLayer(tf.keras.layers.Layer):
    def __init__(self, max_scale, **kwargs):
        super(FastAlphaOLSLayer, self).__init__(**kwargs)
        self.max_scale = max_scale
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
        super(FastAlphaOLSLayer, self).build(input_shape)

    def call(self, x, training=False):
        x = sample_min_max_scaling(x)
        measures = tf.stack([conv(x) for conv in self.conv_list], axis=-1)
        alphas = LeastSquaresFittingLayer_(max_scale=self.max_scale)(measures)
        return self.bn(alphas, training=training)


class SpatialFastAlphaOLSSqueezeExcite(tf.keras.layers.Layer):
    def __init__(self, r, max_scale, **kwargs):
        super(SpatialFastAlphaOLSSqueezeExcite, self).__init__(**kwargs)
        self.alpha_layer = FastAlphaOLSLayer(max_scale=max_scale)
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
        super(SpatialFastAlphaOLSSqueezeExcite, self).build(input_shape)

    def call(self, x):
        alphas = self.alpha_layer(x)
        squeeze = self.gap(alphas)
        excite = self.w2(self.w1(squeeze))
        #return (x * excite[:, tf.newaxis, tf.newaxis, :]) + (self.beta * (tf.nn.sigmoid(alphas)))
        return x + (self.beta * (tf.nn.sigmoid(alphas)))

def get_spatial_alpha_ols_encoders_unet(channels_per_level, r=2, r1=2, r2=4, input_shape=[224, 224, 3], n_classes=2,
                                        with_bn=True):
    input = tf.keras.layers.Input(shape=input_shape)
    n_levels = len(channels_per_level) - 1  # hierarchy levels in U-Net
    skips = []  # keep record of the output of each CNN contracting block
    output_channel_dim = 1 if n_classes <= 2 else n_classes
    final_activation_fn = 'sigmoid' if output_channel_dim == 1 else 'softmax'
    x = input  # We need to keep the input variable to give it as input to the Model constructor later in the function
    # Encoder
    for n_channel in channels_per_level[:-1]:
        x = ContractingLayer(n_filters=n_channel, kernel_size=3, padding='same', with_bn=with_bn)(x)
        x = SpatialFastAlphaOLSSqueezeExcite(r=2, max_scale=3)(x)
        skips.append(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    # Bottleneck
    x = ContractingLayer(n_filters=channels_per_level[-1], kernel_size=3, padding='same', with_bn=with_bn)(x)
    # Decoder
    for i, skip in enumerate(skips[::-1]):
        x = UpsampleExpandingLayer(n_filters=channels_per_level[n_levels - i - 1], kernel_size=3, padding='same',
                                   with_bn=with_bn)(x, skip)
        # x = AlphaSqueezeExcite(r=r, local_scale=6)(x)
    # 1x1 for dense classification
    output = tf.keras.layers.Conv2D(output_channel_dim, kernel_size=1, activation=final_activation_fn)(x)
    return tf.keras.models.Model(inputs=[input], outputs=[output])
