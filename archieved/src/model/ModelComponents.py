import tensorflow as tf


# Squeeze Excitation Block
@tf.keras.utils.register_keras_serializable(package="SqueezeExcitationLayer")
class SqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, channels, reduction_ratio=16, **kwargs):
        assert "name" in kwargs
        super(SqueezeExcitation, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.name = kwargs["name"]

    def build(self, input_shape):
        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D(name = self.name + "_global_pooling")
        self.squeeze_conv = tf.keras.layers.Conv2D(
            filters=input_shape[-1] // self.reduction_ratio,
            kernel_size=(1, 1),
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=False,
            name = self.name + "_squeeze_conv"
        )
        self.excitation_conv = tf.keras.layers.Conv2D(
            filters=self.channels,
            kernel_size=(1, 1),
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=False,
            name = self.name + "_excitation_conv"
        )
        super(SqueezeExcitation, self).build(input_shape)

    def call(self, inputs):
        x = self.global_pooling(inputs)
        x = tf.keras.layers.Reshape((1, 1, self.channels))(x)
        x = self.squeeze_conv(x)
        x = self.excitation_conv(x)
        return inputs * x

    def compute_output_shape(self, input_shape):
        return input_shape 
    
# Feature Projection Layer
@tf.keras.utils.register_keras_serializable(package="FeatureProjectionLayer")
class FeatureProjectionLayer(tf.keras.layers.Layer):
    def __init__(self, projection_dim, **kwargs):
        assert "name" in kwargs
        super(FeatureProjectionLayer, self).__init__(**kwargs)
        self.projection_dim = projection_dim
        self.name = kwargs["name"]

    def build(self, input_shape):
        self.point_conv = tf.keras.layers.Conv2D(
            filters=self.projection_dim,
            kernel_size=(1, 1),
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=False,
            name = self.name + "_point_conv"
        )

        self.se = SqueezeExcitation(channels=self.projection_dim, reduction_ratio=16, name = self.name + "_se")
        super(FeatureProjectionLayer, self).build(input_shape)

    def call(self, inputs):
        x = self.point_conv(inputs)
        x = self.se(x)
        return x 
    
# Channel Attention Layer
@tf.keras.utils.register_keras_serializable(package="ChannelAttentionLayer")
class ChannelAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, projection_dim, num_heads, **kwargs):
        assert "name" in kwargs
        super(ChannelAttentionLayer, self).__init__(**kwargs)
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.name = kwargs["name"]

    def build(self, input_shape):
        self.projectors = [
            FeatureProjectionLayer(projection_dim=self.projection_dim, name = f"{self.name}_projector_{i}")
            for i in range(self.num_heads)
        ]

    def call(self, inputs):
      projection_outputs = [projector(inputs) for projector in self.projectors]
      return projection_outputs

# Binary Thresholding
@tf.keras.utils.register_keras_serializable(package="BinaryThresholding")
class BinaryThresholding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        assert "name" in kwargs
        super(BinaryThresholding, self).__init__(**kwargs)
        self.name = kwargs["name"]

    def call(self, input):
      return tf.cast(input > tf.reduce_mean(input, axis=[1,2], keepdims=True), dtype=tf.float32)

# Spatial Attention Adapter (Purpose is to normalize sideoutputs from different exits)
@tf.keras.utils.register_keras_serializable(package="SpatAttentionAdapter")
class SpatAttentionAdapter(tf.keras.layers.Layer):
    def __init__(self, down_sample_order,**kwargs):
        assert "name" in kwargs
        super(SpatAttentionAdapter, self).__init__(**kwargs)
        self.name = kwargs["name"]
        self.down_sample_order = down_sample_order

    def build(self, input_shape):
        self.se = SqueezeExcitation(channels=input_shape[-1], reduction_ratio=16, name = self.name + "_se")
        # self.point_conv = tf.keras.layers.Conv2D(
        #     filters=1,
        #     kernel_size=(1, 1),
        #     activation='relu',
        #     kernel_initializer='he_normal',
        #     use_bias=False,
        #     name = self.name + "_point_conv"
        # )
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(self.down_sample_order, self.down_sample_order), name = self.name + "_pool")

    def call(self, inputs):
        x = inputs
        # x = self.se(x)
        # x = self.point_conv(x)
        x = self.pool(x)
        # # min max normalize
        # x = (x-tf.reduce_min(x, axis=[1,2], keepdims=True))/(tf.reduce_max(x, axis=[1,2], keepdims=True)-tf.reduce_min(x, axis=[1,2], keepdims=True)+1e-9)
        # # Sigmoid
        # x = tf.keras.activations.sigmoid(2*x)
        # x = tf.cast(x > tf.reduce_mean(x, axis=[1,2], keepdims=True), dtype=tf.float32)
        return x

# Spatial Attention Generator( Receives List of features with same spatial dimention and generate single channel map)
@tf.keras.utils.register_keras_serializable(package="SpatAttentionGen")
class SpatAttentionGen(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        assert "name" in kwargs
        super(SpatAttentionGen, self).__init__(**kwargs)
        self.name = kwargs["name"]

    def build(self, input_shape):
      self.conv = tf.keras.layers.Conv2D(
          filters=1,
          kernel_size=3,
          activation='sigmoid',
          padding='same',
          name = self.name + "_conv"
      )
      self.binary_thresholding = BinaryThresholding(name = self.name + "_binary_thresholding")

    def call(self, inputs):
      concat_result = tf.keras.layers.Concatenate(axis=-1, name = self.name + "_concat0")(inputs)  # (BatchSize x 7 x 7 x C)
      avg_pool = tf.reduce_mean(concat_result, axis=-1, keepdims=True, name = self.name + "_avg_pool")  # (BatchSize x 7 x 7 x 1)
      # max_pool = tf.reduce_max(concat_result, axis=-1, keepdims=True, name = self.name + "_max_pool")  # (BatchSize x 7 x 7 x 1)
      # pooled_features = tf.keras.layers.Concatenate(axis=-1, name = self.name + "_concat1")([avg_pool, max_pool])  # (BatchSize x 7 x 7 x 2)
      # spatial_attention_map = self.conv(pooled_features)  # (BatchSize x 7 x 7 x 1)
      # return spatial_attention_map
      activated_pool = tf.keras.activations.sigmoid(avg_pool)
      return activated_pool
      # binarized_mask = self.binary_thresholding(activated_pool)
      # return binarized_mask

# Channel Pruning Layer
@tf.keras.utils.register_keras_serializable(package="ChannelPruningLayer")
class ChannelPruningLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        assert "name" in kwargs
        super(ChannelPruningLayer, self).__init__(**kwargs)
        self.name = kwargs["name"]

    def call(self, input):
      channel_wise_variance = tf.math.reduce_variance(input, axis=[1,2], keepdims=True)
      mean_var = tf.reduce_mean(channel_wise_variance)
      mask = tf.cast(channel_wise_variance > mean_var, dtype=tf.float32)
      relevant_features = input * mask
      map = tf.reduce_sum(relevant_features, axis=-1, keepdims=True) / tf.reduce_sum(mask, axis=-1, keepdims=True)
      return map


# Function Implementaion
def channel_pruning_layer(input):
  channel_wise_variance = tf.math.reduce_variance(input, axis=[1,2], keepdims=True)
  mean_var = tf.reduce_mean(channel_wise_variance)
  mask = tf.cast(channel_wise_variance > mean_var, dtype=tf.float32)
  relevant_features = input * mask
  map = tf.reduce_sum(relevant_features, axis=-1, keepdims=True) / tf.reduce_sum(mask, axis=-1, keepdims=True)
  return map

# Vector Sampling Layer
@tf.keras.utils.register_keras_serializable(package="VectorSamplingLayer")
class VectorSamplingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        assert "name" in kwargs
        super(VectorSamplingLayer, self).__init__(**kwargs)
        self.name = kwargs["name"]

    def call(self, input):
      """
      feat : (batch_size, height, width, channels)
      mask : (batch_size, height, width, 1)
      # noise : (batch_size, batch_size, height, width, 1)
      """
      feat , mask = input
      fg = mask * feat # (batch_size, height, width, channels)
      bg = (1-mask)*feat # (batch_size, height, width, channels)

      bg_flat = tf.reshape(bg, (-1, bg.shape[-1])) # (batch_size*height*width, channels)
      bg_flat_sampled = tf.keras.random.shuffle(bg_flat, axis=0) # (batch_size*height*width, channels)
      sampled_bg = tf.reshape(bg_flat_sampled, (-1, bg.shape[1], bg.shape[2], bg.shape[3])) # (batch_size, height, width, channels)
      sampled_bg_masked = (1-mask)*sampled_bg # (batch_size, height, width, channels)

      augmented_feat = fg + sampled_bg_masked # (batch_size, height, width, channels)
      return augmented_feat

# Classification Head
@tf.keras.utils.register_keras_serializable(package="ClassificationHead")
class ClassificationHead(tf.keras.layers.Layer):
  def __init__(self, num_classes, temp=1.0, return_prob=True, **kwargs):
    assert "name" in kwargs
    super(ClassificationHead, self).__init__(**kwargs)
    self.num_classes = num_classes
    self.temp = temp
    self.return_prob = return_prob
    self.name = kwargs["name"]

  def build(self, input_shape):
    self.global_pooling = tf.keras.layers.GlobalAveragePooling2D(name = self.name + "_global_pooling")
    self.dense = tf.keras.layers.Dense(
        units=self.num_classes,
        activation=None,
        kernel_initializer='he_normal',
        name = self.name + "_dense"
    )
    self.drop_out = tf.keras.layers.Dropout(rate=0.5, name = self.name + "_drop_out")
    super(ClassificationHead, self).build(input_shape)

  def call(self, inputs):
    x = self.global_pooling(inputs)
    x = self.drop_out(x)
    x = self.dense(x)
    if self.return_prob:
      x = x / self.temp
      x = tf.keras.activations.softmax(x)
    return x 
  
# Post Feature Extraction
@tf.keras.utils.register_keras_serializable(package="PostFeatureExtraction")
class PostFeatureExtraction(tf.keras.layers.Layer):
    def __init__(self, num_classes, num_heads, final_side_output_layer_name, **kwargs):
        assert "name" in kwargs
        super(PostFeatureExtraction, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.final_side_output_layer_name = final_side_output_layer_name
        self.name = kwargs["name"]

    def build(self, input_shape):
        self.channel_attention_layer = ChannelAttentionLayer(
            name=f"CA_{self.final_side_output_layer_name}",
            projection_dim=input_shape[-1],
            num_heads=self.num_heads)

        self.channel_attention_concatenated_layer = tf.keras.layers.Concatenate(
            axis=-1,
            name=f"CA_concat_{self.final_side_output_layer_name}")

        self.classification_head = ClassificationHead(
            name=f"classification_head_{self.final_side_output_layer_name}",
            num_classes=self.num_classes,
            return_prob=False,
            temp=1.0)

        self.multiply_layer = tf.keras.layers.Multiply(name=f"CA_multiply_spatial_map_{self.final_side_output_layer_name}")

        super(PostFeatureExtraction, self).build(input_shape)

    def call(self, inputs):
      final_channel_activated_side_output = self.channel_attention_layer(inputs)
      final_channel_activated_concatenated_output = self.channel_attention_concatenated_layer(final_channel_activated_side_output)
      final_classification_result_logits = self.classification_head(final_channel_activated_concatenated_output)

      final_channel_activated_side_output_spatial_maps = [
          (tf.keras.activations.sigmoid(channel_pruning_layer(output)))
          for output in final_channel_activated_side_output
      ]
      final_channel_activated_side_output_single_spatial_map = self.multiply_layer(final_channel_activated_side_output_spatial_maps)

      return final_channel_activated_side_output_single_spatial_map, final_channel_activated_side_output, final_classification_result_logits 
    
