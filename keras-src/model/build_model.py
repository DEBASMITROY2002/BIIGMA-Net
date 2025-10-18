import tensorflow as tf
from .custom_model import MyModel
from .components import SpatAttentionAdapter, SpatAttentionGen, VectorSamplingLayer, PostFeatureExtraction

def build_model(input_shape, num_classes, num_heads, temperature, side_output_layer_names = ['pool3_relu','pool4_relu','relu']):

    pretrained_model = tf.keras.applications.DenseNet121(
      include_top=False ,
      weights='imagenet' ,
      input_shape=input_shape)

    inputs = pretrained_model.input

    side_outputs = [
        pretrained_model.get_layer(name).output
        for name in side_output_layer_names
    ]

    spatial_adapter_results = [
        SpatAttentionAdapter(down_sample_order=2**(len(side_output_layer_names)-i-1), name=f"spatial_adapter_{name}")(output)
        for i, (name, output) in enumerate(zip(side_output_layer_names, side_outputs))
    ]

    single_spatial_map = SpatAttentionGen(name="single_spatial_map")(spatial_adapter_results)


    final_side_output_original,final_side_output_layer_name = side_outputs[-1], side_output_layer_names[-1]
    final_side_output = final_side_output_original
    augmented_final_side_output = VectorSamplingLayer(name=f"vector_sampling_{final_side_output_layer_name}")([final_side_output_original, single_spatial_map])


    # Post Feature Extraction Layer
    post_feature_extraction_layer_object = PostFeatureExtraction(
        num_classes=num_classes,
        num_heads=num_heads,
        final_side_output_layer_name=final_side_output_layer_name,
        name="post_feature_extraction")

    # Post Feature Extraction
    final_channel_activated_side_output_single_spatial_map, final_channel_activated_side_output, final_classification_result_logits = post_feature_extraction_layer_object(final_side_output)

    # Post Feature Extraction with Augmented Feature
    augmented_final_channel_activated_side_output_single_spatial_map,  augmented_final_channel_activated_side_output, augmented_final_classification_result_logits = post_feature_extraction_layer_object(augmented_final_side_output)


    # Probabilties
    final_classification_result = tf.keras.activations.softmax(final_classification_result_logits)
    augmented_final_classification_result = tf.keras.activations.softmax(augmented_final_classification_result_logits)
    final_classification_result_soft = tf.keras.activations.softmax(final_classification_result_logits/temperature)
    augmented_final_classification_result_soft = tf.keras.activations.softmax(augmented_final_classification_result_logits/temperature)

    model = MyModel(inputs=inputs, outputs=[
        side_outputs,
        single_spatial_map,
        [
            final_side_output,
            augmented_final_side_output
        ],
        [
            final_channel_activated_side_output_single_spatial_map,
            augmented_final_channel_activated_side_output_single_spatial_map
        ],
        [
            final_channel_activated_side_output,
            augmented_final_channel_activated_side_output
        ],
        [
            final_classification_result,
            augmented_final_classification_result,
            final_classification_result_soft,
            augmented_final_classification_result_soft
        ]
    ])

    return model
