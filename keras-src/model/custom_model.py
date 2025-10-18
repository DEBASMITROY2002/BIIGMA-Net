import tensorflow as tf
from tensorflow import keras

@tf.keras.utils.register_keras_serializable(package="MyModel")
class MyModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_trackers()

    @property
    def metrics(self):
        return [
            ]

    def compile(self, loss_obj_dict, optimizer_obj):
        super().compile()
        self.loss_obj_dict = loss_obj_dict
        self.optimizer = optimizer_obj

    def setup_trackers(self):
        pass

    @tf.function
    def train_step(self, batch: tf.Tensor) -> tf.Tensor:
        x,y_true = batch

        with tf.GradientTape() as tape:
          tape.watch(x)
          side_outputs, single_spatial_map, [final_side_output,
                                          augmented_final_side_output
                                          ], [final_channel_activated_side_output_single_spatial_map,
                                              augmented_final_channel_activated_side_output_single_spatial_map
                                              ], [final_channel_activated_side_output,
                                                  augmented_final_channel_activated_side_output
                                                  ], probs = self(x, training=True)
          final_classification_result, augmented_final_classification_result , final_classification_result_soft, augmented_final_classification_result_soft = probs

          weights = self.trainable_variables

          kernel_loss ,  weighted_xentropy1, weighted_xentropy2, map_dist, kld_val = self.loss_obj_dict['HybridLoss'](y_true=y_true,
                                                 y_pred1=final_classification_result,
                                                 y_pred2=augmented_final_classification_result,
                                                 feature_blocks=final_channel_activated_side_output,
                                                 weights=weights,
                                                 map1=single_spatial_map,
                                                 map2=final_channel_activated_side_output_single_spatial_map,
                                                 softlogit1=final_classification_result_soft,
                                                 softlogit2=augmented_final_classification_result_soft)

          loss = self.loss_obj_dict['HybridLoss'].kernel_loss_factor * kernel_loss
          loss += self.loss_obj_dict['HybridLoss'].weighted_xentropy1_factor * weighted_xentropy1
          loss += self.loss_obj_dict['HybridLoss'].weighted_xentropy2_factor * weighted_xentropy2
          loss += self.loss_obj_dict['HybridLoss'].map_dist_factor * map_dist
          loss += self.loss_obj_dict['HybridLoss'].kld_factor * kld_val

        grad = tape.gradient(loss, weights)

        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        return {
            "train_loss_metric": loss,
            "train_kernel_loss_metric": kernel_loss,
            "train_weighted_xentropy1_metric": weighted_xentropy1,
            "train_weighted_xentropy2_metric": weighted_xentropy2,
            "train_map_dist_metric": map_dist,
            "train_kld_metric": kld_val,
            "y_pred": final_classification_result,
            "y_true": y_true
        }

    def test_step(self, batch: tf.Tensor) -> tf.Tensor:
        x,y_true = batch
        side_outputs, single_spatial_map, [final_side_output,
                                          augmented_final_side_output
                                          ], [final_channel_activated_side_output_single_spatial_map,
                                              augmented_final_channel_activated_side_output_single_spatial_map
                                              ], [final_channel_activated_side_output,
                                                  augmented_final_channel_activated_side_output
                                                  ], probs = self(x, training=False)

        final_classification_result, augmented_final_classification_result , final_classification_result_soft, augmented_final_classification_result_soft = probs

        kernel_loss ,  weighted_xentropy1,weighted_xentropy2, map_dist, kld_val = self.loss_obj_dict['HybridLoss'](y_true=y_true,
                                                 y_pred1 = final_classification_result,
                                                 y_pred2 = augmented_final_classification_result,
                                                 feature_blocks=final_channel_activated_side_output,
                                                 weights = self.trainable_variables,
                                                 map1=single_spatial_map,
                                                 map2=final_channel_activated_side_output_single_spatial_map,
                                                 softlogit1=final_classification_result_soft,
                                                 softlogit2=augmented_final_classification_result_soft)

        loss = self.loss_obj_dict['HybridLoss'].kernel_loss_factor * kernel_loss
        loss += self.loss_obj_dict['HybridLoss'].weighted_xentropy1_factor * weighted_xentropy1
        loss += self.loss_obj_dict['HybridLoss'].weighted_xentropy2_factor * weighted_xentropy2
        loss += self.loss_obj_dict['HybridLoss'].map_dist_factor * map_dist
        loss += self.loss_obj_dict['HybridLoss'].kld_factor * kld_val

        # val_ is prepended internally
        return {
            "val_loss_metric": loss,
            "val_kernel_loss_metric": kernel_loss,
            "val_weighted_xentropy1_metric": weighted_xentropy1,
            "val_weighted_xentropy2_metric": weighted_xentropy2,
            "val_map_dist_metric": map_dist,
            "val_kld_metric": kld_val,
            "y_pred": final_classification_result,
            "y_true": y_true
        }
