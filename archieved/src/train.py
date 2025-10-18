import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import argparse
from datetime import datetime

# Import custom loss functions
from Loss.HybridLoss import HybridLoss
from model.ModelComponents import *  # Assuming your model components are here


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

          kernel_loss ,  weighted_xentropy1, weighted_xentropy2, map_dist = self.loss_obj_dict['HybridLoss'](y_true=y_true,
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
          # loss += self.loss_obj_dict['HybridLoss'].kld_factor * kld_val

        grad = tape.gradient(loss, weights)

        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        return {
            "train_loss_metric": loss,
            "train_kernel_loss_metric": kernel_loss,
            "train_weighted_xentropy1_metric": weighted_xentropy1,
            "train_weighted_xentropy2_metric": weighted_xentropy2,
            "train_map_dist_metric": map_dist,
            # "train_kld_metric": kld_val,
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

        kernel_loss ,  weighted_xentropy1,weighted_xentropy2, map_dist = self.loss_obj_dict['HybridLoss'](y_true=y_true,
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
        # loss += self.loss_obj_dict['HybridLoss'].kld_factor * kld_val

        # val_ is prepended internally
        return {
            "val_loss_metric": loss,
            "val_kernel_loss_metric": kernel_loss,
            "val_weighted_xentropy1_metric": weighted_xentropy1,
            "val_weighted_xentropy2_metric": weighted_xentropy2,
            "val_map_dist_metric": map_dist,
            # "val_kld_metric": kld_val,
            "y_pred": final_classification_result,
            "y_true": y_true
        }


def create_model():
    """
    Create and return the model instance.
    You'll need to implement your model architecture here based on your ModelComponents.
    """
    # TODO: Implement your model architecture
    # This is a placeholder - replace with your actual model implementation
    model = MyModel()
    return model


def load_data(data_path, batch_size=32, image_size=(256, 256)):
    """
    Load and preprocess your dataset.
    
    Args:
        data_path: Path to your dataset
        batch_size: Batch size for training
        image_size: Input image size
    
    Returns:
        train_dataset, val_dataset
    """
    # TODO: Implement your data loading logic
    # This is a placeholder - replace with your actual data loading
    
    # Example for image datasets:
    # train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     os.path.join(data_path, 'train'),
    #     image_size=image_size,
    #     batch_size=batch_size,
    #     label_mode='categorical'  # or 'binary' depending on your task
    # )
    
    # val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     os.path.join(data_path, 'val'),
    #     image_size=image_size,
    #     batch_size=batch_size,
    #     label_mode='categorical'
    # )
    
    # For now, return None - you'll need to implement this
    return None, None


def setup_callbacks(checkpoint_dir, log_dir):
    """
    Setup training callbacks.
    
    Args:
        checkpoint_dir: Directory to save model checkpoints
        log_dir: Directory for TensorBoard logs
    
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'model_checkpoint_{epoch:02d}_{val_loss:.2f}.h5'),
        monitor='val_loss_metric',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss_metric',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Reduce learning rate on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss_metric',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    )
    callbacks.append(tensorboard_callback)
    
    return callbacks


def main():
    parser = argparse.ArgumentParser(description='Train BIIGMA-Net model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256], help='Input image size')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for TensorBoard logs')
    
    # Loss function hyperparameters
    parser.add_argument('--kernel_loss_factor', type=float, default=0.5, help='Kernel loss weight')
    parser.add_argument('--weighted_xentropy1_factor', type=float, default=0.5, help='Weighted cross-entropy 1 weight')
    parser.add_argument('--weighted_xentropy2_factor', type=float, default=0.5, help='Weighted cross-entropy 2 weight')
    parser.add_argument('--map_dist_factor', type=float, default=0.5, help='Map distance weight')
    parser.add_argument('--kld_factor', type=float, default=0.5, help='KLD weight')
    parser.add_argument('--sigma', type=float, default=1.0, help='Sigma parameter for kernel loss')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Add timestamp to log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    
    print("=== BIIGMA-Net Training Script ===")
    print(f"Data path: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Image size: {args.image_size}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Log directory: {log_dir}")
    
    # Set up GPU if available
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"GPU available: {physical_devices[0]}")
    else:
        print("No GPU found, using CPU")
    
    # Load data
    print("\nLoading data...")
    train_dataset, val_dataset = load_data(args.data_path, args.batch_size, tuple(args.image_size))
    
    if train_dataset is None or val_dataset is None:
        print("ERROR: Data loading not implemented. Please implement the load_data function.")
        return
    
    # Create model
    print("\nCreating model...")
    model = create_model()
    
    # Setup loss function
    hybrid_loss = HybridLoss(
        kernel_loss_factor=args.kernel_loss_factor,
        weighted_xentropy1_factor=args.weighted_xentropy1_factor,
        weighted_xentropy2_factor=args.weighted_xentropy2_factor,
        map_dist_factor=args.map_dist_factor,
        kld_factor=args.kld_factor,
        sigma=args.sigma
    )
    
    loss_obj_dict = {'HybridLoss': hybrid_loss}
    
    # Setup optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    # Compile model
    print("Compiling model...")
    model.compile(loss_obj_dict=loss_obj_dict, optimizer_obj=optimizer)
    
    # Setup callbacks
    callbacks = setup_callbacks(args.checkpoint_dir, log_dir)
    
    # Start training
    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(args.checkpoint_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    print("Training completed!")


if __name__ == "__main__":
    main()
