import tensorflow as tf
from tqdm import tqdm
from .metrics import MulticlassMetrics, MeanTracker
from config import get_config

config = get_config()
BATCH_SIZE = config['BATCH_SIZE']

class NaNCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(NaNCallback, self).__init__()
        self.nan_detected = False

    def _check_tensor_for_nans(self, tensor, tensor_name):
        """Helper function to check a tensor for NaN values."""
        if tensor is None:
            return False

        if isinstance(tensor, (list, tuple)):
            return any(self._check_tensor_for_nans(t, f"{tensor_name}[{i}]")
                      for i, t in enumerate(tensor))

        has_nans = tf.math.reduce_any(tf.math.is_nan(tensor))
        if has_nans:
            print(f"\nNaN detected in {tensor_name}")
            return True
        return False

    def _check_model_outputs(self, logs):
        """Check all model outputs for NaNs."""
        if not hasattr(self.model, 'last_outputs'):
            return False

        outputs = self.model.last_outputs
        output_names = [
            "side_outputs",
            "single_spatial_map",
            "final_side_output",
            "augmented_final_side_output",
            "final_channel_activated_side_output_single_spatial_map",
            "augmented_final_channel_activated_side_output_single_spatial_map",
            "final_channel_activated_side_output",
            "augmented_final_channel_activated_side_output",
            "final_classification_result",
            "augmented_final_classification_result",
            "final_classification_result_soft",
            "augmented_final_classification_result_soft"
        ]

        has_nans = False
        for name, output in zip(output_names, outputs):
            if self._check_tensor_for_nans(output, f"Model output: {name}"):
                has_nans = True
        return has_nans

    def _check_loss_components(self, logs):
        """Check all loss components for NaNs."""
        if logs is None:
            return False

        loss_components = [
            'loss_metric',
            'kernel_loss_metric',
            'weighted_xentropy1_metric',
            'weighted_xentropy2_metric',
            'map_dist_metric',
            'kld_metric'
        ]

        has_nans = False
        for base_name in loss_components:
            # Check both train and val metrics
            for prefix in ['train_', 'val_']:
                name = prefix + base_name
                if name in logs:
                    value = logs[name]
                    if tf.math.is_nan(value):
                        print(f"\nNaN detected in loss component: {name}")
                        has_nans = True
        return has_nans

    def _check_gradients(self):
        """Check gradients of all trainable variables for NaNs."""
        if not hasattr(self.model, 'trainable_variables'):
            return False

        has_nans = False
        for var in self.model.trainable_variables:
            grad = var.gradient if hasattr(var, 'gradient') else None
            if grad is not None and self._check_tensor_for_nans(grad, f"Gradient for {var.name}"):
                has_nans = True
        return has_nans

    def on_train_batch_end(self, batch, logs=None):
        """Check for NaNs after each training batch."""
        if self.nan_detected:
            return

        has_nans = False

        # Check model outputs
        if self._check_model_outputs(logs):
            has_nans = True
            print("NaN values detected in model outputs!")

        # Check loss components
        if self._check_loss_components(logs):
            has_nans = True
            print("NaN values detected in loss components!")

        # Check gradients and weights
        if self._check_gradients():
            has_nans = True
            print("NaN values detected in gradients!")

        if has_nans:
            self.nan_detected = True
            self.model.stop_training = True
            print("\nTraining stopped due to NaN values!")
            print(f"\nBatch number: {batch}")
            if logs:
                print("\nLoss values at failure:")
                for name, value in logs.items():
                    print(f"{name}: {value}")

class ModelSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, period, path):
        super(ModelSaveCallback, self).__init__()
        self.period = period
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            self.model.save(self.path)
            print(f"Model saved at epoch {epoch + 1}")

class SamplingLossActivationCallback(tf.keras.callbacks.Callback):
    def __init__(self, target_epoch):
        super(SamplingLossActivationCallback, self).__init__()
        self.target_epoch = target_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch > self.target_epoch:
          self.model.loss_obj_dict['HybridLoss'].weighted_xentropy2_factor = 0.5
          self.model.loss_obj_dict['HybridLoss'].kld_factor = 0.5
          print("\nweighted_xentropy2_factor changed to 0.5 , kld_factor changed to 0.5")

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self,best_model_crit,path,val_crit_score, num_classes, total_train_samples, total_val_samples):
        super(MetricsCallback, self).__init__()
        self.train_metrics_tracker = MulticlassMetrics(num_classes, average='macro')
        self.val_metrics_tracker = MulticlassMetrics(num_classes, average='macro')
        self.train_mean_tracker = MeanTracker()
        self.val_mean_tracker = MeanTracker()
        self.t_steps_per_epoch = total_train_samples // BATCH_SIZE + int((total_train_samples % BATCH_SIZE)!=0)
        self.v_steps_per_epoch = total_val_samples // BATCH_SIZE + int((total_val_samples % BATCH_SIZE)!=0)
        self.best_model_crit = best_model_crit
        self.val_crit_score = val_crit_score
        self.path = path
        self.reset_pbar()

    def reset_pbar(self):
      self.pbar = tqdm(
          total=self.t_steps_per_epoch,
          position=0,
          leave=True,
          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')

    def get_verbose_description(self, logs, mode='train'):
       verbose_text = ""

       if mode == 'train':
        for k,v in self.train_mean_tracker.result().items():
          verbose_text += f"{k}: {v:.4f} | "

        for k,v in self.train_metrics_tracker.result().items():
          verbose_text += f"{k}: {v:.4f} | "
       else:
        for k,v in self.val_mean_tracker.result().items():
          verbose_text += f"{k}: {v:.4f} | "

        for k,v in self.val_metrics_tracker.result().items():
          verbose_text += f"{k}: {v:.4f} | "

       return verbose_text

    def on_train_batch_end(self, batch, logs=None):
        y_pred = logs['y_pred']
        y_true = logs['y_true']
        self.train_metrics_tracker.update_state(y_pred, y_true)
        self.train_mean_tracker.update_state({
            k:v for k,v in logs.items() if "metric" in k
        })

        verbose = self.get_verbose_description(logs, mode='train')
        self.pbar.set_description(verbose)
        self.pbar.update()

    def on_test_batch_end(self, batch, logs=None):
        y_pred = logs['y_pred']
        y_true = logs['y_true']
        self.val_metrics_tracker.update_state(y_pred, y_true)
        self.val_mean_tracker.update_state({
            k:v for k,v in logs.items() if "metric" in k
        })

    def on_epoch_begin(self, epoch, logs=None):
        self.reset_pbar()
        self.train_metrics_tracker.reset_state()
        self.val_metrics_tracker.reset_state()
        self.train_mean_tracker.reset_state()
        self.val_mean_tracker.reset_state()
        print(f"\n[START OF RESULT]\nEpoch {epoch+1}")

    def on_epoch_end(self, epoch, logs=None):
        train_verbose = self.get_verbose_description(logs, mode='train')
        val_verbose = self.get_verbose_description(logs, mode='val')
        print("\n" + train_verbose + "\n" + val_verbose + "\n[END OF RESULT]")

        val_crit_score = self.val_metrics_tracker.result()[self.best_model_crit]
        if val_crit_score > self.val_crit_score:
          self.model.save(f"{self.path}")
          print(f"Model saved at epoch {epoch + 1} as val {self.best_model_crit} improved from {self.val_crit_score:.4f} to {val_crit_score:.4f}")
          self.val_crit_score = val_crit_score
