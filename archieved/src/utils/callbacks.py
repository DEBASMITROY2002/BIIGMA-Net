import tensorflow as tf
from tqdm import tqdm
from .metrics import MulticlassMetrics, MeanTracker
from ..dataset import NUM_LAB, TOTAL_TRAIN_SAMPLES, TOTAL_VAL_SAMPLES, BATCH_SIZE

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
    def __init__(self,best_model_crit,path,val_crit_score):
        super(MetricsCallback, self).__init__()
        self.train_metrics_tracker = MulticlassMetrics(NUM_LAB, average='macro')
        self.val_metrics_tracker = MulticlassMetrics(NUM_LAB, average='macro')
        self.train_mean_tracker = MeanTracker()
        self.val_mean_tracker = MeanTracker()
        self.t_steps_per_epoch = TOTAL_TRAIN_SAMPLES // BATCH_SIZE + int((TOTAL_TRAIN_SAMPLES % BATCH_SIZE)!=0)
        self.v_steps_per_epoch = TOTAL_VAL_SAMPLES // BATCH_SIZE + int((TOTAL_VAL_SAMPLES % BATCH_SIZE)!=0)
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
        # self.pbar.refresh()
        # self.pbar.reset()
