import tensorflow as tf
from collections import defaultdict

class MulticlassMetrics:
    def __init__(self, num_classes, average='macro'):
        self.num_classes = num_classes
        self.average = average
        self.reset_state()

    def reset_state(self):
        try:
            del self.predictions
            del self.ground_truths
        except:
            pass
        self.predictions = []
        self.ground_truths = []

    def update_state(self, predictions, ground_truths):
        if len(predictions.shape) == 2 and predictions.shape[1] == self.num_classes:
            predictions = tf.argmax(predictions, axis=1)
        if len(ground_truths.shape) == 2 and ground_truths.shape[1] == self.num_classes:
            ground_truths = tf.argmax(ground_truths, axis=1)
        self.predictions.append(predictions.numpy())
        self.ground_truths.append(ground_truths.numpy())

    def result(self):
        predictions = tf.concat(self.predictions, axis=0)
        ground_truths = tf.concat(self.ground_truths, axis=0)
        acc = tf.reduce_mean(tf.cast(tf.equal(predictions, ground_truths), tf.float32))

        cm = tf.math.confusion_matrix(ground_truths, predictions, num_classes=self.num_classes)
        cm = tf.cast(cm, tf.float32)
        tp = tf.linalg.diag_part(cm)
        fp = tf.reduce_sum(cm, axis=0) - tp
        fn = tf.reduce_sum(cm, axis=1) - tp
        tn = tf.reduce_sum(cm) - (tp + fp + fn)

        precision_per_class = tp / (tp + fp + 1e-10)
        recall_per_class = tp / (tp + fn + 1e-10)
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + 1e-10)
        specificity_per_class = tn / (tn + fp + 1e-10)

        if self.average == 'micro':
            precision = tf.reduce_sum(tp) / (tf.reduce_sum(tp) + tf.reduce_sum(fp) + 1e-10)
            recall = tf.reduce_sum(tp) / (tf.reduce_sum(tp) + tf.reduce_sum(fn) + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            specificity = tf.reduce_sum(tn) / (tf.reduce_sum(tn) + tf.reduce_sum(fp) + 1e-10)
        elif self.average == 'macro':
            precision = tf.reduce_mean(precision_per_class)
            recall = tf.reduce_mean(recall_per_class)
            f1 = tf.reduce_mean(f1_per_class)
            specificity = tf.reduce_mean(specificity_per_class)
        elif self.average == 'weighted':
            weights = tf.reduce_sum(cm, axis=1) / tf.reduce_sum(cm)
            precision = tf.reduce_sum(precision_per_class * weights)
            recall = tf.reduce_sum(recall_per_class * weights)
            f1 = tf.reduce_sum(f1_per_class * weights)
            specificity = tf.reduce_sum(specificity_per_class * weights)
        else:
            raise ValueError("Invalid average type. Choose from ['micro', 'macro', 'weighted', 'NUM_CLASS'].")

        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity
        }
    

class MeanTracker:
  def __init__(self):
    self.metric_results = defaultdict(list)

  def update_state(self, metrics):
    for metric_name, metric_value in metrics.items():
      self.metric_results[metric_name].append(metric_value)

  def result(self):
    return {metric_name: tf.reduce_mean(metric_values) for metric_name, metric_values in self.metric_results.items()}

  def reset_state(self):
    del self.metric_results
    self.metric_results = defaultdict(list)