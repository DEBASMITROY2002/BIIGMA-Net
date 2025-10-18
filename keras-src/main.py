import os
import tensorflow as tf
from config import set_default_config, get_config
from .data.dataset import get_data_generators
from .model.build_model import build_model
from .loss.hybrid_loss import HybridLoss
from .utils.callbacks import MetricsCallback, SamplingLossActivationCallback

def main():
    set_default_config()
    config = get_config()

    os.environ["TF_GPU_THREAD_MODE"] = config['TF_GPU_THREAD_MODE']
    tf.config.optimizer.set_jit(True)

    train_generator, test_generator = get_data_generators()

    TOTAL_TRAIN_SAMPLES = train_generator.n
    TOTAL_VAL_SAMPLES = test_generator.n
    NUM_LAB = len(train_generator.class_indices.keys())

    my_model = build_model(
        input_shape=(config['HEIGHT'], config['WIDTH'], 3),
        num_classes=NUM_LAB,
        num_heads=config['NUM_HEADS'],
        temperature=config['TEMPERATURE'],
        side_output_layer_names=config['SIDE_OUTPUT_LAYER_NAMES']
    )

    my_model.compile(
        loss_obj_dict={
            "HybridLoss": HybridLoss(
                kernel_loss_factor=config['KERNEL_LOSS_FACTOR'],
                weight_regularization_loss_factor=config['WEIGHT_REGULARIZATION_LOSS_FACTOR'],
                weighted_xentropy1_factor=config['WEIGHTED_XENTROPY1_FACTOR'],
                weighted_xentropy2_factor=config['WEIGHTED_XENTROPY2_FACTOR'],
                map_dist_factor=config['MAP_DIST_FACTOR'],
                kld_factor=config['KLD_FACTOR'],
                sigma=config['SIGMA']
            )
        },
        optimizer_obj=tf.keras.optimizers.Adam(learning_rate=config['LEARNING_RATE'])
    )

    ver = 1
    PROG_NAME = config['PROG_NAME']

    my_model.fit(
        train_generator,
        epochs=config['EPOCHS'],
        validation_data=test_generator,
        verbose=0,
        callbacks=[
            MetricsCallback(
                best_model_crit="f1_score",
                path=f"../data/results/{PROG_NAME}_{ver}.keras",
                val_crit_score=0.0,
                num_classes=NUM_LAB,
                total_train_samples=TOTAL_TRAIN_SAMPLES,
                total_val_samples=TOTAL_VAL_SAMPLES
            ),
            SamplingLossActivationCallback(target_epoch=config['SAMPLING_LOSS_ACTIVATION_EPOCH'])
        ]
    )

if __name__ == '__main__':
    main()
