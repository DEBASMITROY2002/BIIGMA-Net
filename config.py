import os
import json

def set_default_config():
    """
    Sets the default configuration in environment variables.
    """
    os.environ["PROG_NAME"] = "BIGMA_HAM_4H"
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
    os.environ["TRAIN_DIR"] = "../data/train"
    os.environ["TEST_DIR"] = "../data/test"
    os.environ["BASE_PATH"] = "../data/"
    os.environ["BATCH_SIZE"] = "2"
    os.environ["HEIGHT"] = "224"
    os.environ["WIDTH"] = "224"
    os.environ["KERNEL_LOSS_FACTOR"] = "0.5"
    os.environ["WEIGHT_REGULARIZATION_LOSS_FACTOR"] = "0.5"
    os.environ["WEIGHTED_XENTROPY1_FACTOR"] = "0.5"
    os.environ["WEIGHTED_XENTROPY2_FACTOR"] = "0.0"
    os.environ["MAP_DIST_FACTOR"] = "0.5"
    os.environ["KLD_FACTOR"] = "0.0"
    os.environ["SIGMA"] = "2.5"
    os.environ["LEARNING_RATE"] = "0.00005"
    os.environ["NUM_HEADS"] = "3"
    os.environ["TEMPERATURE"] = "2.0"
    os.environ["SIDE_OUTPUT_LAYER_NAMES"] = "['pool3_relu', 'pool4_relu', 'relu']"
    os.environ["EPOCHS"] = "65"
    os.environ["SAMPLING_LOSS_ACTIVATION_EPOCH"] = "10"

def get_config():
    """
    Returns a dictionary with the configuration.
    """
    return {
        "PROG_NAME": os.environ.get("PROG_NAME"),
        "TF_GPU_THREAD_MODE": os.environ.get("TF_GPU_THREAD_MODE"),
        "TRAIN_DIR": os.environ.get("TRAIN_DIR"),
        "TEST_DIR": os.environ.get("TEST_DIR"),
        "BASE_PATH": os.environ.get("BASE_PATH"),
        "BATCH_SIZE": int(os.environ.get("BATCH_SIZE")),
        "HEIGHT": int(os.environ.get("HEIGHT")),
        "WIDTH": int(os.environ.get("WIDTH")),
        "KERNEL_LOSS_FACTOR": float(os.environ.get("KERNEL_LOSS_FACTOR")),
        "WEIGHT_REGULARIZATION_LOSS_FACTOR": float(os.environ.get("WEIGHT_REGULARIZATION_LOSS_FACTOR")),
        "WEIGHTED_XENTROPY1_FACTOR": float(os.environ.get("WEIGHTED_XENTROPY1_FACTOR")),
        "WEIGHTED_XENTROPY2_FACTOR": float(os.environ.get("WEIGHTED_XENTROPY2_FACTOR")),
        "MAP_DIST_FACTOR": float(os.environ.get("MAP_DIST_FACTOR")),
        "KLD_FACTOR": float(os.environ.get("KLD_FACTOR")),
        "SIGMA": float(os.environ.get("SIGMA")),
        "LEARNING_RATE": float(os.environ.get("LEARNING_RATE")),
        "NUM_HEADS": int(os.environ.get("NUM_HEADS")),
        "TEMPERATURE": float(os.environ.get("TEMPERATURE")),
        "SIDE_OUTPUT_LAYER_NAMES": json.loads(os.environ.get("SIDE_OUTPUT_LAYER_NAMES").replace("'", '"')),
        "EPOCHS": int(os.environ.get("EPOCHS")),
        "SAMPLING_LOSS_ACTIVATION_EPOCH": int(os.environ.get("SAMPLING_LOSS_ACTIVATION_EPOCH")),
    }
