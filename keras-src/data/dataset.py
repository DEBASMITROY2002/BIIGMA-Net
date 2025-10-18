import tensorflow as tf
import zipfile
from config import get_config

config = get_config()

def extract_zip(src_path, targ_path):
    with zipfile.ZipFile(src_path,"r") as zip_ref:
        zip_ref.extractall(targ_path)

def get_data_generators():
    TRAIN_DIR = config['TRAIN_DIR']
    TEST_DIR = config['TEST_DIR']
    BASE_PATH = config['BASE_PATH']
    BATCH_SIZE = config['BATCH_SIZE']
    HEIGHT = config['HEIGHT']
    WIDTH = config['WIDTH']

    extract_zip(BASE_PATH + 'ham_train.zip', TRAIN_DIR)
    extract_zip(BASE_PATH + 'ham_test.zip', TEST_DIR)

    TRAIN_DATAGEN = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255,
    )

    TEST_DATAGEN = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1.0/255
    )

    train_generator = TRAIN_DATAGEN.flow_from_directory(
        TRAIN_DIR,
        target_size=(HEIGHT, WIDTH),
        batch_size= BATCH_SIZE,
        class_mode='categorical',
    )

    test_generator = TEST_DATAGEN.flow_from_directory(
        TEST_DIR,
        target_size = (HEIGHT, WIDTH),
        batch_size = BATCH_SIZE,
        class_mode = 'categorical',
    )

    return train_generator, test_generator
