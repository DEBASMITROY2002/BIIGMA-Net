import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
HEIGHT = 224
WIDTH = 224
BATCH_SIZE = 32
TRAIN_DIR = 'path/to/train'
VAL_DIR = 'path/to/validation'
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Using tf.data with ImageDataGenerator for augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=45,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1.0/255
)

# Flow from directory
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Convert to tf.data.Dataset for performance
train_ds = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, HEIGHT, WIDTH, 3], [None, train_generator.num_classes])
)

val_ds = tf.data.Dataset.from_generator(
    lambda: val_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, HEIGHT, WIDTH, 3], [None, val_generator.num_classes])
)

# Flatten batches and optimize
train_ds = (train_ds
    .unbatch()
    .shuffle(buffer_size=1000)
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=AUTOTUNE)
)

val_ds = (val_ds
    .unbatch()
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=AUTOTUNE)
)

# Class mapping and sample counts
CLASS_MAP = train_generator.class_indices
NUM_CLASSES = train_generator.num_classes
TOTAL_TRAIN_SAMPLES = train_generator.n
TOTAL_VAL_SAMPLES = val_generator.n

print(f"Classes: {CLASS_MAP}")
print(f"Total training samples: {TOTAL_TRAIN_SAMPLES}")
print(f"Total validation samples: {TOTAL_VAL_SAMPLES}")
