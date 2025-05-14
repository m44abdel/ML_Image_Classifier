import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "images")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "tf_model.keras")

BATCH_SIZE = 64
IMG_SIZE = (224, 224)
EPOCHS = 10

early_stop = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 2, restore_best_weights = True)

def load_datasets():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        seed = 123,
        validation_split = 0.2,
        subset = "training",
        image_size = IMG_SIZE,
        batch_size = BATCH_SIZE
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        seed = 123,
        validation_split = 0.2,
        subset = "validation",
        image_size = IMG_SIZE,
        batch_size = BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"Class names: {class_names}")

    AUTOTONE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size = AUTOTONE)
    val_ds = val_ds.prefetch(buffer_size = AUTOTONE)

    return train_ds, val_ds, class_names

def build_model(num_classes):
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape = IMG_SIZE + (3,)),
        layers.Conv2D(32, 3, activation = "relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation = "relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation = "relu"),
        layers.Dense(num_classes, activation = "softmax")
    ])

    return model

def train():
    train_ds, val_ds, class_names = load_datasets()
    model = build_model(len(class_names))

    model.compile(
        optimizer = "adam",
        loss = "sparse_categorical_crossentropy",
        metrics = ["accuracy"]
    )
    model.fit(train_ds, validation_data = val_ds, epochs = EPOCHS, callbacks = [early_stop])

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok = True)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
    print("Training completed.")
