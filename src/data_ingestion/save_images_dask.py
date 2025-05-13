import os
import sys
import numpy as np
import cv2
from dask import delayed, compute
from data_ingestion.pandas_loader import load_batch, load_cifar10_metadata

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "cifar-10-batches-py")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "images")
IMAGE_SIZE = (224, 224)

label_names = load_cifar10_metadata()

def save_image(image_array, label_idx, image_idx):
    label_name = label_names[label_idx]
    output_path = os.path.join(OUTPUT_DIR, label_name)
    os.makedirs(output_path, exist_ok=True)
    img = image_array.reshape(3, 32, 32).transpose(1, 2, 0)
    img = cv2.resize(img, IMAGE_SIZE)
    img_path = os.path.join(output_path, f"{label_name}_{image_idx}.png")
    cv2.imwrite(img_path, img)

def load_all_train_batches():
    all_data = []
    all_labels = []

    for i in range(1, 6):
        batch_file = os.path.join(RAW_DIR, f"data_batch_{i}")
        data, labels = load_batch(batch_file)
        all_data.append(data)
        all_labels.extend(labels)

    full_data = np.vstack(all_data)
    return full_data, all_labels

def main():
    data, labels = load_all_train_batches()
    print(f"Loaded {len(data)} images.")
    tasks = []

    for idx in range(len(data)):
        img = data[idx]
        label = labels[idx]
        tasks.append(delayed(save_image)(img, label, idx))

    print("Processing and saving images with Dask...")
    compute(*tasks)
    print("All images saved.")


if __name__ == "__main__":
    main()