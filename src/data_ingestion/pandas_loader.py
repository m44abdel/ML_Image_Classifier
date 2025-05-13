import pickle
import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
# label_names_path = os.path.join(DATA_DIR, "cifar-10-batches-py", "batches.meta")

def load_batch(batch_file):
    with open(batch_file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']
        return data, labels
    
def load_cifar10_metadata():
    label_names_path = os.path.join(DATA_DIR, "cifar-10-batches-py", "batches.meta")
    with open(label_names_path, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
        return [label.decode("utf-8") for label in meta[b'label_names']]
    
# def download_cifar10():
#     url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
#     filename = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")

#     if not os.path.exists(DATA_DIR):
#         os.makedirs(DATA_DIR)

#     if not os.path.exists(filename):
#         print("Downloading CIFAR-10...")
#         urllib.request.urlretrieve(url, filename)

#     with tarfile.open(filename, "r:gz") as tar:
#         tar.extractall(path=DATA_DIR)
#     print("Download and extraction complete.")