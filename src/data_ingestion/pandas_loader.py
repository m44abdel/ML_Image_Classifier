import pickle
import numpy as np

def load_batch(batch_file):

    with open(batch_file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']
        return data, labels