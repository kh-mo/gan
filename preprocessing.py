""" Preprocess MNIST dataset """

import os
import torch
import codecs
import numpy as np

def preprocessing():
    raw_folder = "raw_data"
    preprocessed_folder = "preprocess"
    training_file = 'train.pt'
    test_file = 'test.pt'

    # read image, label file
    training_set = (
        read_image_file(os.path.join("./", raw_folder, 'train-images-idx3-ubyte')),
        read_label_file(os.path.join("./", raw_folder, 'train-labels-idx1-ubyte'))
    )
    test_set = (
        read_image_file(os.path.join("./", raw_folder, 't10k-images-idx3-ubyte')),
        read_label_file(os.path.join("./", raw_folder, 't10k-labels-idx1-ubyte'))
    )

    # folder for preprocess
    try:
        os.makedirs(os.path.join("./", preprocessed_folder))
    except FileExistsError as e:
        pass

    # save train, test file
    with open(os.path.join("./", preprocessed_folder, training_file), 'wb') as f:
        torch.save(training_set, f)
    with open(os.path.join("./", preprocessed_folder, test_file), 'wb') as f:
        torch.save(test_set, f)

    print("preprocessing done.")

def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols).type(torch.FloatTensor)

def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

if __name__=="__main__":
    preprocessing()
