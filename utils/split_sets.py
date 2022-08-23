import os
import numpy as np
import torch
from utils.general import read_raw_data


def __binary_score(y):
    score_threshold = 6
    labels = np.array(y)
    labels[labels <= score_threshold] = 0
    labels[labels > score_threshold] = 1
    unique, counts = np.unique(labels, return_counts=True)
    for idx, c in enumerate(counts):
        print(f'label: {idx}: {c} samples')
    return labels


def split_data():
    """
    splits the raw csv data to constant train and test sets and save those
    :return: none
    """
    shot_type = 'fco'

    sizes = {'fts': {'test_size': 50, 'val_size': 50},
             'fco': {'test_size': 50, 'val_size': 32},
             }

    test_size = sizes[shot_type]['test_size']
    val_size = sizes[shot_type]['val_size']

    output_path = f'../split_data/{shot_type}'
    os.makedirs(output_path, exist_ok=True)

    x, y, metadata = read_raw_data(score_path=f'../raw_data/{shot_type}_score.csv',
                                   data_path=f'../raw_data/{shot_type}_data.csv')
    labels = __binary_score(y)

    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    np.random.shuffle(pos)
    np.random.shuffle(neg)

    test_pos = pos[0: test_size]
    test_neg = neg[0: test_size]
    val_pos = pos[test_size: test_size+val_size]
    val_neg = neg[test_size: test_size+val_size]
    train_pos = pos[test_size+val_size:]
    train_neg = neg[test_size+val_size:]

    x_test = torch.vstack((x[test_pos], x[test_neg]))
    y_test = np.concatenate((labels[test_pos], labels[test_neg]))
    metadata_test = np.concatenate((metadata[test_pos], metadata[test_neg]))

    x_val = torch.vstack((x[val_pos], x[val_neg]))
    y_val = np.concatenate((labels[val_pos], labels[val_neg]))
    metadata_val = np.concatenate((metadata[val_pos], metadata[val_neg]))

    x_train = torch.vstack((x[train_pos], x[train_neg]))
    y_train = np.concatenate((labels[train_pos], labels[train_neg]))
    metadata_train = np.concatenate((metadata[train_pos], metadata[train_neg]))

    torch.save(x_train, f'{output_path}/x_train.pt')
    torch.save(x_test, f'{output_path}/x_test.pt')
    torch.save(x_val, f'{output_path}/x_val.pt')

    np.save(f'{output_path}/y_train', y_train)
    np.save(f'{output_path}/y_test', y_test)
    np.save(f'{output_path}/y_val', y_val)
    np.save(f'{output_path}/metadata_train', metadata_train)
    np.save(f'{output_path}/metadata_test', metadata_test)
    np.save(f'{output_path}/metadata_val', metadata_val)


def load_data(shot_type):
    data_path = f'split_data/{shot_type}'

    x_train = torch.load(f'{data_path}/x_train.pt')
    x_test = torch.load(f'{data_path}/x_test.pt')
    x_val = torch.load(f'{data_path}/x_val.pt')

    y_train = np.load(f'{data_path}/y_train.npy')
    y_test = np.load(f'{data_path}/y_test.npy')
    y_val = np.load(f'{data_path}/y_val.npy')

    metadata_train = np.load(f'{data_path}/metadata_train.npy', allow_pickle=True)
    metadata_test = np.load(f'{data_path}/metadata_test.npy', allow_pickle=True)
    metadata_val = np.load(f'{data_path}/metadata_val.npy', allow_pickle=True)

    return x_train, x_test, x_val, y_train, y_test, y_val, metadata_train, metadata_test, metadata_val


if __name__ == "__main__":
    split_data()
