import numpy as np
import torch
from matplotlib import pyplot as plt
from utils.split_sets import load_data
import pandas as pd
from utils.general import rand_shot
from tqdm import tqdm
from utils.learn import train_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, accuracy_score
from utils.learn import predict
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math
import torch.nn.functional as F


def print_set_labels(y_set, name):
    unique, counts = np.unique(y_set, return_counts=True)
    for idx, c in enumerate(counts):
        print(f'{name}: label: {idx}: {c} samples')


def stack_data(data_pos_, data_neg_, data_new, stack_foo):
    """
    stack x,y, metadata at the same fashion such that the indices will align
    """
    data_pos_ = stack_foo((data_pos_, data_new))
    return stack_foo((data_pos_, data_neg_))


def feature_selection(x_set, remove_features):
    idxs = []
    for p in remove_features:
        idxs.extend(list(range(p * 4, p * 4 + 4)))
    return np.delete(x_set, idxs, axis=2)


# config
# Network
HIDDEN_DIM = 64
OUTPUT_DIM = 2
BATCH_SIZE = 32
IS_BID = True
NUM_LAYERS = 2
NUM_FEATURES = 100

# Optimizer
lr = 1e-3 / 5
weight_decay = 0.001

# Scheduler
step_size = 10
gamma = 0.5

early_stop_patience = 8


class TTDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels.iloc[idx]['label'], idx


def create_dataloaders(batch_size):
    train_data = TTDataset(x_train, train_labels)
    test_data = TTDataset(x_test, test_labels)
    validation_data = TTDataset(x_val, val_labels)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    return train_loader, validation_loader, test_loader


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale = 1. / math.sqrt(hidden_size)

    def forward(self, hidden, outputs):
        hidden = hidden.unsqueeze(1)
        weights = torch.bmm(hidden, outputs.transpose(1, 2))

        scores = F.softmax(weights.mul_(self.scale), dim=2)
        linear_combination = torch.bmm(scores, outputs).squeeze(1)
        return linear_combination


class Network(nn.Module):
    def __init__(self, input_dim, h_dim, out_dim, num_layers, bidirectional, use_attention, device):
        super().__init__()
        self.num_layers = num_layers
        self.h_dim = h_dim
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.dropout = nn.Dropout(0.5)
        self.device = device

        hid_bidirectional = h_dim * 2 if bidirectional else h_dim
        self.atten = Attention(hid_bidirectional)

        # LSTM
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=h_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=0.5)
        # FC layer
        self.fc = nn.Linear(hid_bidirectional, out_dim, bias=True)

        # To convert class scores to log-probability we'll apply log-softmax
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, X, batch_size):
        lay_times_dir = self.num_layers * 2 if self.bidirectional else self.num_layers
        h0 = torch.zeros(lay_times_dir, batch_size, HIDDEN_DIM).to(self.device)
        c0 = torch.zeros(lay_times_dir, batch_size, HIDDEN_DIM).to(self.device)

        out, (h_t, c_t) = self.lstm(X, (h0, c0))
        if self.bidirectional:
            cell_state = torch.cat([c_t[-1], c_t[-2]], dim=1)
        else:
            cell_state = c_t[-1]

        # local attention!
        classify_input = self.atten(cell_state, out) if self.use_attention else cell_state
        y = self.fc(classify_input)

        yt_log_proba = self.log_softmax(y)
        return yt_log_proba


if __name__ == "__main__":
    # Reading data
    x_train, x_test, x_val, y_train, y_test, y_val, metadata_train, metadata_test, metadata_val = load_data('fco')
    print(f'orignal data sizes: train size: {x_train.shape[0]} test size: {x_test.shape[0]} val size: {x_val.shape[0]}')

    print_set_labels(y_train, 'y_train')
    print_set_labels(y_test, 'y_test')
    print_set_labels(y_val, 'y_test')

    print('\ntrain data still un balanced!')

    # Augmentation + Up sampling
    frame_prob = 0

    pos = np.where(y_train == 1)
    neg = np.where(y_train == 0)

    metadata_pos = metadata_train[pos]
    metadata_neg = metadata_train[neg]

    data_pos = x_train[pos]
    data_neg = x_train[neg]

    y_pos = y_train[pos]
    y_neg = y_train[neg]

    up_sample_amount = len(neg[0]) - len(pos[0])
    print(f'generating {up_sample_amount} shots')

    new_data = []
    new_metadata = []
    new_y = np.ones(up_sample_amount)

    rand_amount = 0
    run_ = True

    with tqdm(total=up_sample_amount) as pbar:
        while run_:
            for shot_idx, shot in enumerate(data_pos):
                amount = metadata_pos[shot_idx]['frames']
                name = metadata_pos[shot_idx]['name']
                new_shot = rand_shot(shot, amount, frame_prob=frame_prob)

                new_metadata.append({'name': f'{name}_rand_{rand_amount}', 'frames': amount})
                new_data.append(new_shot)

                rand_amount += 1
                pbar.update(1)

                if rand_amount == up_sample_amount:
                    run_ = False
                    break

    new_data_pos = torch.stack([shot for shot in new_data])

    x_train = stack_data(data_pos, data_neg, new_data_pos, torch.vstack)
    y_train = stack_data(y_pos, y_neg, new_y, np.concatenate)
    metadata_train = stack_data(metadata_pos, metadata_neg, new_metadata, np.concatenate)

    print('x train shape', x_train.shape)
    print_set_labels(y_train, 'y_train')
    print('\ntrain data is now balanced!')

    # data set & loaders
    train_labels = pd.DataFrame({'label': y_train, 'metadata': metadata_train})
    val_labels = pd.DataFrame({'label': y_val, 'metadata': metadata_val})
    test_labels = pd.DataFrame({'label': y_test, 'metadata': metadata_test})

    print(f'data sizes: train size: {x_train.shape[0]} val size: {x_val.shape[0]} test size: {x_test.shape[0]}')

    train_loader, validation_loader, test_loader = create_dataloaders(BATCH_SIZE)

    # device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    print('device: ', device)

    # prepare model
    use_attention = True
    model = Network(NUM_FEATURES, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, IS_BID, use_attention, device).to(device)

    features_name = 'all_features' if NUM_FEATURES == 100 else 'no_head'
    aug_name = 'with_aug' if frame_prob > 0 else 'no_aug'
    att_name = 'with_attn' if use_attention else 'no_attn'
    model_name = f'b{BATCH_SIZE}_lr{lr}_sz{step_size}_g{gamma}_h{HIDDEN_DIM}_nl{NUM_LAYERS}_{features_name}_{aug_name}_{att_name}.pt'
    print(model_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.NLLLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_model(model, model_name, BATCH_SIZE, device, early_stop_patience,
                train_loader, validation_loader, test_loader,
                optimizer, scheduler, loss_fn)

    # false analysis
    model.load_state_dict(torch.load('model_results/' + model_name))
    model.eval()
    correct_idx, wrong_idx, y_pred, y_true, _ = predict(model, test_loader, device, batch_size=1)
    print('prec', precision_score(y_true, y_pred))
    print('acc', accuracy_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
