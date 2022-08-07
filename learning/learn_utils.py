import torch
import numpy as np
import matplotlib.pyplot as plt


def validation_loss(model, val_loader, device, batch_size, loss_fn):
    losses = []

    with torch.no_grad():
        for batch in val_loader:
            X, y = batch
            if X.shape[0] != batch_size:
                break
            y_pred_log_proba = model(X.to(device))
            loss = loss_fn(y_pred_log_proba, y.to(device).long())
            losses.append((loss.detach().cpu()))

    return np.mean(losses)


def calculate_acc(model, dataset_loader, device, batch_size):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for batch in dataset_loader:
            X, y = batch
            if X.shape[0] != batch_size:
                break
            y_pred_log_proba = model(X.to(device))
            y_pred = torch.argmax(y_pred_log_proba, dim=1).view(batch_size)
            n_correct += torch.sum(y_pred == y.to(device)).float().item()
            n_total += batch_size

    acc = n_correct / n_total
    return acc


def plot_results(train_, valid_, patience, unit):
    plt.figure()
    plt.plot(train_, label='train')
    plt.plot(valid_, label='validation')
    plt.axvline(len(valid_) - patience - 1, color='purple', linestyle='--', label='best model')
    plt.title(unit)
    plt.xlabel('Epochs')
    plt.ylabel(unit)
    plt.legend()
    plt.show()


def train_model(model, model_name, batch_size, device, patience,
                train_loader, validation_loader, test_loader,
                optimizer, scheduler, loss_fn):
    train_loss = []
    valid_loss = []
    train_acc = []
    val_acc = []

    j = 0
    i = 0
    min_val_lost = 1000

    while j < patience:
        losses = []
        model.train()
        for batch in train_loader:
            X, y = batch
            if X.shape[0] != batch_size:
                break
            y_pred_log_proba = model(X.to(device))
            optimizer.zero_grad()

            loss = loss_fn(y_pred_log_proba, y.to(device).long())
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu())

        scheduler.step()
        model.eval()

        # Loss graphs
        tr_loss = np.mean(losses)
        train_loss.append(tr_loss)
        val_loss = validation_loss(model, validation_loader, device, batch_size, loss_fn)
        valid_loss.append(val_loss)
        print("epoch {} | train loss : {} validation loss: {} ".format(i, tr_loss, val_loss))

        # Accuracy graphs
        tr_acc = calculate_acc(model, train_loader, device, batch_size)
        train_acc.append(tr_acc)
        v_acc = calculate_acc(model, validation_loader, device, batch_size)
        val_acc.append(v_acc)
        print("epoch {} | train acc : {} validation acc: {} ".format(i, tr_acc, v_acc))

        # early stopping
        if val_loss < min_val_lost:
            min_val_lost = val_loss
            j = 0
            torch.save(model.state_dict(), model_name)
        else:
            j += 1

        i += 1

    model.load_state_dict(torch.load(model_name))
    model.eval()
    test_accuracy = calculate_acc(model, test_loader, device, batch_size)
    print(f'Test accuracy: {test_accuracy}')

    plot_results(train_loss, valid_loss, patience, 'Loss')
    plot_results(train_acc, val_acc, patience, 'Accuracy')
