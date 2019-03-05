import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader
import heteroscedastic_dropout_nn.impl as htffnn
from torch import optim

from sklearn.preprocessing import StandardScaler

def data():
    x = np.linspace(0.0, 20.0, 1000)
    y = np.sin(x) + np.tanh(x/8) * 4
    y = y + stats.norm(7).pdf(x) * 10

    x_train = x[200:800]
    y_train = y[200:800]
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    x_test = x
    y_test = y

    def f(x):
        return x.reshape(-1,1)

    y_train = y_train + np.random.randn(len(y_train)) * (np.sin(x_train/2-0.5)*0.5 + 0.25)

    return f(x_train), f(y_train), f(x_test), f(y_test)

def test_data():
    xtr, ytr, xte, yte = data()
    fig, ax = plt.subplots(1,1)
    ax.scatter(xtr, ytr, color='r', marker='x', label='training data')
    ax.plot(xte, yte, c='g', label='test data')
    ax.legend()
    plt.savefig('./data.png')

def test_fit():
    xtr, ytr, xte, yte = data()
    assert xtr.ndim == 2

    bs=64

    idx = np.random.permutation(len(xtr))
    tr_idx = idx[::2]
    val_idx = idx[1::2]

    scaler = StandardScaler()
    xtr_sc = scaler.fit_transform(xtr)
    xte_sc = scaler.transform(xte)

    scaler_out = StandardScaler()
    ytr_sc = scaler_out.fit_transform(ytr)

    train_ds = TensorDataset(torch.from_numpy(xtr_sc[tr_idx].astype(np.float32)), torch.from_numpy(ytr_sc[tr_idx].astype(np.float32)))
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

    val_ds = TensorDataset(torch.from_numpy(xtr_sc[val_idx].astype(np.float32)), torch.from_numpy(ytr_sc[val_idx].astype(np.float32)))
    val_dl = DataLoader(val_ds, batch_size=2*bs)

    def loss_batch(model, loss_func, xb, yb, opt=None):
        loss = loss_func(model(xb), yb)

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item(), len(xb)

    def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_dl:
                loss_batch(model, loss_func, xb, yb, opt)

            model.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[loss_batch(model, loss_func, xb, yb, opt=None) for xb, yb in valid_dl]
                )
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            if epoch % 100 == 0:
                print(epoch, val_loss)

    model = htffnn.DropoutFFNN(input_dim=1, output_dim=1*2, n_units=[64, 64, 64], activations=['relu']*3, dropout_ps=[0.1]*3)
    opt = optim.Adam(model.parameters())

    fit(500, model, htffnn.heteroscedastic_loss_1d, opt, train_dl, val_dl)

    model.eval()
    model.apply(htffnn.activate_dropout)
    n_samples_test = 30
    xte_torch = torch.from_numpy(xte_sc.astype(np.float32))
    with torch.no_grad():
        preds = []
        for i in range(n_samples_test):
            preds.append(model(xte_torch))
    preds = np.stack(preds) # n_samples_test x len(xte) x 2
    y_preds = preds[:,:,0].squeeze()
    y_log_precisions = preds[:,:,1].squeeze()

    y_mean = np.mean(y_preds, axis=0)
    y_stddev = np.mean(1 / np.exp(y_log_precisions), axis=0)

    # invert output scaling
    y_mean = scaler_out.inverse_transform(y_mean)
    y_stddev = y_stddev * scaler_out.scale_

    fig, ax = plt.subplots(1,1)
    ax.scatter(xtr[tr_idx], ytr[tr_idx], color='r', marker='x', label='training data')
    ax.plot(xte, yte, c='g', label='test data')
    ax.plot(xte, y_mean, c='b', label='prediction')
    ax.fill_between(xte.flatten(), (y_mean - 2*y_stddev).flatten(), (y_mean + 2*y_stddev).flatten(), color='b', label='2 sigma', alpha=0.3)
    ax.legend()
    plt.savefig('./fit.png')