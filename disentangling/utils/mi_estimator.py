import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import scale
import numpy as np


###
# references:
# 1. https://github.com/MasanoriYamada/Mine_pytorch/blob/master/mine.ipynb
# 2. https://github.com/sungyubkim/MINE-Mutual-Information-Neural-Estimation-/blob/master/MINE.ipynb
# 3. https://github.com/gtegner/mine-pytorch/blob/master/mine/models/mine.py#L49
# 4. https://github.com/mboudiaf/Mutual-Information-Variational-Bounds
# 5. https://github.com/burklight/MINE-PyTorch/blob/master/src/mine.py
###


def atleast_2d(arr):
    if torch.is_tensor(arr):
        arr = arr.cpu().numpy()
    if len(arr.shape) == 1:
        return arr.reshape(arr.shape[0], 1)
    return arr


def hardly_decrease(history, patience=3, min_delta=0):
    history = np.array(history)
    if len(history) > patience:
        wait_idx = np.argmax((history - np.min(history)) <= min_delta)
        if len(history) - wait_idx > patience:
            print(
                "wait, ",
                wait_idx,
                history,
                (history - np.min(history)) <= min_delta,
            )
        return len(history) - wait_idx > patience
    return False


def default_transform(X):
    return scale(X, with_mean=False)


def is_1d(x):
    return len(x.shape) <= 1 or (len(x.shape) == 2 and x.shape[1] == 1)


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        x, y = self.data
        return x[idx], y[idx]


def init_weights(m):
    if (
        isinstance(m, nn.Linear)
        or isinstance(m, nn.ConvTranspose2d)
        or isinstance(m, nn.Conv2d)
    ):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


class Net(nn.Module):
    def __init__(self, input_shape=2):
        super(Net, self).__init__()
        n_hidden_dims = input_shape * 32
        # nn.Linear(n_hidden_dims, n_hidden_dims),
        #     nn.BatchNorm1d(n_hidden_dims),
        #     nn.ReLU(),
        self.backbone = nn.Sequential(
            nn.Linear(input_shape, n_hidden_dims),
            nn.ReLU(),
            nn.Linear(n_hidden_dims, n_hidden_dims),
            nn.ReLU(),
            nn.Linear(n_hidden_dims, n_hidden_dims),
            nn.ReLU(),
            nn.Linear(n_hidden_dims, n_hidden_dims),
            nn.ReLU(),
            nn.Linear(n_hidden_dims, 1),
        )
        # self.apply(init_weights)

    def forward(self, inputs):
        outs = self.backbone(inputs)
        return outs


class ema_log_mean_exp(torch.autograd.Function):
    # val: t_marginal.exp().mean().log()
    # grad: t_marginal.exp() / t_marginal.exp().mean() / t_marginal.shape[0]
    # ema grad: t_marginal.exp() / ema_et / t_marginal.shape[0]
    # ema val: t_marginal.exp().mean() / ema_et
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        return input.exp().mean().log()

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp() / running_mean / input.shape[0]
        return grad, None


def ema(val, val_mean, ema_rate=0.01):
    return (1 - ema_rate) * val_mean + ema_rate * torch.mean(val)


def mine(x, y, net):
    y_marginal = y[torch.randperm(y.shape[0])]
    t = net(torch.hstack([x, y]))
    t_marginal = net(torch.hstack([x, y_marginal]))
    mi = t.mean() - t_marginal.exp().mean().log()
    return mi, t, t_marginal


def estimate_mutual_information(x, y, **kwargs):
    x = atleast_2d(x).astype("float32")
    y = atleast_2d(y).astype("float32")
    if is_1d(x):
        x = default_transform(x)
    if is_1d(y):
        y = default_transform(y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(input_shape=(x.shape[1] + y.shape[1]))
    net.to(device)
    mi = train_mine(net, data=(x, y), **kwargs)
    return mi


def train_mine(
    net,
    data,
    loss_type="mi",
    n_epochs=200,
    batch_size=256,
    ema_rate=0.01,
    lr=1e-3,
    early_stop_min_delta=0.05,
    early_stop_patience=5,
):

    data = (torch.tensor(data[0]), torch.tensor(data[1]))
    dataloader = DataLoader(
        SimpleDataset(data),
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.95)
    ema_et = 1.0
    device = next(net.parameters()).device
    history = []
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    for i in range(n_epochs):
        for i_batch, (x, y) in enumerate(dataloader):
            mi, t, t_marginal = mine(x.to(device), y.to(device), net)
            # unbiasing use moving average, biased: -mi
            if loss_type == "mi":
                et = t_marginal.exp().mean()
                ema_et = ema(et, ema_et, ema_rate=ema_rate)
                val = ema_log_mean_exp.apply(t_marginal, ema_et.detach())
                loss = -(torch.mean(t) - val)
                # loss = -mi
            elif loss_type == "fdiv":
                loss = -(torch.mean(t) - torch.exp(t_marginal - 1).mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            est_mi, _, _ = mine(data[0].to(device), data[1].to(device), net)
        est_mi = est_mi.cpu().numpy()
        history.append(-est_mi)
        print("num_epoch", i, est_mi, history[-4:])
        if early_stop_patience > 0:
            if hardly_decrease(
                history,
                min_delta=early_stop_min_delta,
                patience=early_stop_patience,
            ):
                break
    return -np.min(history[-early_stop_patience:])
