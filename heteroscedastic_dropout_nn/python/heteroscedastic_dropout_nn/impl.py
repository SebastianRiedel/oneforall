import math
import torch
import torch.nn as nn

str_2_activation=dict(relu=nn.ReLU,
                      elu=nn.ELU)

def activate_dropout(m):
    """
    Set dropout layers to train mode aka apply dropout.
    """
    if type(m) == nn.Dropout:
        m.train()

def heteroscedastic_loss_1d(inputs, targets):
    """
    Assumes first network output is predicted value and second
    is the log_precision of the predicted gaussian observation
    uncertainty. I follow the convention 
       precision = 1 / std.dev (not variance aka std.dev^2)
    """
    pred = inputs[:,0]
    log_precision = inputs[:,1]
    nllh = -log_precision + 0.9189385332046727 - (-torch.pow(torch.exp(log_precision), 2) * torch.pow((torch.flatten(targets) - pred), 2) / 2)
    return torch.sum(nllh - log_precision)


class DropoutFFNN(nn.Module):
    """
    Simple feed-forward nn with dropout at every hidden layer, linear output layer.
    """
    def __init__(self, input_dim=1, output_dim=1, n_units=[10], activations=['relu'], dropout_ps=[0.05]):
        super(DropoutFFNN, self).__init__()

        D_ins = [input_dim] + n_units[:-1] # last layer is connected to output
        D_outs = n_units

        for i, (D_in, D_out, activation, dropout_p) in enumerate(zip(D_ins, D_outs, activations, dropout_ps)):
            name = 'linear_{}'.format(i)
            self.add_module(name, nn.Linear(D_in, D_out))

            name = 'activation_{}'.format(i)
            self.add_module(name, str_2_activation[activation]())

            name = 'dropout_{}'.format(i)
            self.add_module(name, nn.Dropout(p=dropout_p))

        name = 'output'
        self.add_module(name, nn.Linear(D_outs[-1], output_dim))

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x