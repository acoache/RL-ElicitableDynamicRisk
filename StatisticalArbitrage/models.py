"""
Models
Policy: single fully-connected ANN
Value function: two fully-connected ANNs 
"""
# numpy
import numpy as np
# pytorch
import torch as T
import torch.nn as nn
from torch.nn.functional import silu
import torch.optim as optim
# misc
from pdb import set_trace

# normalize features of the neural nets
def normalize_features(x, env):
    # normalize features with environment parameters
    x[...,0] /= env.params["Ndt"] # time
    x[...,1] = (x[...,1] - env.params["theta"]) / 0.5 # price of the asset
    x[...,2] /= env.params["max_q"] # inventory

    return x


# build a fully-connected neural net for the policy
class PolicyANN(nn.Module):
    # constructor
    def __init__(self, input_size, hidden_size, n_layers, env,
                    learn_rate=0.01, step_size=50, gamma=0.95):
        super(PolicyANN, self).__init__()
        
        self.input_size = input_size # number of inputs
        self.hidden_size = hidden_size # number of hidden nodes
        self.output_size = 1 # number of outputs
        self.n_layers = n_layers # number of layers
        self.env = env # environment (for normalisation purposes)

        # build all layers
        self.layer_in = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.n_layers-1)])
        self.layer_out = nn.Linear(self.hidden_size, self.output_size)

        # initializers for weights and biases
        nn.init.normal_(self.layer_in.weight, mean=0, std=1/np.sqrt(input_size)/2)
        nn.init.constant_(self.layer_in.bias, 0)
        for layer in self.hidden_layers:
            nn.init.normal_(layer.weight, mean=0, std=1/np.sqrt(input_size)/2)
            nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.layer_out.weight, mean=0, std=1/np.sqrt(input_size)/2)
        nn.init.constant_(self.layer_out.bias, 0)

        # optimizer and scheduler
        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate) 
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    

    # forward propagation
    def forward(self, x):
        # normalize features with environment parameters
        x = normalize_features(x, self.env)
        
        # mean of the Gaussian policy
        loc = silu(self.layer_in(x))
        
        for layer in self.hidden_layers:
            loc = silu(layer(loc))

        # output layer attempts
        loc = -5 + (10)*T.sigmoid(self.layer_out(loc))

        # standard deviation of the Gaussian policy
        scale = 0.02

        return loc, scale


# build a fully-connected neural net for the Value-at-Risk
class VaRANN(nn.Module):
    # constructor
    def __init__(self, input_size, hidden_size, n_layers, env,
                    learn_rate=0.01, step_size=100, gamma=0.95):
        super(VaRANN, self).__init__()
        
        self.input_size = input_size # number of inputs
        self.hidden_size = hidden_size # number of hidden nodes
        self.output_size = 1 # number of outputs
        self.n_layers = n_layers # number of layers
        self.env = env # environment (for normalisation purposes)
        
        # build all layers
        self.layer_in = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.n_layers-1)])
        self.layer_out = nn.Linear(self.hidden_size, self.output_size)
        
        # optimizer and scheduler
        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate) 
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    

    # forward propagation
    def forward(self, x):
        # normalize features with environment parameters
        x = normalize_features(x, self.env)

        # value of the value function
        x = silu(self.layer_in(x))
        
        for layer in self.hidden_layers:
            x = silu(layer(x))

        x = -18 + 2*18*T.sigmoid(self.layer_out(x))

        return x


# build a fully-connected neural net for the difference between CVaR and VaR
class DiffCVaRANN(nn.Module):
    # constructor
    def __init__(self, input_size, hidden_size, n_layers, env,
                    learn_rate=0.01, step_size=100, gamma=0.95):
        super(DiffCVaRANN, self).__init__()
        
        self.input_size = input_size # number of inputs
        self.hidden_size = hidden_size # number of hidden nodes
        self.output_size = 1 # number of outputs
        self.n_layers = n_layers # number of layers
        self.env = env # environment (for normalisation purposes)
        
        # build all layers
        self.layer_in = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.n_layers-1)])
        self.layer_out = nn.Linear(self.hidden_size, self.output_size)
        
        # optimizer and scheduler
        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate) 
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    

    # forward propagation
    def forward(self, x):
        # normalize features with environment parameters
        x = normalize_features(x, self.env)

        # value of the value function
        x = silu(self.layer_in(x))
        
        for layer in self.hidden_layers:
            x = silu(layer(x))

        x = 6*T.sigmoid(self.layer_out(x))

        return x