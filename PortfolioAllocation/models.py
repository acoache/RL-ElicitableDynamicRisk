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
from torch.nn.functional import silu, softplus
import torch.optim as optim
# misc
from pdb import set_trace

# normalise features of the neural nets
def normalise_features(x, env):
    # normalise features with environment parameters
    x[...,0] = x[...,0] / len(env.t) # time index 
    x[...,1:(1+len(env.S0))] = x[...,1:(1+len(env.S0))]/T.tensor(env.S0) - 1.0 # asset prices

    return x


# build a fully-connected neural net for the policy
class PolicyANN(nn.Module):
    # constructor
    def __init__(self, input_size, hidden_size, n_layers, env,
                    learn_rate=0.01, step_size=100, gamma=0.97):
        super(PolicyANN, self).__init__()

        self.input_size = input_size # number of inputs
        self.hidden_size = hidden_size # number of hidden nodes
        self.output_size = len(env.S0) # number of outputs
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
        # normalise features with environment parameters
        x = normalise_features(x, self.env)
        
        # mean of the Gaussian policy
        loc = silu(self.layer_in(x))
        
        for layer in self.hidden_layers:
            loc = silu(layer(loc))

        # output layer
        mu_mvn = 6.0*T.sigmoid(self.layer_out(loc))
        sigma_mvn = 2e-3*T.eye(self.output_size)

        return mu_mvn, sigma_mvn


# build a fully-connected neural net for the Value-at-Risk
class VaRANN(nn.Module):
    # constructor
    def __init__(self, input_size, hidden_size, n_layers, env,
                    learn_rate=0.01, step_size=50, gamma=0.95):
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
        # normalise features with environment parameters
        x = normalise_features(x, self.env)

        # value of the VaR
        x = silu(self.layer_in(x))
        
        for layer in self.hidden_layers:
            x = silu(layer(x))

        x = -4 + 2*4*T.sigmoid(self.layer_out(x))

        return x


# build a fully-connected neural net for the difference between CVaR and VaR
class DiffCVaRANN(nn.Module):
    # constructor
    def __init__(self, input_size, hidden_size, n_layers, env,
                    learn_rate=0.01, step_size=50, gamma=0.95):
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
        # normalise features with environment parameters
        x = normalise_features(x, self.env)

        # value of the difference between CVaR and VaR
        x = silu(self.layer_in(x))
        
        for layer in self.hidden_layers:
            x = silu(layer(x))

        x = 2*T.sigmoid(self.layer_out(x))

        return x