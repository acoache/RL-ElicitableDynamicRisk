"""
Hyperparameters - Same Sharpe ratios
Initialization of all hyperparameters
Portfolio allocation example
"""
# numpy
import numpy as np

# initialize parameters for the environment and algorithm
def init_params():
    repo_name = 'set1' # name of the repository

    # parameters for the model
    envParams = {"mu" : np.array([0.03,0.06,0.09]), # mean of risky assets
                "sigma" : np.array([0.06,0.12,0.18]), # volatility of risk assets
                "rho" : np.array([[1, 0.2, 0.2],[0.2, 1, 0.2],[0.2, 0.2, 1]]), # correlation of all assets
                "r0" : 0.00, # initial interest rate
                "S0" : np.array([1.0, 1.0, 1.0]), # initial risky asset prices
                "x0" : 1, # initial wealth
                "T" : 1, # time horizon of the problem
                "dt" : 1/12 # time elapsed between periods
                }

    # parameters for the algorithm
    algoParams = {'Ntrajectories' : 10_000, # number of generated trajectories
                'Nepochs' : 2_000, # number of epochs of the whole algorithm
                'Nepochs_V_init' : 2_000, # number of epochs for the estimation of V during the first epoch
                'Nepochs_V' : 1_000, # number of epochs for the estimation of V
                'replace_target' : 300, # number of epochs before replacing the target networks
                'lr_V' : 5e-3, # learning rate of the neural net associated with V
                'batch_V' : 1_000, # number of trajectories for each mini-batch in estimating V
                'hidden_V' : 16, # number of hidden nodes in the neural net associated with V
                'layers_V' : 4, # number of layers in the neural net associated with V
                'Nepochs_pi' : 10, # number of epoch for the update of pi
                'lr_pi' : 5e-3, # learning rate of the neural net associated with pi
                'batch_pi' : 1_000, # number of trajectories for each mini-batch when updating pi
                'hidden_pi' : 16, # number of hidden nodes in the neural net associated with pi
                'layers_pi' : 4, # number of layers in the neural net associated with pi
                'seed' : None} # set seed for replication purposes

    return repo_name, envParams, algoParams


# print parameters for the environment and algorithm
def print_params(envParams, algoParams):
    print('*  mu: ', envParams["mu"],
            '\n   sigma: ', envParams["sigma"],
            '\n   rho: ', envParams["rho"],
            '\n   S0: ', envParams["S0"],
            ' r0: ', envParams["r0"],
            ' x0: ', envParams["x0"],
            ' T: ', envParams["T"],
            ' dt: ', envParams["dt"])
    print('*  Ntrajectories: ', algoParams["Ntrajectories"],
            ' Nepochs: ', algoParams["Nepochs"])
    print('*  Nepochs_V_init: ', algoParams["Nepochs_V_init"],
            ' Nepochs_V: ', algoParams["Nepochs_V"],
            ' replace_target: ', algoParams["replace_target"],
            ' lr_V: ', algoParams["lr_V"], 
            ' batch_V: ', algoParams["batch_V"],
            ' hidden_V: ', algoParams["hidden_V"],
            ' layers_V: ', algoParams["layers_V"])
    print('*  Nepochs_pi: ', algoParams["Nepochs_pi"],
            ' lr_pi: ', algoParams["lr_pi"], 
            ' batch_pi: ', algoParams["batch_pi"],
            ' hidden_pi: ', algoParams["hidden_pi"],
            ' layers_pi: ', algoParams["layers_pi"])