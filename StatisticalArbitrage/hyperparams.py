"""
Hyperparameters
Initialization of all hyperparameters
Statistical arbitrage example
"""

# initialize parameters for the environment and algorithm
def init_params():
    # name of the repository
    repo_name = 'set1'

    # parameters for the model
    envParams = {'kappa' : 2, # kappa of the OU process
              'sigma' : 0.2, # standard deviation of the OU process
              'theta' : 1, # mean-reversion level of the OU process
              'phi' : 0.005, # transaction costs
              'psi' : 0.5, # terminal penalty on the inventory
              'T' : 1, # trading horizon
              'Ndt' : 5, # number of periods
              'min_q' : -5, # minimum value for the inventory
              'max_q' : 5, # maximum value for the inventory
              'min_u' : -2, # minimum value for the trades
              'max_u' : 2} # maximum value for the trades

    # parameters for the algorithm
    algoParams = {'Ntrajectories' : 10_000, # number of generated trajectories
                'Nepochs' : 1_500, # number of epochs of the whole algorithm
                'Nepochs_V_init' : 2_000, # number of epochs for the estimation of V during the first epoch
                'Nepochs_V' : 1_000, # number of epochs for the estimation of V
                'replace_target' : 400, # number of epochs before replacing the target networks
                'lr_V' : 5e-3, # learning rate of the neural net associated with V
                'batch_V' : 750, # number of trajectories for each mini-batch in estimating V
                'hidden_V' : 16, # number of hidden nodes in the neural net associated with V
                'layers_V' : 4, # number of layers in the neural net associated with V
                'Nepochs_pi' : 30, # number of epoch for the update of pi
                'lr_pi' : 5e-3, # learning rate of the neural net associated with pi
                'batch_pi' : 500, # number of trajectories for each mini-batch when updating pi
                'hidden_pi' : 16, # number of hidden nodes in the neural net associated with pi
                'layers_pi' : 4, # number of layers in the neural net associated with pi
                'seed' : None} # set seed for replication purposes

    return repo_name, envParams, algoParams


# print parameters for the environment and algorithm
def print_params(envParams, algoParams):
    print('*  T: ', envParams["T"],
            ' Ndt: ', envParams["Ndt"],
            ' kappa: ', envParams["kappa"],
            ' sigma: ', envParams["sigma"],
            ' theta: ', envParams["theta"],
            ' phi: ', envParams["phi"],
            ' psi: ', envParams["psi"])
    print('*  min_q: ', envParams["min_q"],
            ' max_q: ', envParams["max_q"],
            ' min_u: ', envParams["min_u"],
            ' max_u: ', envParams["max_u"])
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