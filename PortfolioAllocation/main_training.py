"""
Main training script
Value function modeled by two ANNs; updated by consistent scoring functions
Portfolio allocation example
"""
# numpy
import numpy as np
# personal modules
from hyperparams import init_params, print_params
from envs_meanrevert import Environment
from utils import directory
from models import PolicyANN, VaRANN, DiffCVaRANN
from risk_measures import RiskMeasure
from agents import Agent
# misc
from time import time
from datetime import datetime
from os import getenv
from pdb import set_trace

"""
Parameters
"""

computer = 'personal' # 'cluster' | 'personal'
preload = False # load pre-trained model prior to the training phase
alphas_cvar = [0.05, 0.25, 0.5, 0.7] # threshold for the conditional value-at-risk

_, env_params, algo_params = init_params() # parameters for the model and algorithm
repo_name = "set1" # repo name

print_progress = 50 # number of epochs before printing the losses for policy/value function
plot_progress = 500 # number of epochs before plotting the policy/value function
save_progress = 500 # number of epochs before saving the policy/value function ANNs

"""
End of Parameters
"""

# create a new directory
if(computer == 'personal'): # personal computer
    repo = repo_name
    data_repo = repo_name
if(computer == 'cluster'): # Compute Canada server
    data_dir = getenv("HOME")
    output_dir = getenv("SCRATCH")
    repo = output_dir + '/' + repo_name
    data_repo = data_dir + '/PortfolioAllocation/' + repo_name

# create labels to use for figures and folders
rm_labels = [] 
directory(repo)
for idx_alpha, alpha in enumerate(alphas_cvar):
    rm_label = 'CVaR' + str( round(alpha, 3) )
    
    rm_labels.append(rm_label)
    
    # create a subfolder to store results
    directory(repo + '/' + rm_label)
    directory(repo + '/' + rm_label + '/evolution')

# print all parameters for reproducibility purposes
print('\n*** Name of the repository: ', repo_name, ' ***')
print_params(env_params, algo_params)
print('***   ', rm_labels, '   ***\n')


# loop for all risk measures
for idx_alpha, alpha in enumerate(alphas_cvar):
    # print progress
    print('\n*** Dynamic risk = ', rm_labels[idx_alpha], ' ***\n')
    start_time = time()
    
    # create the environment and risk measure objects
    env = Environment(env_params)
    risk_measure = RiskMeasure(alpha=alpha, C=10.0)

    # create ANNs (main and target) for policy, VaR and DiffCVaR
    policy = PolicyANN(input_size=1+len(env.S0),
                        hidden_size=algo_params["hidden_pi"],
                        n_layers=algo_params["layers_pi"],
                        env=env,
                        learn_rate=algo_params["lr_pi"])
    VaR_main = VaRANN(input_size=1+len(env.S0),
                    hidden_size=algo_params["hidden_V"],
                    n_layers=algo_params["layers_V"],
                    env=env,
                    learn_rate=algo_params["lr_V"])
    VaR_target = VaRANN(input_size=1+len(env.S0),
                    hidden_size=algo_params["hidden_V"],
                    n_layers=algo_params["layers_V"],
                    env=env,
                    learn_rate=algo_params["lr_V"])
    DiffCVaR_main = DiffCVaRANN(input_size=1+len(env.S0),
                                hidden_size=algo_params["hidden_V"],
                                n_layers=algo_params["layers_V"],
                                env=env,
                                learn_rate=algo_params["lr_V"])
    DiffCVaR_target = DiffCVaRANN(input_size=1+len(env.S0),
                                hidden_size=algo_params["hidden_V"],
                                n_layers=algo_params["layers_V"],
                                env=env,
                                learn_rate=algo_params["lr_V"])
    
    # initialize the actor-critic algorithm
    actor_critic = Agent(env=env,
                        risk_measure=risk_measure,
                        policy=policy,
                        VaR_main=VaR_main,
                        VaR_target=VaR_target,
                        DiffCVaR_main=DiffCVaR_main,
                        DiffCVaR_target=DiffCVaR_target)

    # load the weights of the pre-trained model
    if preload:
        actor_critic.load_models(repo = data_repo + '/' + rm_labels[idx_alpha])

    ##############################################################################
    ###########################     TRAINING PHASE     ###########################

    # set models in training mode
    actor_critic.set_train_mode(train=True)

    # first estimate of the value function
    actor_critic.estimate_V(Ntrajectories=algo_params["Ntrajectories"],
                            Nminibatch=algo_params["batch_V"],
                            Nepochs=algo_params["Nepochs_V_init"],
                            replace_target=algo_params["replace_target"],
                            init_lr=algo_params["lr_V"])

    for epoch in range(algo_params["Nepochs"]):
        # estimate the value function of the current policy
        actor_critic.estimate_V(Ntrajectories=algo_params["Ntrajectories"],
                                Nminibatch=algo_params["batch_V"],
                                Nepochs=algo_params["Nepochs_V"],
                                replace_target=algo_params["replace_target"],
                                init_lr=2e-3)
        
        # update the policy by policy gradient
        actor_critic.update_policy(Nminibatch=int(algo_params["batch_pi"]/(1-alpha)),
                                    Nepochs=algo_params["Nepochs_pi"],
                                    min_lr=3e-4)

        # plot current policy
        if epoch % plot_progress == 0 or epoch == algo_params["Nepochs"] - 1:
            actor_critic.plot_current_policy(repo = repo + '/' + rm_labels[idx_alpha] + '/evolution')
            actor_critic.plot_current_V(repo = repo + '/' + rm_labels[idx_alpha] + '/evolution')

        # save progress
        if epoch % save_progress == 0:
            actor_critic.save_models(repo = repo + '/' + rm_labels[idx_alpha] + '/evolution')
        # print progress
        if epoch % print_progress == 0 or epoch == algo_params["Nepochs"] - 1:
            print('*** Epoch = ', str(epoch) ,
                    ' completed, Duration = ', "{:.2f}".format((time() - start_time)/60), ' mins ***')
            print('Estimation of V -- Loss: ', str(np.round( np.mean(actor_critic.loss_history_V[-actor_critic.loss_trail_V:]) , 7)))
            print('Update of pi -- Loss: ', str(np.round( np.mean(actor_critic.loss_history_pi[-actor_critic.loss_trail_pi:]) ,5)),
                      ', lr: ', str(np.round( actor_critic.policy.optimizer.param_groups[0]['lr'] , 5)))
            start_time = time()
        
    ###########################     END OF TRAINING     ##########################
    ##############################################################################

    # save the neural networks
    actor_critic.save_models(repo = repo + '/' + rm_labels[idx_alpha])

    # print progress
    print('*** Training phase completed! ***')