"""
Main training script
Value function modeled by two ANNs; updated by consistent scoring functions
Statistical arbitrage example
"""
# personal files
from utils import directory
from hyperparams import init_params, print_params
from models import PolicyANN, VaRANN, DiffCVaRANN
from risk_measures import RiskMeasure
from envs import Environment
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
alphas_cvar = [0.5, 0.8, 0.9] # threshold for the conditional value-at-risk

repo_name, envParams, algoParams = init_params() # parameters for the model and algorithm
repo_name = "set1" # overwrite the repo name

print_progress = 300 # number of epochs before printing the time/loss
plot_progress = 300 # number of epochs before plotting the policy/value function
save_progress = 300 # number of epochs before saving the policy/value function ANNs

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
    data_repo = data_dir + '/Elicitability/' + repo_name

directory(repo)

# create labels to use for figures and folders
rm_labels = [] 
for idx_alpha, alpha in enumerate(alphas_cvar):
    rm_label = 'CVaR' + str( round(alpha, 3) )
    
    rm_labels.append(rm_label)
    
    # create a subfolder to store results
    directory(repo + '/' + rm_label)
    directory(repo + '/' + rm_label + '/evolution')

# print all parameters for reproducibility purposes
print('\n*** Name of the repository: ', repo_name, ' ***\n')
print_params(envParams, algoParams)
print('***   ', rm_labels, '   ***\n')


# loop for all risk measures
for idx_alpha, alpha in enumerate(alphas_cvar):
    # print progress
    print('\n*** Dynamic risk = ', rm_labels[idx_alpha], ' ***\n')
    start_time = time()

    # create the environment and risk measure objects
    env = Environment(envParams)
    risk_measure = RiskMeasure(alpha=alpha, C=25.0)

    # create ANNs (main and target) for policy, VaR and DiffCVaR
    policy = PolicyANN(input_size=3,
                        hidden_size=algoParams["hidden_pi"],
                        n_layers=algoParams["layers_pi"],
                        env=env,
                        learn_rate=algoParams["lr_pi"])
    VaR_main = VaRANN(input_size=3,
                    hidden_size=algoParams["hidden_V"],
                    n_layers=algoParams["layers_V"],
                    env=env,
                    learn_rate=algoParams["lr_V"])
    VaR_target = VaRANN(input_size=3,
                    hidden_size=algoParams["hidden_V"],
                    n_layers=algoParams["layers_V"],
                    env=env,
                    learn_rate=algoParams["lr_V"])
    DiffCVaR_main = DiffCVaRANN(input_size=3,
                                hidden_size=algoParams["hidden_V"],
                                n_layers=algoParams["layers_V"],
                                env=env,
                                learn_rate=algoParams["lr_V"])
    DiffCVaR_target = DiffCVaRANN(input_size=3,
                                hidden_size=algoParams["hidden_V"],
                                n_layers=algoParams["layers_V"],
                                env=env,
                                learn_rate=algoParams["lr_V"])
    
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

    # first estimate of the value function
    actor_critic.estimate_V(Ntrajectories=algoParams["Ntrajectories"],
                            Nminibatch=algoParams["batch_V"],
                            Nepochs=algoParams["Nepochs_V_init"],
                            replace_target=algoParams["replace_target"],
                            init_lr=algoParams["lr_V"],
                            rng_seed=algoParams["seed"])

    for epoch in range(algoParams["Nepochs"]):
        # estimate the value function of the current policy
        actor_critic.estimate_V(Ntrajectories=algoParams["Ntrajectories"],
                                Nminibatch=algoParams["batch_V"],
                                Nepochs=algoParams["Nepochs_V"],
                                replace_target=algoParams["replace_target"],
                                init_lr=algoParams["lr_V"],
                                rng_seed=algoParams["seed"])
        
        # update the policy by policy gradient
        actor_critic.update_policy(Nminibatch=int(algoParams["batch_pi"]/(1-alpha)),
                                    Nepochs=algoParams["Nepochs_pi"],
                                    rng_seed=algoParams["seed"])

        # print progress
        if epoch % print_progress == 0 or epoch == algoParams["Nepochs"] - 1:
            print('*** Epoch = ', str(epoch) ,
                    ' completed, Duration = ', "{:.3f}".format(time() - start_time), ' secs ***')
            start_time = time()

        # plot current policy
        if epoch % plot_progress == 0 or epoch == algoParams["Nepochs"] - 1:
            actor_critic.plot_current_policy(repo = repo + '/' + rm_labels[idx_alpha] + '/evolution')
            actor_critic.plot_current_V(repo = repo + '/' + rm_labels[idx_alpha] + '/evolution')

        # save progress
        if epoch % save_progress == 0:
            actor_critic.save_models(repo = repo + '/' + rm_labels[idx_alpha] + '/evolution')

    ###########################     END OF TRAINING     ##########################
    ##############################################################################

    # save the neural networks
    actor_critic.save_models(repo = repo + '/' + rm_labels[idx_alpha])
    # to load the model, M = ModelClass(*args, **kwargs); M.load_state_dict(T.load(PATH))

    # print progress
    print('*** Training phase completed! ***')