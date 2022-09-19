"""
Main testing script
2 plots: learnt policies, terminal reward
Statistical arbitrage example
"""
# numpy
import numpy as np
# plotting
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# pytorch
import torch as T
# personal files
from utils import directory, cmap, colors
from hyperparams import init_params, print_params
from models import PolicyANN, VaRANN, DiffCVaRANN
from risk_measures import RiskMeasure
from envs import Environment
from agents import Agent
# misc
from pdb import set_trace

"""
Parameters
"""

rm_labels = ['CVaR0.5', 'CVaR0.8', 'CVaR0.9'] # risk measures used
seed = 4321 # set seed for replication purposes
Nsimulations = 30_000 # number of simulations following the optimal strategy

_, envParams, algoParams = init_params()
repo = "set1" # repo name

"""
End of Parameters
"""

# print all parameters for reproducibility purposes
print('\n*** Name of the repository: ', repo, ' ***\n')
print_params(envParams, algoParams)

directory(repo) # create the directory

env = Environment(envParams) # create the environment

costs = np.zeros((Nsimulations, envParams["Ndt"], len(rm_labels))) # matrix to store all testing trajectories

for idx_method, method in enumerate(rm_labels):
    # print progress
    print('\n*** Method = ', method, ' ***\n')

    # create ANNs for policy, VaR and DiffCVaR
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
    DiffCVaR_main = DiffCVaRANN(input_size=3,
                                hidden_size=algoParams["hidden_V"],
                                n_layers=algoParams["layers_V"],
                                env=env,
                                learn_rate=algoParams["lr_V"])
    
    # initialize the actor-critic algorithm
    actor_critic = Agent(env=env,
                        risk_measure=[],
                        policy=policy,
                        VaR_main=VaR_main,
                        VaR_target=[],
                        DiffCVaR_main=DiffCVaR_main,
                        DiffCVaR_target=[])

    # load the trained model
    actor_critic.load_models(repo = repo + '/' + method)

    ##############################################################################
    ###########################     TESTING PHASE     ############################

    # set seed for reproducibility purposes
    T.manual_seed(seed)
    np.random.seed(seed)

    # initialize the starting state
    time_t, s_t, q_t = env.reset(Nsimulations)
    
    for t_idx in env.spaces["t_space"]:
        # simulate transitions according to the policy
        action_t, _ = actor_critic.select_actions(time_t, s_t, q_t, 'best')
        time_t, s_t, q_t, cost_t = env.step(time_t, s_t, q_t, action_t)

        # store costs
        costs[:,t_idx,idx_method] = cost_t.detach().numpy()

    # print progress
    print('*** Testing phase completed! ***')

    ###########################     END OF TESTING     ###########################
    ##############################################################################

    ##############################################################################
    #####################     learnt policy at every time     ####################
    
    # figure parameters
    plt.rcParams.update({'font.size': 16, 'figure.figsize': (10,7)})
    plt.rc('axes', labelsize=20)

    for t_idx, t_val in enumerate(env.spaces["t_space"]):        
        # initialize 2D histogram
        hist2dim_policy = np.zeros([len(env.spaces["s_space"]), len(env.spaces["q_space"])])

        for s_idx, s_val in enumerate(env.spaces["s_space"]):
            for q_idx, q_val in enumerate(env.spaces["q_space"]):
                # best action according to the policy
                hist2dim_policy[len(env.spaces["s_space"])-s_idx-1, q_idx], _ = \
                        actor_critic.select_actions(t_val*T.ones(1, device=actor_critic.device),
                                                    s_val*T.ones(1, device=actor_critic.device),
                                                    q_val*T.ones(1, device=actor_critic.device),
                                                    'best')

        # plot the 2D histogram
        plt.imshow(hist2dim_policy,
                interpolation='none',
                cmap=cmap,
                extent=[np.min(env.spaces["q_space"]),
                        np.max(env.spaces["q_space"]),
                        np.min(env.spaces["s_space"]),
                        np.max(env.spaces["s_space"])],
                aspect='auto',
                vmin=env.params["min_u"],
                vmax=env.params["max_u"])
        
        plt.title('Learned; Period:' + str(t_idx))
        plt.xlabel("Inventory")
        plt.ylabel("Price")
        cbar = plt.colorbar()
        cbar.set_ticks([env.params["min_u"],-1,0,1,env.params["max_u"]])
        cbar.set_ticklabels([env.params["min_u"],-1,0,1,env.params["max_u"]])
        plt.tight_layout()
        plt.savefig(repo + '/best_actions-' + method + '-period' + str(t_idx) + '.pdf', transparent=True)
        plt.clf()

    #####################     learnt policy at every time     ####################
    ##############################################################################

    ##############################################################################
    #####################     learnt policy in one figure     ####################
    
    # figure parameters
    plt.rcParams.update({'font.size': 16, 'figure.figsize': (10,4)})
    plt.rc('axes', labelsize=20)

    for t_idx, t_val in enumerate([0, int(env.spaces["t_space"][-1]/2.0), env.spaces["t_space"][-1]]):
        # allocate the subplot
        plt.subplot(1, 3, t_idx+1)
        
        # initialize 2D histogram
        hist2dim_policy = np.zeros([len(env.spaces["s_space"]), len(env.spaces["q_space"])])

        for s_idx, s_val in enumerate(env.spaces["s_space"]):
            for q_idx, q_val in enumerate(env.spaces["q_space"]):
                # best action according to the policy
                hist2dim_policy[len(env.spaces["s_space"])-s_idx-1, q_idx], _ = \
                        actor_critic.select_actions(t_val*T.ones(1, device=actor_critic.device),
                                                    s_val*T.ones(1, device=actor_critic.device),
                                                    q_val*T.ones(1, device=actor_critic.device),
                                                    'best')

        # plot the 2D histogram
        plt.imshow(hist2dim_policy,
                interpolation='none',
                cmap=cmap,
                extent=[np.min(env.spaces["q_space"]),
                        np.max(env.spaces["q_space"]),
                        np.min(env.spaces["s_space"]),
                        np.max(env.spaces["s_space"])],
                aspect='auto',
                vmin=env.params["min_u"],
                vmax=env.params["max_u"])
        
        plt.title('Learned; Time:' + str(t_val))
        plt.xlabel("Inventory")
        plt.ylabel("Price")
        plt.tight_layout()

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(repo + '/best_actions-' + method + '.pdf', transparent=True)
    plt.clf()

    #####################     learnt policy in one figure     ####################
    ##############################################################################


##############################################################################
##################     distribution of terminal reward     ###################

# figure parameters
plt.rcParams.update({'font.size': 16, 'figure.figsize': (10,7)})
plt.rc('axes', labelsize=20)

# plot rewards instead of costs
rewards_total = -1 * np.sum(costs, axis=1)

# set a grid for the histogram
grid = np.linspace(-0.2, 0.4, 100)

for idx_method, method in enumerate(rm_labels):
    # plot the histogram for each method
    plt.hist(x=rewards_total[:,idx_method],
            alpha=0.4,
            bins=grid,
            color=colors[idx_method],
            density=True)

plt.legend(rm_labels)
plt.xlabel("Terminal reward")
plt.ylabel("Density")
plt.title("Distribution of the terminal reward")

for idx_method, method in enumerate(rm_labels):
    # plot gaussian KDEs
    kde = gaussian_kde(rewards_total[:,idx_method], bw_method='silverman')
    plt.plot(grid,
            kde(grid),
            color=colors[idx_method],
            linewidth=1.5)

    # plot quantiles of the distributions
    plt.axvline(x=np.quantile(rewards_total[:,idx_method],0.1),
                linestyle='dashed',
                color=colors[idx_method],
                linewidth=1.0)
    plt.axvline(x=np.quantile(rewards_total[:,idx_method],0.9),
                linestyle='dashed',
                color=colors[idx_method],
                linewidth=1.0)

plt.xlim(-0.2,0.4)
plt.tight_layout()
plt.savefig(repo + '/comparison_terminal_cost.pdf', transparent=True)
plt.clf()

##################     distribution of terminal reward     ###################
##############################################################################