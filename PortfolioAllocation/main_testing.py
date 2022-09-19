"""
Main testing script
3 plots: learnt policies at different times, terminal reward, wealth evolution
Portfolio allocation example
"""
# numpy
import numpy as np
# plotting
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# pytorch
import torch as T
# personal modules
from hyperparams import init_params, print_params
from envs_meanrevert import Environment
from utils import directory, cmap, colors
from models import PolicyANN, VaRANN, DiffCVaRANN
from risk_measures import RiskMeasure
from agents import Agent
# misc
from pdb import set_trace

"""
Parameters
"""

rm_labels = ['CVaR0.05', 'CVaR0.25', 'CVaR0.5', 'CVaR0.7'] # risk measures used
seed = 4321 # set seed for replication purposes
Nsimulations = 30_000 # number of simulations following the optimal strategy

_, envParams, algoParams = init_params() # parameters for the model and algorithm
repo = "set1" # repo name

"""
End of Parameters
"""

# print all parameters for reproducibility purposes
print('\n*** Name of the repository: ', repo, ' ***\n')
print_params(envParams, algoParams)

directory(repo) # create the directory

env = Environment(envParams) # create the environment

costs = np.zeros((Nsimulations, len(env.t)-1, len(rm_labels))) # matrix to store all testing trajectories
x_paths = np.zeros((Nsimulations, len(env.t), len(rm_labels))) # matrix to store all testing trajectories

for idx_method, method in enumerate(rm_labels):
    # print progress
    print('\n*** Method = ', method, ' ***')

    # create ANNs for policy, VaR and DiffCVaR
    policy = PolicyANN(input_size=1+len(env.S0),
                        hidden_size=algoParams["hidden_pi"],
                        n_layers=algoParams["layers_pi"],
                        env=env,
                        learn_rate=algoParams["lr_pi"])
    VaR_main = VaRANN(input_size=1+len(env.S0),
                    hidden_size=algoParams["hidden_V"],
                    n_layers=algoParams["layers_V"],
                    env=env,
                    learn_rate=algoParams["lr_V"])
    DiffCVaR_main = DiffCVaRANN(input_size=1+len(env.S0),
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

    actor_critic.set_train_mode(train=False)
    
    # set seed for reproducibility purposes
    T.manual_seed(seed)
    np.random.seed(seed)

    # optimal strategy -- ANN
    trajs = actor_critic.sim_trajectories(Ntrajectories=Nsimulations, choose='random')
    costs[:,:,idx_method] = trajs["cost_t"].detach().numpy()
    x_paths[:,:,idx_method] = trajs["x_t"].detach().numpy()

    ###########################     END OF TESTING     ###########################
    ##############################################################################

    ##############################################################################
    #############     learnt policy for different S_3 values     #################
    
    # figure parameters
    plt.rcParams.update({'font.size': 16, 'figure.figsize': (10,7)})
    plt.rc('axes', labelsize=20)
    nrows = len(env.S0) 
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, sharey='all', sharex='all')
    grid = plt.GridSpec(nrows, ncols)

    # values of the price of asset 3
    s3_spaces = [env.spaces["s3_space"][int((len(env.spaces["s3_space"])-1)/4)], \
                env.spaces["s3_space"][int((len(env.spaces["s3_space"])-1)/2)], \
                env.spaces["s3_space"][-int((len(env.spaces["s3_space"])-1)/4)]]
    # fixed value for the time
    t_fixed = env.t[0]

    for s3_idx, s3_val in enumerate(s3_spaces):            
        # compute 2D histograms for all assets
        hist2dim_policy = T.zeros((len(env.spaces["s2_space"]), len(env.spaces["s1_space"]), len(env.S0)))
        for s2_idx, s2_val in enumerate(env.spaces["s2_space"]):
            for s1_idx, s1_val in enumerate(env.spaces["s1_space"]):
                hist2dim_policy[len(env.spaces["s2_space"])-s2_idx-1, s1_idx, :], _ = \
                        actor_critic.select_actions(t_fixed*T.ones(1, device=actor_critic.device),
                                                    T.Tensor([s1_val,s2_val,s3_val], device=actor_critic.device).unsqueeze(0),
                                                    'best')
        # optimal policy for each asset                     
        for idx_asset in range(len(env.S0)):
            if s3_idx != 0:
                plt.setp(axes[idx_asset,s3_idx].get_yticklabels(), visible=False)

            temp = axes[idx_asset,s3_idx].imshow(hist2dim_policy[:,:,idx_asset].detach().numpy(),
                                    interpolation='none',
                                    cmap=cmap,
                                    extent=[np.min(env.spaces["s1_space"]),
                                            np.max(env.spaces["s1_space"]),
                                            np.min(env.spaces["s2_space"]),
                                            np.max(env.spaces["s2_space"])],
                                    aspect='auto',
                                    vmin=np.min(env.spaces["action_space"]),
                                    vmax=np.max(env.spaces["action_space"]))

    # titles for columns
    columns = []
    for idx in range(len(s3_spaces)):
        columns.append(fig.add_subplot(grid[:,idx], frameon=False))
        columns[idx].set_title(r"$\mathbf{S_3=}$" + str(np.round(s3_spaces[idx], 2)) + '\n', fontweight='semibold')
        columns[idx].axis('off')

    # labels for all plots
    xyaxis=fig.add_subplot(grid[:,:], frameon=False)
    xyaxis.set_xlabel(r"$S_1$")
    xyaxis.set_ylabel(r"$S_2$")
    xyaxis.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    # titles for rows
    rows = []
    for idx in range(len(env.S0)):
        twin = fig.add_subplot(grid[idx,:], frameon=False)
        rows.append(twin.twinx())
        twin.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        rows[idx].tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        rows[idx].set_ylabel(r"$\mathbf{{ \pi_{:d} }}$".format(idx+1), fontweight='semibold', rotation=0)
        rows[idx].set_frame_on(False)
    
    # add a common colorbar for the figure
    fig.colorbar(temp, \
            ax=axes.ravel().tolist() + [xyaxis] + columns + rows, \
            orientation='horizontal', shrink=0.8)
    plt.savefig(repo + '/best_actions-' + method + '-varyS3.pdf', transparent=True, bbox_inches='tight')
    plt.clf()
    plt.close()

    #############     learnt policy for different S_3 values     #################
    ##############################################################################

    ##############################################################################
    ################     learnt policy for different times     ###################
    
    # figure parameters
    plt.rcParams.update({'font.size': 16, 'figure.figsize': (10,7)})
    plt.rc('axes', labelsize=20)
    nrows = len(env.S0) 
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, sharey='all', sharex='all')
    grid = plt.GridSpec(nrows, ncols)

    # values of the time
    t_spaces = [env.t[0], env.t[int((len(env.t)-1)/2)], env.t[-1]]
    # fixed value for the price of asset 3
    s3_fixed = env.S0[-1]

    for t_idx, t_val in enumerate(t_spaces):            
        # initialize 2D histogram
        hist2dim_policy = T.zeros((len(env.spaces["s2_space"]), len(env.spaces["s1_space"]), len(env.S0)))
        for s2_idx, s2_val in enumerate(env.spaces["s2_space"]):
            for s1_idx, s1_val in enumerate(env.spaces["s1_space"]):
                # best action according to the policy
                hist2dim_policy[len(env.spaces["s2_space"])-s2_idx-1, s1_idx, :], _ = \
                        actor_critic.select_actions(t_val*T.ones(1, device=actor_critic.device),
                                                    T.Tensor([s1_val,s2_val,s3_fixed], device=actor_critic.device).unsqueeze(0),
                                                    'best')
                                   
        # allocate the subplot
        for idx_asset in range(len(env.S0)):
            if t_idx != 0:
                plt.setp(axes[idx_asset,t_idx].get_yticklabels(), visible=False)

            # plot the 2D histogram
            temp = axes[idx_asset,t_idx].imshow(hist2dim_policy[:,:,idx_asset].detach().numpy(),
                                    interpolation='none',
                                    cmap=cmap,
                                    extent=[np.min(env.spaces["s1_space"]),
                                            np.max(env.spaces["s1_space"]),
                                            np.min(env.spaces["s2_space"]),
                                            np.max(env.spaces["s2_space"])],
                                    aspect='auto',
                                    vmin=np.min(env.spaces["action_space"]),
                                    vmax=np.max(env.spaces["action_space"]))

    # titles for columns
    columns = []
    for t_idx, t_val in enumerate(t_spaces):
        columns.append(fig.add_subplot(grid[:,t_idx], frameon=False))
        columns[t_idx].set_title(r"$\mathbf{{ t={:.3f} }}$".format(t_val.numpy()) + "\n", fontweight='semibold')
        columns[t_idx].axis('off')

    # labels for all plots
    xyaxis=fig.add_subplot(grid[:,:], frameon=False)
    xyaxis.set_xlabel(r"$S_1$")
    xyaxis.set_ylabel(r"$S_2$")
    xyaxis.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    # titles for rows
    rows = []
    for idx in range(len(env.S0)):
        twin = fig.add_subplot(grid[idx,:], frameon=False)
        rows.append(twin.twinx())
        twin.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        rows[idx].tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        rows[idx].set_ylabel(r"$\mathbf{{ \pi_{:d} }}$".format(idx+1), fontweight='semibold', rotation=0)
        rows[idx].set_frame_on(False)

    # add a common colorbar for the figure
    fig.colorbar(temp, \
                ax=axes.ravel().tolist() + [xyaxis] + columns + rows, \
                orientation='horizontal', shrink=0.8)
    plt.savefig(repo + '/best_actions-' + method + '-varytime.pdf', transparent=True, bbox_inches='tight')
    plt.clf()
    plt.close()

    ################     learnt policy for different times     ###################
    ##############################################################################


##############################################################################
##############     evolution of PnL with learnt policies     #################

# get rewards instead of costs
rewards_total = x_paths[:,-1,:] - env.x0

# figure parameters
plt.rcParams.update({'font.size': 16, 'figure.figsize': (10,7)})
plt.rc('axes', labelsize=20)
fig = plt.figure()
gs = fig.add_gridspec(1,3)
ax_main = fig.add_subplot(gs[:-1])
ax_hist = fig.add_subplot(gs[-1], sharey=ax_main)

# plot the quantiles of the rewards through time
for idx_method, method in enumerate(rm_labels):
    ax_main.fill_between(x=env.t.numpy(),
                    y1=np.quantile(x_paths[:,:,idx_method].transpose() - env.x0, 0.0, axis=1),
                    y2=np.quantile(x_paths[:,:,idx_method].transpose() - env.x0, 1.0, axis=1),
                    facecolor=colors[idx_method]+(0.4,), edgecolor=colors[idx_method]+(1.0,),
                    linewidth=1.0)

ax_main.legend(rm_labels, loc="upper left")

ax_main.set_xlabel(r"$t$")
ax_main.set_ylabel(r"P&L")
ax_main.set_xlim(env.t[0], env.t[-1])

for idx_method, method in enumerate(rm_labels):
    ax_main.plot(env.t.numpy(),
                np.quantile(x_paths[:,:,idx_method].transpose() - env.x0, (0.1,0.9), axis=1).transpose(),
                color=colors[idx_method],
                linewidth=1.5, linestyle='dashed')

# plot histogram of terminal rewards and quantiles
grid = np.linspace(-0.3, 0.9, 100)
for idx_method, method in enumerate(rm_labels):
    ax_hist.hist(x=rewards_total[:,idx_method], bins=grid,
            alpha=0.4, color=colors[idx_method], density=True, orientation='horizontal')
    ax_hist.axhline(np.quantile(rewards_total[:,idx_method],0.1),
                    linestyle='dashed',
                    color=colors[idx_method],
                    linewidth=1.5)
    ax_hist.axhline(np.quantile(rewards_total[:,idx_method],0.9),
                    linestyle='dashed',
                    color=colors[idx_method],
                    linewidth=1.5)

# plot gaussian KDEs
for idx_method, method in enumerate(rm_labels):
    kde = gaussian_kde(rewards_total[:,idx_method], bw_method='silverman')
    ax_hist.plot(kde(grid), grid, color=colors[idx_method], linewidth=1.5)

ax_hist.set_xlabel("Density")
ax_hist.set_ylim(np.min(grid), np.max(grid))
plt.setp(ax_hist.get_yticklabels(), visible=False)

plt.tight_layout()
plt.savefig(repo + '/best_PnLevo.pdf', transparent=True, bbox_inches='tight')
plt.clf()
plt.close()

##############     evolution of PnL with learnt policies     #################
##############################################################################