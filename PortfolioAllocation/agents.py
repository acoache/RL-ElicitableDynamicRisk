"""
Agent for the actor-critic style algorithm -- Portfolio allocation example
Actor: update policy given a value function by policy gradient
Critic: update value function given a policy by consistent scoring functions
"""
# numpy
import numpy as np
from scipy.stats import gaussian_kde
# plotting
import matplotlib.pyplot as plt
# pytorch
import torch as T
# misc
from .utils import cmap, colors
from os.path import exists
from datetime import datetime
from pdb import set_trace

class Agent():
    # constructor
    def __init__(self,
                env, # environment
                risk_measure, # risk measure
                policy, # ANN structure for the policy
                VaR_main, # ANN structure for the value-at-risk
                VaR_target, # copy of the ANN for the value-at-risk
                DiffCVaR_main, # ANN structure for the difference between CVaR and VaR,
                DiffCVaR_target): # copy of the ANN for the difference between CVaR and VaR

        self.env = env # environment
        self.risk_measure = risk_measure # risk measure
        
        self.policy = policy # policy (ACTOR)

        self.VaR_main = VaR_main # VaR (CRITIC)
        self.DiffCVaR_main = DiffCVaR_main # difference between CVaR and VaR (CRITIC)
        
        self.VaR_target = VaR_target # copy of VaR (TARGET)
        self.DiffCVaR_target = DiffCVaR_target # copy of difference between CVaR and VaR (TARGET)
        
        self.device = self.policy.device # PyTorch device
        
        # initialize loss objects
        self.loss_history_pi = [] # keep track of all losses for the policy
        self.loss_history_V = [] # keep track of all losses for the V
        self.loss_trail_pi = 100 # number of epochs for the loss moving average for policy
        self.loss_trail_V = 500 # number of epochs for the loss moving average for V


    # select an action according to the policy ('best' or 'random')
    def select_actions(self,
                        idx_t, # time
                        S_t, # risky asset prices
                        choose): # 'best' | 'random'
        
        # observations as a formatted tensor
        obs_t = T.cat( (idx_t.clone().unsqueeze(-1), 
                        S_t.clone()), -1)
        
        # obtain parameters of the distribution
        actions_mu, actions_sigma = self.policy(obs_t.clone())

        # create action distributions with a Normal distribution
        actions_dist = T.distributions.MultivariateNormal(actions_mu, actions_sigma)
        
        # get action from the policy
        if choose=='random':
            actions_sample = actions_dist.rsample() # random sample from the multivariate normal
        elif choose=='best':
            actions_sample = actions_mu  # mean of the normal
        else:
            raise ValueError("Type of action selection is unknown ('random' or 'best').")

        # transform to simplex
        actions_t = T.exp(actions_sample) / T.sum(T.exp(actions_sample), axis=-1).unsqueeze(-1)

        # get log-probabilities of the action
        log_prob_t = actions_dist.log_prob(actions_sample.detach()).squeeze()
        
        return actions_t, log_prob_t

    
    # simulate trajectories from the policy
    def sim_trajectories(self,
                        Ntrajectories=100, # number of (outer) trajectories
                        choose='random'): # how to choose the actions
        
        # initialize tables for all trajectories
        idx_t = T.zeros((Ntrajectories, len(self.env.t)), dtype=T.long, requires_grad=False, device=self.device)
        S_t = T.zeros((Ntrajectories, len(self.env.t), len(self.env.S0)), dtype=T.float, requires_grad=False, device=self.device)
        x_t = T.zeros((Ntrajectories, len(self.env.t)), dtype=T.float, requires_grad=False, device=self.device)
        action_t = T.zeros((Ntrajectories, len(self.env.S0), len(self.env.t)-1), dtype=T.float, requires_grad=False, device=self.device)
        log_prob_t = T.zeros((Ntrajectories, len(self.env.t)-1), dtype=T.float, requires_grad=False, device=self.device)
        cost_t = T.zeros((Ntrajectories, len(self.env.t)-1), dtype=T.float, requires_grad=False, device=self.device)
        
        # starting (outer) state with multiple random states
        idx_t[:,0], S_t[:,0,:], x_t[:,0] = self.env.reset(Ntrajectories)
        
        # simulate N whole trajectories
        for t_idx in range(len(self.env.t)-1):
            # get actions from the policy
            action_t[:,:,t_idx], log_prob_t[:,t_idx] = \
                            self.select_actions(idx_t[:,t_idx],
                                                S_t[:,t_idx,:],
                                                choose)
            
            # simulate transitions
            idx_t[:,t_idx+1], S_t[:,t_idx+1,:], x_t[:,t_idx+1], cost_t[:,t_idx] = \
                            self.env.step(idx_t[:,t_idx],
                                        S_t[:,t_idx,:],
                                        x_t[:,t_idx].clone(),
                                        action_t[:,:,t_idx])
            
        # store (outer) trajectories in a dictionary
        trajs = {'idx_t' : idx_t, 'S_t' : S_t, 'x_t' : x_t, # states -- time x prices x wealth
                'cost_t' : cost_t, 'action_t' : action_t, 'log_prob_t' : log_prob_t} # costs, actions and log-probs

        return trajs

    # estimate the value function for all time steps (critic)
    def estimate_V(self,
                    Ntrajectories, # number of trajectories
                    Nminibatch=50, # batch size for the update
                    Nepochs=100, # number of epochs
                    replace_target=250, # number of epochs before updating target networks
                    init_lr=None): # manually modify the learning rate

        if Nminibatch >= Ntrajectories:
            raise ValueError("Ntrajectories must be larger than Nminibatch.")
        
        # generate full trajectories from policy
        trajs = self.sim_trajectories(Ntrajectories, choose="random")
        
        # manually change the learning rate
        if init_lr is not None:
            self.VaR_main.optimizer.param_groups[0]['lr'] = init_lr
            self.DiffCVaR_main.optimizer.param_groups[0]['lr'] = init_lr

        for epoch in range(Nepochs):
            # zero grad
            self.VaR_main.zero_grad()
            self.DiffCVaR_main.zero_grad()

            # replace target networks
            if epoch % replace_target == 0:
                self.VaR_target.load_state_dict(self.VaR_main.state_dict())
                self.DiffCVaR_target.load_state_dict(self.DiffCVaR_main.state_dict())

            # sample a batch of states at time t+1
            batch_idx = np.random.choice(Ntrajectories, size=Nminibatch, replace=False)
            cost_batch = trajs["cost_t"][batch_idx, :]
            
            # compute predicted values for VaR and DiffCVaR
            obs_t = T.cat( (trajs["idx_t"][batch_idx, :-1].unsqueeze(-1), 
                            trajs["S_t"][batch_idx, :-1, :]), -1).detach()
            VaR_pred = self.VaR_main(obs_t.clone()).squeeze()
            DiffCVaR_pred = self.DiffCVaR_main(obs_t.clone()).squeeze()
            
            # value function at the next time step
            obs_tp1 = T.cat( (trajs["idx_t"][batch_idx, 1:-1].unsqueeze(-1), 
                            trajs["S_t"][batch_idx, 1:-1, :]), -1)
            VaR_tp1 = self.VaR_target(obs_tp1.clone()).squeeze()
            DiffCVaR_tp1 = self.DiffCVaR_target(obs_tp1.clone()).squeeze()
            
            # costs plus value function for other time steps
            cost_batch[:, :-1] += VaR_tp1 + DiffCVaR_tp1
            
            # calculate the loss function and optimize both ANNs
            consistent_loss = self.risk_measure.consistent_CVaR_loss(VaR_pred,
                                                                    VaR_pred + DiffCVaR_pred,
                                                                    cost_batch.detach())
            consistent_loss.to(self.device).backward()

            # optimize both neural networks
            self.VaR_main.optimizer.step()
            self.DiffCVaR_main.optimizer.step()

            # decay the learning rates
            self.VaR_main.scheduler.step()
            self.DiffCVaR_main.scheduler.step()

            # keep track of the loss
            self.loss_history_V.append(consistent_loss.detach().numpy())
            
            # # uncomment to print progress during the critic procedure
            # if epoch % 100 == 0 or epoch == Nepochs - 1:
            #     print('Estimation of V -- Loss: ', str(np.round( np.mean(self.loss_history_V[-self.loss_trail_V:]) , 3)))


    # update the policy according to a batch of trajectories (actor)
    def update_policy(self,
                        Nminibatch=50, # batch size for the update
                        Nepochs=100, # number of epochs
                        min_lr=1e-7): # minimum learning rate

        for epoch in range(Nepochs):
            # zero grad
            self.policy.zero_grad()
            
            # sample a batch of transitions
            trajs = self.sim_trajectories(Nminibatch, choose='random')

            # get outputs from the ANNs
            obs_t = T.cat( (trajs["idx_t"][:, :-1].unsqueeze(-1), 
                            trajs["S_t"][:, :-1, :]), -1)
            VaR_t = self.VaR_main(obs_t.clone()).squeeze()
            DiffCVaR_t = self.DiffCVaR_main(obs_t.clone()).squeeze()

            # costs plus value function for other time steps
            trajs["cost_t"][:, :-1] += VaR_t[:, 1:] + DiffCVaR_t[:, 1:]
            
            # compute the gradient loss function
            grad_loss = self.risk_measure.get_V_loss(VaR_t.detach(),
                                                    trajs["log_prob_t"],
                                                    trajs["cost_t"].detach())
            grad_loss.to(self.device).backward()

            # optimize the neural network
            self.policy.optimizer.step()

            # decay the learning rate
            if self.policy.optimizer.param_groups[0]["lr"] >= min_lr:
                self.policy.scheduler.step()
            else:
                self.policy.optimizer.param_groups[0]["lr"] = min_lr

            # keep track of the loss
            self.loss_history_pi.append(grad_loss.detach().numpy())

            # # uncomment to print progress during the actor procedure
            # if epoch % 100 == 0 or epoch == Nepochs - 1:
            #     print('Update of pi -- Loss: ', str(np.round( np.mean(self.loss_history_pi[-self.loss_trail_pi:]) ,5)),
            #           ', lr: ', str(np.round( self.policy.optimizer.param_groups[0]['lr'] , 5)))


    # load parameters of the (pre-trained) neural networks
    def load_models(self, repo):
        # verify the repository
        if not exists(repo):
            raise IOError("The specified repository does not exist.")

        self.policy.load_state_dict(T.load(repo + '/policy_model.pt'))
        self.VaR_main.load_state_dict(T.load(repo + '/VaR_model.pt'))
        self.DiffCVaR_main.load_state_dict(T.load(repo + '/DiffCVaR_model.pt'))


    # save parameters of the (trained) neural networks
    def save_models(self, repo):
        # verify the repository
        if not exists(repo):
            raise IOError("The specified repository does not exist.")
            
        T.save(self.policy.state_dict(), repo + '/policy_model.pt')
        T.save(self.VaR_main.state_dict(), repo + '/VaR_model.pt')
        T.save(self.DiffCVaR_main.state_dict(), repo + '/DiffCVaR_model.pt')


    # set training/eval mode for neural networks
    def set_train_mode(self, train=True):
        if train:
            self.VaR_main.train()
            self.DiffCVaR_main.train()
            self.policy.train()
        else:
            self.VaR_main.eval()
            self.DiffCVaR_main.eval()
            self.policy.eval()


    # plot the strategy at any point in the algorithm
    def plot_current_policy(self,
                            repo, # name of the repository for storage
                            Nsimulations=20_000, # number of simulated trajectories for the reward histogram
                            seed=1234): # seed (for replication purposes)
        # verify the repository
        if not exists(repo):
            raise IOError("The specified repository does not exist.")

        # figure parameters
        plt.rcParams.update({'font.size': 16, 'figure.figsize': (10,7)})
        plt.rc('axes', labelsize=20)
        nrows = 2 
        ncols = len(self.env.S0)
        fig, axes = plt.subplots(nrows, ncols, sharey='row', sharex='row')
        gridspec = plt.GridSpec(nrows, ncols)

        # compute 2D histograms for the policy
        hist2dim_policy = T.zeros((len(self.env.spaces["s2_space"]), len(self.env.spaces["s1_space"]), len(self.env.S0)))
        for s2_idx, s2_val in enumerate(self.env.spaces["s2_space"]):
            for s1_idx, s1_val in enumerate(self.env.spaces["s1_space"]):
                # best action according to the policy
                hist2dim_policy[len(self.env.spaces["s2_space"])-s2_idx-1, s1_idx, :], _ = \
                        self.select_actions(self.env.t[0]*T.ones(1, device=self.device),
                                            T.Tensor([s1_val,s2_val,self.env.S0[-1]], device=self.device).unsqueeze(0),
                                            'best')
                                   
        # plot the 2D histogram for each asset
        for idx_asset in range(len(self.env.S0)):
            temp = axes[0,idx_asset].imshow(hist2dim_policy[:,:,idx_asset].detach().numpy(),
                                    interpolation='none',
                                    cmap=cmap,
                                    extent=[np.min(self.env.spaces["s1_space"]),
                                            np.max(self.env.spaces["s1_space"]),
                                            np.min(self.env.spaces["s2_space"]),
                                            np.max(self.env.spaces["s2_space"])],
                                    aspect='auto',
                                    vmin=np.min(self.env.spaces["action_space"]),
                                    vmax=np.max(self.env.spaces["action_space"]))
            axes[0,idx_asset].set_title(r"$\mathbf{{ \pi_{:d} }}$".format(idx_asset+1), fontweight='semibold')
            if idx_asset != 0:
                plt.setp(axes[0,idx_asset].get_yticklabels(), visible=False)
            axes[1,idx_asset].tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            axes[1,idx_asset].set_frame_on(False)
        
        # simulate trajectories under the optimal policy
        T.manual_seed(seed)
        np.random.seed(seed)
        trajs = self.sim_trajectories(Ntrajectories=Nsimulations, choose='best')

        # compute rewards instead of costs
        rewards_total = -1 * T.sum(trajs["cost_t"], axis=1).detach().numpy()
        kde = gaussian_kde(rewards_total, bw_method='silverman')
        
        # plot the distribution of terminal rewards
        grid = np.linspace(-1.0, 1.0, 100)
        PnLhist = fig.add_subplot(gridspec[1,:])
        PnLhist.hist(x=rewards_total, bins=grid, alpha=0.4, color=colors[0], density=True)

        # plot gaussian KDEs
        PnLhist.plot(grid, kde(grid), color=colors[0], linewidth=1.5)

        # plot quantiles of the distributions
        PnLhist.axvline(x=np.quantile(rewards_total,0.1), linestyle='dashed', color=colors[0], linewidth=1.0)
        PnLhist.axvline(x=np.quantile(rewards_total,0.9), linestyle='dashed', color=colors[0], linewidth=1.0)
        PnLhist.set_xlabel("Terminal PnL")
        PnLhist.set_ylabel("Density")
        PnLhist.set_xlim(np.min(grid), np.max(grid))

        # add a colorbar for the 2D histograms
        fig.colorbar(temp, ax=axes.ravel().tolist() + [PnLhist], location='right', shrink=0.8)
        now = datetime.now()
        plt.savefig(repo + '/best_actions-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) + '.png',
                    transparent=False, bbox_inches='tight')
        plt.clf()
        plt.close()


    # plot the value function at any point in the algorithm
    def plot_current_V(self,
                        repo): # name of the repository for storage
        # verify the repository
        if not exists(repo):
            raise IOError("The specified repository does not exist.")

        # figure parameters
        plt.rcParams.update({'font.size': 16, 'figure.figsize': (10,7)})
        plt.rc('axes', labelsize=20)
        fig, axes = plt.subplots(1, 3, sharey='all')

        for t_idx, t_val in enumerate([self.env.t[0], self.env.t[int((len(self.env.t)-1)/2)], self.env.t[-1]]):
            # compute 2D histogram for the value function
            hist2dim_V = T.zeros((len(self.env.spaces["s2_space"]), len(self.env.spaces["s1_space"])))
            for s2_idx, s2_val in enumerate(self.env.spaces["s2_space"]):
                for s1_idx, s1_val in enumerate(self.env.spaces["s1_space"]):
                    obs_t = T.cat( (t_val*T.ones(1, device=self.device).unsqueeze(-1),
                                    T.Tensor([s1_val,s2_val,self.env.S0[-1]], device=self.device).unsqueeze(0) ), -1)
                    hist2dim_V[len(self.env.spaces["s2_space"])-s2_idx-1, s1_idx] = \
                            (self.VaR_main(obs_t.clone()) + self.DiffCVaR_main(obs_t.clone())).squeeze()

            # plot the 2D histogram
            temp = axes[t_idx].imshow(hist2dim_V.detach().numpy(),
                                        interpolation='none',
                                        cmap=cmap,
                                        extent=[np.min(self.env.spaces["s1_space"]),
                                                np.max(self.env.spaces["s1_space"]),
                                                np.min(self.env.spaces["s2_space"]),
                                                np.max(self.env.spaces["s2_space"])],
                                        aspect='auto')
            axes[t_idx].set_title(r"$V$; $t=$" + str(np.round(t_val.numpy(), 3)))
            axes[t_idx].set_xlabel(r"$S_1$")
            if t_idx == 0:
                axes[t_idx].set_ylabel(r"$S_2$")
            else:
                plt.setp(axes[t_idx].get_yticklabels(), visible=False)
            plt.colorbar(temp, ax=axes[t_idx])
            plt.tight_layout()
        
        now = datetime.now()
        plt.savefig(repo + '/value_func-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) + '.png', transparent=False)
        plt.clf()
        plt.close()