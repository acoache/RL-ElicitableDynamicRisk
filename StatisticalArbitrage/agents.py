"""
Agent for the actor-critic style algorithm -- Statistical arbitrage example
Actor: update policy given a value function by policy gradient
Critic: update value function given a policy by consistent scoring functions
"""
# numpy
import numpy as np
# plotting
import matplotlib.pyplot as plt
# pytorch
import torch as T
# misc
from utils import cmap
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

        # assign objects to the actor_critic instance
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
        self.loss_trail_V = 500 # number of epochs for the loss moving average for V
        self.loss_print_V = 500 # number of epochs before printing the loss for V


    # select an action according to the policy ('best' or 'random')
    def select_actions(self,
                        time_t, # time
                        s_t, # price of the asset
                        q_t, # inventory
                        choose, # 'best' | 'random'
                        seed=None):
        
        # freeze the set of random normal variables
        if seed is not None:
            T.manual_seed(seed)
            np.random.seed(seed)

        # observations as a formatted tensor
        obs_t = T.stack((time_t.clone(), \
                        s_t.clone(), \
                        q_t.clone()), -1)
        
        # obtain parameters of the distribution
        actions_mu, actions_scale = self.policy(obs_t.clone())

        # create action distributions with a Normal distribution
        actions_dist = T.distributions.Normal(actions_mu, actions_scale)
     
        # get action from the policy
        if choose=='random':
            actions_sample = actions_dist.rsample()  # random sample from the Normal
        elif choose=='best':
            actions_sample = actions_mu  # mode of the Normal
        else:
            raise ValueError("Type of action selection is unknown ('random' or 'best').")
        
        # obtain lower and upper bounds for the actions
        min_action = T.maximum(T.tensor(self.env.params["min_u"], device=self.device), self.env.params["min_q"] - q_t)
        max_action = T.minimum(T.tensor(self.env.params["max_u"], device=self.device), self.env.params["max_q"] - q_t)

        # get actions on the appropriate range
        u_t = min_action + (max_action-min_action) *  (1.0 / (1.0 + T.exp(-actions_sample.squeeze(-1))))

        # get log-probabilities of the action
        log_prob_t = actions_dist.log_prob(actions_sample.detach()).squeeze()
        
        return u_t, log_prob_t

    
    # simulate trajectories from the policy
    def sim_trajectories(self,
                        Ntrajectories=100, # number of (outer) trajectories
                        choose='random', # how to choose the actions
                        seed=None): # random seed
        
        # freeze the seed
        if seed is not None:
            T.manual_seed(seed)
            np.random.seed(seed)
        
        # initialize tables for all trajectories
        time_t = T.zeros((Ntrajectories, self.env.params["Ndt"]+1), dtype=T.float, requires_grad=False, device=self.device)
        s_t = T.zeros((Ntrajectories, self.env.params["Ndt"]+1), dtype=T.float, requires_grad=False, device=self.device)
        q_t = T.zeros((Ntrajectories, self.env.params["Ndt"]+1), dtype=T.float, requires_grad=False, device=self.device)
        action_t = T.zeros((Ntrajectories, self.env.params["Ndt"]), dtype=T.float, requires_grad=False, device=self.device)
        log_prob_t = T.zeros((Ntrajectories, self.env.params["Ndt"]), dtype=T.float, requires_grad=False, device=self.device)
        cost_t = T.zeros((Ntrajectories, self.env.params["Ndt"]), dtype=T.float, requires_grad=False, device=self.device)
        
        # starting (outer) state with multiple random states
        time_t[:,0], s_t[:,0], q_t[:,0] = self.env.random_reset(Ntrajectories)

        # simulate N whole trajectories
        for t_idx in self.env.spaces["t_space"]:
            # get actions from the policy
            action_t[:,t_idx], log_prob_t[:,t_idx] = \
                            self.select_actions(time_t[:,t_idx],
                                                s_t[:,t_idx],
                                                q_t[:,t_idx],
                                                choose)

            # simulate transitions
            time_t[:,t_idx+1], s_t[:,t_idx+1], q_t[:,t_idx+1], cost_t[:,t_idx] = \
                            self.env.step(time_t[:,t_idx],
                                        s_t[:,t_idx],
                                        q_t[:,t_idx],
                                        action_t[:,t_idx])

        # store (outer) trajectories in a dictionary
        trajs = {'time_t' : time_t, 's_t' : s_t, 'q_t' : q_t, # states -- time x price x inventory
                'cost_t' : cost_t, 'action_t' : action_t, 'log_prob_t' : log_prob_t} # costs, actions and log-probs

        return trajs

    # estimate the value function for all time steps (critic)
    def estimate_V(self,
                    Ntrajectories, # number of trajectories
                    Nminibatch=50, # batch size for the update
                    Nepochs=100, # number of epochs
                    replace_target=250, # number of epochs before updating target networks
                    init_lr=None, # manually modify the learning rate
                    rng_seed=None): # random seed

        if Nminibatch >= Ntrajectories:
            raise ValueError("Ntrajectories must be larger than Nminibatch.")

        # print progress
        print('--Estimation of V--')

        # set ANNs in training mode
        self.VaR_main.train()
        self.DiffCVaR_main.train()
        
        # generate full trajectories from policy
        trajs = self.sim_trajectories(Ntrajectories, choose="random", seed=rng_seed)
        
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
            obs_t = T.stack((trajs["time_t"][batch_idx, :-1],
                            trajs["s_t"][batch_idx, :-1],
                            trajs["q_t"][batch_idx, :-1]), -1).detach()
            VaR_pred = self.VaR_main(obs_t.clone()).squeeze()
            DiffCVaR_pred = self.DiffCVaR_main(obs_t.clone()).squeeze()

            # value function at the next time step
            obs_tp1 = T.stack((trajs["time_t"][batch_idx, 1:-1],
                                trajs["s_t"][batch_idx, 1:-1],
                                trajs["q_t"][batch_idx, 1:-1]), -1)
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
            
            # print progress
            if epoch % self.loss_print_V == 0 or epoch == Nepochs - 1:
                print('   Epoch = ',
                        str(epoch),
                        ', Loss: ',
                        str(np.round( np.mean(self.loss_history_V[-self.loss_trail_V:]) , 3)),
                        ', Learning rate: ',
                        str(np.round( self.VaR_main.optimizer.param_groups[0]['lr'] , 5)))
        
        # set ANNs in evaluation mode
        self.VaR_main.eval()
        self.DiffCVaR_main.eval()


    # update the policy according to a batch of trajectories (actor)
    def update_policy(self,
                        Nminibatch=50, # batch size for the update
                        Nepochs=100, # number of epochs
                        rng_seed=None): # random seed
        # print progress
        print('--Update of pi--')
        
        # set the policy in training mode
        self.policy.train()
        
        for epoch in range(Nepochs):
            # zero grad
            self.policy.zero_grad()
            
            # sample a batch of transitions
            trajs = self.sim_trajectories(Nminibatch, choose='random', seed=rng_seed)

            # get outputs from the ANNs
            obs_t = T.stack((trajs["time_t"][:, :-1],
                            trajs["s_t"][:, :-1],
                            trajs["q_t"][:, :-1]), -1)
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
            if self.policy.optimizer.param_groups[0]["lr"] >= 5e-4:
                self.policy.scheduler.step()
            else:
                self.policy.optimizer.param_groups[0]["lr"] = 5e-4

            # keep track of the loss
            self.loss_history_pi.append(grad_loss.detach().numpy())

            # print progress
            if epoch % Nepochs == 0 or epoch == Nepochs - 1:
                print('   Epoch = ',
                      str(epoch) ,
                      ', Loss: ',
                      str(np.round( np.mean(self.loss_history_pi[-Nepochs:]) ,3)),
                      ', Learning rate: ',
                      str(np.round( self.policy.optimizer.param_groups[0]['lr'] , 5)))

        # set the policy in evaluation mode
        self.policy.eval()


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


    # plot the strategy at any point in the algorithm
    def plot_current_policy(self, repo):
        # verify the repository
        if not exists(repo):
            raise IOError("The specified repository does not exist.")

        # format the figures
        plt.rcParams.update({'font.size': 16,
                            'figure.figsize': (10,4)})
        plt.rc('axes', labelsize=20)

        for t_idx, t_val in enumerate([self.env.spaces["t_space"][0], int(self.env.spaces["t_space"][-1]/2.0), self.env.spaces["t_space"][-1]]):
            # allocate the subplot
            plt.subplot(1, 3, t_idx+1)
            
            # initialize 2D histogram
            hist2dim_policy = np.zeros([len(self.env.spaces["s_space"]), len(self.env.spaces["q_space"])])
            for s_idx, s_val in enumerate(self.env.spaces["s_space"]):
                for q_idx, q_val in enumerate(self.env.spaces["q_space"]):
                    # best action according to the policy
                    hist2dim_policy[len(self.env.spaces["s_space"])-s_idx-1, q_idx], _ = \
                            self.select_actions(t_val*T.ones(1, device=self.device),
                                                s_val*T.ones(1, device=self.device),
                                                q_val*T.ones(1, device=self.device),
                                                'best')

            # plot the 2D histogram
            plt.imshow(hist2dim_policy,
                    interpolation='none',
                    cmap=cmap,
                    extent=[np.min(self.env.spaces["q_space"]),
                            np.max(self.env.spaces["q_space"]),
                            np.min(self.env.spaces["s_space"]),
                            np.max(self.env.spaces["s_space"])],
                    aspect='auto',
                    vmin=self.env.params["min_u"],
                    vmax=self.env.params["max_u"])
            
            plt.title('Learned; Period:' + str(t_val))
            plt.xlabel("Inventory")
            plt.ylabel("Price")
            plt.colorbar()
            plt.tight_layout()
        
        now = datetime.now()
        plt.savefig(repo + '/best_actions-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) + '.png', transparent=False)
        plt.clf()


    # plot the value function at any point in the algorithm
    def plot_current_V(self, repo):
        # verify the repository
        if not exists(repo):
            raise IOError("The specified repository does not exist.")

        # format the figures
        plt.rcParams.update({'font.size': 16,
                            'figure.figsize': (10,4)})
        plt.rc('axes', labelsize=20)

        for t_idx, t_val in enumerate([self.env.spaces["t_space"][0], int(self.env.spaces["t_space"][-1]/2.0), self.env.spaces["t_space"][-1]]):
            # allocate the subplot
            plt.subplot(1, 3, t_idx+1)
            
            # initialize 2D histogram
            hist2dim_V = np.zeros([len(self.env.spaces["s_space"]), len(self.env.spaces["q_space"])])
            for s_idx, s_val in enumerate(self.env.spaces["s_space"]):
                for q_idx, q_val in enumerate(self.env.spaces["q_space"]):
                    # best action according to the policy
                    obs_t = T.stack((t_val*T.ones(1, device=self.device), \
                                    s_val*T.ones(1, device=self.device), \
                                    q_val*T.ones(1, device=self.device)), -1)
                    hist2dim_V[len(self.env.spaces["s_space"])-s_idx-1, q_idx] = \
                            (self.VaR_main(obs_t.clone()) + self.DiffCVaR_main(obs_t.clone())).squeeze().detach().numpy()

            # plot the 2D histogram
            plt.imshow(hist2dim_V,
                    interpolation='none',
                    cmap=cmap,
                    extent=[np.min(self.env.spaces["q_space"]),
                            np.max(self.env.spaces["q_space"]),
                            np.min(self.env.spaces["s_space"]),
                            np.max(self.env.spaces["s_space"])],
                    aspect='auto')
            
            plt.title('V; Period:' + str(t_val))
            plt.xlabel("Inventory")
            plt.ylabel("Price")
            plt.colorbar()
            plt.tight_layout()
        
        now = datetime.now()
        plt.savefig(repo + '/value_func-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) + '.png', transparent=False)
        plt.clf()