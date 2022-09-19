"""
Environment - Orstein-Uhlenbeck dynamics with drift
Portfolio allocation example

"""
# numpy
import numpy as np
# pytorch
import torch as T
# misc
from pdb import set_trace

class Environment():
    # constructor
    def __init__(self, params):
        # interest rate
        self.r0 = params["r0"]
        
        # risky assets
        self.S0 = params["S0"]
        self.x0 = params["x0"]
        self.mu = T.tensor(params["mu"]).float()
        self.sigma = T.tensor(params["sigma"]).float()
        
        # correlation between assets
        self.rho = T.tensor(params["rho"]).float()
        
        # time horizon and periods
        self.T = params["T"]
        self.dt = params["dt"]
        self.sqrt_dt = np.sqrt(self.dt)
        self.t = T.linspace(0, self.T, int(self.T/self.dt+1))
        
        self.kappa = 2.0
        self.eta = self.sigma * np.sqrt((1 - np.exp(-2*self.kappa*self.dt)) / (2*self.kappa))
        self.theta = self.mu*self.T - 0.5*self.sigma**2 * (1 - np.exp(-2*self.kappa*self.T)) / (2*self.kappa)
        
        # parameters and spaces
        self.params = params
        self.spaces = {'s1_space' : np.exp( self.theta[0].numpy() + np.linspace(-3*self.sigma[0]/np.sqrt(2*self.kappa), 
                                                                2*self.sigma[0]/np.sqrt(2*self.kappa), 21) ),
                    's2_space' : np.exp( self.theta[1].numpy() + np.linspace(-3*self.sigma[1]/np.sqrt(2*self.kappa), 
                                                                2*self.sigma[1]/np.sqrt(2*self.kappa), 21) ),
                    's3_space' : np.exp( self.theta[2].numpy() + np.linspace(-3*self.sigma[2]/np.sqrt(2*self.kappa), 
                                                                2*self.sigma[2]/np.sqrt(2*self.kappa), 21) ),
                    'action_space' : np.linspace(0.0, 1.0, 21)}
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')


    # initialization of the environment with its true initial state
    def reset(self, Nsims=1):
        x_0 = self.x0 * T.ones(Nsims, device=self.device)
        
        S_0 = T.zeros(Nsims, len(self.S0), device=self.device)
        S_0[:,:len(self.S0)] = T.exp( T.normal(0.0, (self.sigma/np.sqrt(2*self.kappa)).repeat(Nsims,1)) )

        idx_0 = T.zeros(Nsims, dtype=T.long, device=self.device)
        
        return idx_0, S_0, x_0


    # simulation engine
    def step(self, idx_t, S_t, x_t, pi):        
        # allocate space for next prices
        S_tp1 = T.zeros( S_t.shape )
        
        # simulate Brownian motions
        Z = T.distributions.MultivariateNormal(T.zeros(len(self.S0)), self.rho)
        dW = Z.sample(S_t.shape[:1])
        
        # update time step
        idx_tp1 = idx_t+1

        # get log-price
        eta_t = self.sigma.repeat(S_t.shape[0],1) * (T.sqrt((1 - T.exp(-2*self.kappa*self.dt*idx_t)) / (2*self.kappa))).unsqueeze(-1)
        h_t = self.mu.repeat(S_t.shape[0],1)*self.dt*idx_t.unsqueeze(-1) - 0.5*eta_t**2
        logS_t = T.log(S_t)-h_t

        # mean-reversion of the log-price around zero
        logS_tp1 = (logS_t)*np.exp(-self.kappa*self.dt) + self.eta*dW

        # update risky assets
        eta_tp1 = self.sigma.repeat(S_t.shape[0],1) * (T.sqrt((1 - T.exp(-2*self.kappa*self.dt*idx_tp1)) / (2*self.kappa))).unsqueeze(-1)
        h_tp1 = self.mu.repeat(S_t.shape[0],1)*self.dt*idx_tp1.unsqueeze(-1) - 0.5*eta_tp1**2
        S_tp1 = T.exp(logS_tp1+h_tp1)
        
        # update the portfolio value
        x_tp1 = x_t * (T.sum(pi * S_tp1 / S_t, axis=-1))
        
        # compute reward
        reward = x_tp1 - x_t
        
        return idx_tp1, S_tp1, x_tp1, -reward