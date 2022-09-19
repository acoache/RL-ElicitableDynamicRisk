"""
Environment - Geometric Brownian motions
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
        
        # parameters and spaces
        self.params = params
        self.spaces = {'s1_space' : np.linspace(self.S0[0] * T.exp( (self.mu[0] - 0.5 * self.sigma[0]**2)*self.T + self.sigma[0]*np.sqrt(self.T)*(-4) ),
                                                self.S0[0] * T.exp( (self.mu[0] - 0.5 * self.sigma[0]**2)*self.T + self.sigma[0]*np.sqrt(self.T)*(4) ), 21),
                      's2_space' : np.linspace(self.S0[1] * T.exp( (self.mu[1] - 0.5 * self.sigma[1]**2)*self.T + self.sigma[1]*np.sqrt(self.T)*(-3) ),
                                                self.S0[1] * T.exp( (self.mu[1] - 0.5 * self.sigma[1]**2)*self.T + self.sigma[1]*np.sqrt(self.T)*(3) ), 21),
                      's3_space' : np.linspace(self.S0[2] * T.exp( (self.mu[2] - 0.5 * self.sigma[2]**2)*self.T + self.sigma[2]*np.sqrt(self.T)*(-3) ),
                                                self.S0[2] * T.exp( (self.mu[2] - 0.5 * self.sigma[2]**2)*self.T + self.sigma[2]*np.sqrt(self.T)*(3) ), 21),
                      'action_space' : np.linspace(0.0, 1.0, 21)}
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')


    # initialization of the environment with its true initial state
    def reset(self, Nsims=1):
        x_0 = self.x0 * T.ones(Nsims, device=self.device)

        S_0 = T.zeros(Nsims, len(self.S0), device=self.device)
        S_0[:,:len(self.S0)] = T.tensor(self.S0)

        idx_0 = T.zeros(Nsims, dtype=T.long, device=self.device)
        
        return idx_0, S_0, x_0


    # simulation engine
    def step(self, idx_t, S_t, x_t, pi):
        # allocate space for new prices
        S_tp1 = T.zeros( S_t.shape )
        
        # simulate Brownian motions
        Z = T.distributions.MultivariateNormal(T.zeros(len(self.S0)), self.rho)
        dW = self.sqrt_dt * Z.sample(S_t.shape[:1])
        
        # update time step
        idx_tp1 = idx_t+1
        
        # update risky assets
        S_tp1 = S_t * T.exp( (self.mu - 0.5*self.sigma**2)*self.dt + self.sigma*dW)
                
        # update the portfolio value
        x_tp1 = x_t * T.sum(pi * S_tp1 / S_t, axis=-1)
        
        # compute reward
        reward = x_tp1 - x_t
        
        return idx_tp1, S_tp1, x_tp1, -reward