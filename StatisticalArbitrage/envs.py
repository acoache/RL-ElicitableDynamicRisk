"""
Environment
Statistical arbitrage example

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
        # parameters and spaces
        self.params = params
        self.spaces = {'t_space' : np.arange(params["Ndt"]),
                      's_space' : np.linspace(params["theta"]-4*params["sigma"]/np.sqrt(2*params["kappa"]), 
                                  params["theta"]+4*params["sigma"]/np.sqrt(2*params["kappa"]), 51),
                      'q_space' : np.linspace(params["min_q"], params["max_q"], 51),
                      'u_space' : np.linspace(params["min_u"], params["max_u"], 21)}
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')


    # initialization of the environment with its true initial state
    def reset(self, Nsims=1):
        time_0 = T.zeros(Nsims, device=self.device)
        s0 = T.normal(self.params["theta"],
                        self.params["sigma"]/np.sqrt(2*self.params["kappa"]),
                        size=(Nsims,),
                        device=self.device)
        s0 = T.minimum(T.maximum(s0, min(self.spaces["s_space"])*T.ones(1)), max(self.spaces["s_space"])*T.ones(1))
        q0 = T.zeros(Nsims, device=self.device)

        return time_0, s0, q0


    # initialization of the environment with multiple random states
    def random_reset(self, Nsims=1):
        time_0 = T.zeros(Nsims, device=self.device)
        s0 = T.normal(self.params["theta"],
                        3*self.params["sigma"]/np.sqrt(2*self.params["kappa"]),
                        size=(Nsims,),
                        device=self.device)
        s0 = T.minimum(T.maximum(s0, min(self.spaces["s_space"])*T.ones(1)), max(self.spaces["s_space"])*T.ones(1))
        q0 = self.params["min_q"] + (self.params["max_q"]-self.params["min_q"])*T.rand(Nsims, device=self.device)

        return time_0, s0, q0
    

    # simulation engine
    def step(self, time_t, s_t, q_t, action_t):
        # input: time, price of asset, inventory, action
        # return: next time, next price, next inventory, cost
        
        # time modification -- step forward
        time_tp1 = time_t + 1

        # price modification -- OU process
        sizes = q_t.shape
        dt = self.params["T"]/self.params["Ndt"]
        sqrt_dt = np.sqrt(dt)
        eta = self.params["sigma"] * \
                np.sqrt((1 - np.exp(-2*self.params["kappa"]*dt)) / (2*self.params["kappa"]))
        
        s_tp1 = self.params["theta"] + \
                (s_t-self.params["theta"]) * np.exp(-self.params["kappa"]*dt) + \
                eta * T.randn(sizes, device=self.device)

        # inventory modification -- add the trade to current inventory
        q_tp1 = q_t + action_t
        
        # reward -- profit with transaction costs
        reward_t = - s_t*action_t - self.params["phi"]*action_t**2 \
                    + (time_t == self.spaces["t_space"][-1])*(q_tp1*s_tp1 - self.params["psi"]*q_tp1**2)
        
        return time_tp1, s_tp1, q_tp1, -reward_t