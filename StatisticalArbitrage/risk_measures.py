"""
Risk measures
Implementation of the CVaR with consistent scoring function

"""
# pytorch
import torch as T
# misc
from pdb import set_trace

class RiskMeasure():
    # constructor
    def __init__(self, alpha=0.9, C=25.0):
        if (alpha < 0) or (alpha >= 1):
            raise ValueError("alpha needs to be in [0,1).")

        self.type = 'CVaR' # name of the risk measure
        self.alpha = alpha # threshold of the CVaR
        self.C = C # bound for the random costs


    # define the (strictly) consistent scoring function for the CVaR
    def consistent_CVaR_loss(self,
                            var, # estimation of the VaR
                            cvar, # estimation of the CVaR
                            rvs): # realisations of the random variable
        scores = T.log((cvar+self.C) / (rvs+self.C)) \
                - cvar / (cvar+self.C) \
                + (1/((cvar+self.C)*(1-self.alpha))) * (var*(1*(rvs<=var)-self.alpha) + rvs*(rvs>var))

        return T.mean(scores)


    # calculate the gradient for the policy gradient step
    def get_V_loss(self,
                    saddle_points, # estimation of the saddle-points
                    logprob, # log-probability (with gradients)
                    rvs): # realisations of the random variable
        loss = T.mean( (rvs - saddle_points) * logprob * (1/(1-self.alpha))*(rvs>saddle_points) )
        
        return loss