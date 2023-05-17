import numpy as np
import pandas as pd

class BasicProblem(features=features_to_use,
    target='ESR',
    target_transform=lambda x: x == 1,
    group='RAC1P',
    preprocess=lambda x: x,
    postprocess=lambda x: np.nan_to_num(x, -1),):
    
    def __init__(self, w = None):
        """ Define instance variables. 
        w -- the weight vector.
        loss_history -- a list of the evolution of the loss over the training period. 
        score_history -- a list of the evolution of the score over the training period.
        """
        self.w = w
        self.loss_history = []
        self.score_history = []
    
    def 
    