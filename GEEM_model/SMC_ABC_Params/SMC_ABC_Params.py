import numpy as np
import copy

class SMC_ABC_Params:
    def __init__(self,
                 bounds,
                 tolerances,
                 number_particles=500,
                 max_unsuccessful_itters=1000,
                 ss_scaling=10,
                 prior_type='uniform',
                 rsme_normalization=None,
                 num_workers=10,
                 sampling='gaussian',
                 num_conditions=1,
                 model_conditions=None,
                 ):
        self.number_particles = number_particles
        self.bounds = bounds
        self.prior_type = prior_type
        self.lower_bounds = copy.deepcopy(bounds.lb)
        self.upper_bounds = copy.deepcopy(bounds.ub)
        self.log_priors = []
        for i in range(len(prior_type)):
            if self.prior_type[i] == 'log-uniform':
                self.log_priors.append(i)
                self.lower_bounds[i] = np.log10(self.lower_bounds[i])
                self.upper_bounds[i] = np.log10(self.upper_bounds[i])
        self.tolerances = tolerances
        self.max_unsuccessful_itters = max_unsuccessful_itters
        self.ss_scaling = ss_scaling
        self.rsme_normalization = rsme_normalization
        self.num_workers = num_workers
        self.sampling=sampling
        self.number_conditions = num_conditions
        self.model_conditions = model_conditions
