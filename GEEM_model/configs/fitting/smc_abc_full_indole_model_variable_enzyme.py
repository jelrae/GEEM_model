import numpy as np
from GEEM_model.SMC_ABC_Params.SMC_ABC_Params import SMC_ABC_Params
from GEEM_model.configs.fitting.bounds_prior_generator import bounds_prior_generator
import multiprocessing


def make_smc_params(
    # Could also maybe just pass the args here to contain the num particles, ss and such
    num_particles,
    model_conditions,
    params_to_fit,
):
    tolerances = np.array([
        1.0, 0.5, 0.25, 0.175, 0.125, 0.0625,
        0.05, 0.045, 0.04, 0.035,
        0.03125, 0.03, 0.0275, 0.025,
        0.0225, 0.0212
    ])

    bounds, priors = bounds_prior_generator(params_to_fit)

    return SMC_ABC_Params(
        bounds=bounds,
        tolerances=tolerances,
        number_particles=num_particles,
        max_unsuccessful_itters=50 * num_particles,
        ss_scaling=100,
        prior_type=priors,
        num_workers=multiprocessing.cpu_count(),
        sampling='multivar_gaussian',
        num_conditions=len(model_conditions),
        model_conditions=model_conditions,
    )