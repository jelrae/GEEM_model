import numpy as np
import multiprocessing as mp
from scipy.integrate import solve_ivp


def perturb_particle(y, mu, cov, distribution='multivar_gaussian', log_priors=None):
    rng = np.random.RandomState()
    # perturbation = rng.multivariate_normal(mu, cov)
    if distribution == 'multivar_gaussian':
        perturbation = rng.multivariate_normal(y, cov)
    elif distribution == 'multivar_log_gaussian':
        perturbation = rng.multivariate_normal(np.log10(y), cov)
        perturbation = 10 ** perturbation
    else:
        print('Not a recognized dist, using multivar_gaussian')
        perturbation = rng.multivariate_normal(y, cov)

    # # Attempt for if we have things saved in log form for the covariance matrix?
    if log_priors:
        perturbation[log_priors] = 10**perturbation[log_priors]

    return perturbation


def generate_perturbed_particle(model, smc_abc_params, init_particles, mvmu, ss):
    rng = np.random.RandomState()
    # print(f'Testing the randomness {rng.choice(range(0, 100))}')
    # Rejection sampling and perturbation
    perturbed_particle = np.ones(model.num_parameters_to_fit).ravel() * -1
    while ((perturbed_particle <= smc_abc_params.bounds.lb).any() or
           (perturbed_particle >= smc_abc_params.bounds.ub).any()):
        sample = init_particles[rng.choice(smc_abc_params.number_particles, 1)].ravel()
        if smc_abc_params.sampling == 'gaussian':
            if smc_abc_params.prior_type == 'log-uniform':
                perturbed_particle = add_noise(sample, mu=0, sigma2=ss)
            else:
                raw_particle = add_noise(sample, mu=0, sigma2=ss)
                perturbed_particle = raw_particle
            # perturbed_particle = add_noise(sample, mu=0, sigma2=ss, distribution=smc_abc_params.prior_type)
        elif smc_abc_params.sampling == 'multivar_gaussian':
            perturbed_particle = perturb_particle(sample, mu=mvmu, cov=ss, distribution=smc_abc_params.sampling,
                                                  log_priors=smc_abc_params.log_priors)
    return perturbed_particle


def add_noise(y, mu=0, sigma2=0.16, distribution='uniform'):
    """
    simple function which adds noise to a vector based on a single mu and sigma squared value.
    The noise which is added is from the same distribution for all of the y's so if we need a more complex one later,
    we need to write it
    :param y:
    :param mu:
    :param sigma2:
    :return:
    a noisy version of y
    """

    rng = np.random.RandomState()
    if distribution == 'uniform':
        noise = rng.normal(mu, sigma2, y.shape)
    elif distribution == 'log-uniform':
        noise = rng.lognormal(mu, sigma2, y.shape)
    else:
        print("distribution not recognized using normal distribution")
        noise = rng.normal(mu, sigma2, y.shape)

    return y + noise


def MSE_loss_calc(y_true, y_pred, ss=0.01, weights=None, norm='range'):
    if norm == 'range':
        # Use (max - min) across timepoints for each output
        scale = y_true.max(axis=1, keepdims=True) - y_true.min(axis=1, keepdims=True)
        mean_val = np.mean(np.abs(y_true), axis=1, keepdims=True)
        scale = np.where(scale == 0, mean_val, scale)
        scale = np.where(scale == 0, 1, scale)  # still guard against zero mean
        y_true = y_true / scale
        y_pred = y_pred / scale

    elif isinstance(norm, np.ndarray):
        scale = norm[:, None]
        y_true = y_true / scale
        y_pred = y_pred / scale

    if weights is None:
        weights = np.ones(y_true.shape)
    # rmse = np.sum(np.sqrt((weights * (y_true - y_pred) ** 2))) / (len(y_true) * ss)
    # Removed the normalization based on the error, since that will cause issues when the ss = 0, as in testing
    # print(f'the size of the true values is {y_true.size}')
    rmse = np.sum(np.sqrt((weights * (y_true - y_pred) ** 2))) / y_true.size
    # print(f'the shape of the inputs after the sqrt is {np.sqrt((weights * (y_true - y_pred) ** 2)).shape}')
    # print(f'the RMSE is {rmse}')
    return rmse
    # return np.log10(rmse + 1e-12)


def run_solve_ivp(model, t_eval):
    return solve_ivp(model.fitting_derivatives, model.time_range, model.get_initial_conditions().ravel(),
                     t_eval=t_eval)


def solve_with_timeout(model, t_eval, timeout=90):
    with mp.get_context("spawn").Pool(1) as pool:
        result = pool.apply_async(run_solve_ivp, (model, t_eval))
        try:
            return result.get(timeout=timeout)
        except mp.TimeoutError:
            pool.terminate()
            return None  # or raise