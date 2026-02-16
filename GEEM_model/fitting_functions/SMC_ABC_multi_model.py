import matplotlib.pyplot as plt
from multiprocessing import Manager, Process, Lock
import time
import copy
import pandas as pd
import seaborn as sns
import platform
from GEEM_model.fitting_functions.universal_functions import *


def generate_initial_particles_worker_multi_model(models,
                                      smc_abc_params,
                                      eval_locs,
                                      t_eval,
                                      y_t,
                                      lock,
                                      accepted_particles,
                                      rmse_tracker,
                                      worker_id,):
    # First we generate the initial set of params
    local_particles = []
    while True:
        with lock:
            if len(accepted_particles) >= smc_abc_params.number_particles:
                break
        raw_particle = np.random.uniform(low=smc_abc_params.lower_bounds, high=smc_abc_params.upper_bounds)
        # # Assuming that the prior type is constant
        # if smc_abc_params.prior_type == 'log-uniform':
        #     particle = 10 ** raw_particle
        # else:
        #     particle = raw_particle
        particle = raw_particle.copy()
        for i in range(models[0].num_parameters_to_fit):
            if smc_abc_params.prior_type[i] == 'log-uniform':
                particle[i] = 10 ** particle[i]
        # print(particle)
        # start = time.time()
        model_outputs = []
        for model in models:
            model.set_constants(copy.deepcopy(particle))
            sol = solve_ivp(model.fitting_derivatives, model.time_range, model.get_initial_conditions().ravel(),
                            t_eval=t_eval, method='LSODA')
            # sol = solve_with_timeout(models, t_eval, timeout=90)
            # if sol is None:
            #     continue  # skip this particle
            # elapsed = time.time() - start
            # print(f"[Worker {worker_id}] solve_ivp took {elapsed:.2f} seconds", flush=True)
            model._reset_concentrations()
            model_outputs.append(sol.y[eval_locs, :])
        model_outputs = np.concatenate(model_outputs, axis=0)
        rmse = MSE_loss_calc(y_t, model_outputs, norm=smc_abc_params.rsme_normalization)
        # rmse = RSE_loss(y_t[eval_locs, :], sol.y[eval_locs, :], norm=smc_abc_params.rsme_normalization)
        if rmse < smc_abc_params.tolerances[0]:
            with lock:
                if len(accepted_particles) < smc_abc_params.number_particles:
                    accepted_particles.append(raw_particle)
                    rmse_tracker.append(rmse)
                    if len(accepted_particles)%100 == 0:
                        # print(f'The rates at t0 are: {models.fitting_derivatives(0, models.get_initial_conditions())}')
                        print(len(accepted_particles))
                    # print(len(accepted_particles))


def generate_particles_worker_multi_model(models,
                              smc_abc_params,
                              eval_locs,
                              t_eval,
                              y_t,
                              ss,
                              tol,
                              lock,
                              init_particles,
                              accepted_particles,
                              rmse_tracker,
                              attempt_counter,
                              worker_id,):
    """
        Worker process: repeatedly sample and evaluate a candidate particle until
        the global accepted_particles list has reached the required number.
    """
    # print(f'Worker {worker_id} started in generate particles worker')
    local_counter = 0
    if smc_abc_params.sampling == 'multivar_gaussian':
        # mvmu = np.zeros(models.num_parameters_to_fit).ravel()
        mvmu = np.mean(init_particles, axis=0).ravel()
    while True:
        # print(f'Worker {worker_id} hit while true')
        with lock:
            if len(accepted_particles) >= smc_abc_params.number_particles:
                # print(f'[Worker {worker_id}] has reached the number of accepted particles')
                break
            # attempt_counter.value += 1
            # if attempt_counter.value >= smc_abc_params.max_unsuccessful_itters:
            #     # print(f'[Worker {worker_id}] has reached the max attempts')
            #     break
        if local_counter >= smc_abc_params.max_unsuccessful_itters:
            break
        local_counter += 1
        # print(f'Worker {worker_id} started generating particles')
        # Rejection sampling and perturbation
        perturbed_particle = generate_perturbed_particle(models[0], smc_abc_params, init_particles, mvmu, ss)
        # print(f'Worker {worker_id} has generated a particle')
        # print('We have an acceoted particle')
        # Model simulation
        model_outputs = []
        for model in models:
            model.set_constants(copy.deepcopy(perturbed_particle))
            sol = solve_ivp(model.fitting_derivatives, model.time_range, model.get_initial_conditions().ravel(),
                            t_eval=t_eval, method='LSODA')
            # sol = solve_with_timeout(models, t_eval, timeout=90)
            # if sol is None:
            #     continue  # skip this particle
            # elapsed = time.time() - start
            # print(f"[Worker {worker_id}] solve_ivp took {elapsed:.2f} seconds", flush=True)
            model._reset_concentrations()
            model_outputs.append(sol.y[eval_locs, :])
        model_outputs = np.concatenate(model_outputs, axis=0)
        rmse = MSE_loss_calc(y_t, model_outputs, norm=smc_abc_params.rsme_normalization)

        # Accept the particle if the RMSE is below the tolerance
        if rmse < tol:
            with lock:
                if len(accepted_particles) < smc_abc_params.number_particles:
                    perturbed_particle[smc_abc_params.log_priors] = np.log10(perturbed_particle[smc_abc_params.log_priors])
                    accepted_particles.append(perturbed_particle)
                    rmse_tracker.append(rmse)
                    # if len(accepted_particles)%100 == 0:
                    #     print(len(accepted_particles))
                    # print(len(accepted_particles))


def parallel_smc_abc_multi_model(models, smc_abc_params, eval_locs, t_eval, y_t, save_fig=False, save_dir=''):
    system_type = platform.system().lower()
    print(system_type)
    if system_type == "linux":
        mp.set_start_method("spawn", force=True)
    else:
        # Windows and macOS: safer to use spawn
        mp.set_start_method("spawn", force=True)
    manager = Manager()
    accepted_particles_list=manager.list()
    attempt_counter = manager.Value('i', 0)
    rmse_tracker_list=manager.list()
    lock = Lock()

    num_workers = smc_abc_params.num_workers

    models_to_pass = []
    for _ in range(num_workers):
        models_array = []
        for model in models:
            models_array.append(copy.deepcopy(model))
        models_to_pass.append(models_array)

    processes = []
    tic = time.time()
    print(f'Starting with initial particle generation with tolerance {smc_abc_params.tolerances[0]}')
    for worker_num in range(num_workers):
        p = Process(target=generate_initial_particles_worker_multi_model, args=(
            models_to_pass[worker_num], smc_abc_params, eval_locs, t_eval, y_t, lock,
            accepted_particles_list, rmse_tracker_list, worker_num))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    toc = time.time()
    print(f'The initial particle generation took {toc - tic} seconds')

    accepted_particles = np.array(accepted_particles_list)
    rmse_tracker = np.array(rmse_tracker_list)
    # df = pd.DataFrame(accepted_particles, columns=models[0].parameter_names)
    # sns.pairplot(df, diag_kind="hist", plot_kws={"s": 10, "alpha": 0.5})
    # plt.title(f'Pairplot of {smc_abc_params.number_particles} accepted particles tolerance {smc_abc_params.tolerances[0]}')
    # if save_fig:
    #     plt.savefig(save_dir + f'Pairplot_{smc_abc_params.number_particles}_particles_tol_{smc_abc_params.tolerances[0]}.png')
    #     plt.close()
    # else:
    #     plt.show()

    # print('The first pairplot was made')
    # print(f'The number of particles is {len(accepted_particles_list)}')

    # Loop through the tolerance values
    for tol in smc_abc_params.tolerances[1:]:
        init_particles = accepted_particles  # New init particles based on accepted ones

        # Check results and calc the new covariance matrix
        init_normal_scale = init_particles.copy()
        init_normal_scale[:,smc_abc_params.log_priors] = 10**init_normal_scale[:,smc_abc_params.log_priors]

        # print(f"Mean of particles found (normal scale): {np.mean(init_normal_scale, axis=0)}")
        ss = (np.max(init_normal_scale, axis=0) - np.min(init_normal_scale, axis=0)) / smc_abc_params.ss_scaling
        # print(f"Range of Parameters (normal scale): {ss * smc_abc_params.ss_scaling}")
        print(f"Mean Variance of particles found (mvg scale): {np.mean(np.var(init_particles, axis=0))}")
        print(f"Mean RMSE: {np.mean(rmse_tracker)}")
        print(f'Lowest RMSE found {np.min(rmse_tracker)}\n')

        if smc_abc_params.sampling == 'multivar_gaussian':
            # print('We do have a multivar gaussian')
            ss = np.cov(init_particles, rowvar=False) * (2.38**2)/models[0].num_parameters_to_fit

        print(f"Starting with tolerance: {tol}")

        # Reset attempt_counter, accepted particles and RMSE tracker for the new tolerance round
        manager = Manager()
        accepted_particles_list = manager.list()
        attempt_counter = manager.Value('i', 0)
        rmse_tracker_list = manager.list()
        lock = Lock()

        # Start workers for the current tolerance
        processes = []
        for worker_num in range(num_workers):
            # print(f'starting worker {worker_num}')
            p = Process(target=generate_particles_worker_multi_model, args=(models_to_pass[worker_num],
                              smc_abc_params,
                              eval_locs,
                              t_eval,
                              y_t,
                              ss,
                              tol,
                              lock,
                              init_particles,
                              accepted_particles_list,
                              rmse_tracker_list,
                              attempt_counter,
                              worker_num))
            p.start()
            processes.append(p)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Convert results back to numpy array
        accepted_particles = np.array(accepted_particles_list)
        rmse_tracker = np.array(rmse_tracker_list)

        # Check if we've reached the target number of accepted particles
        if len(accepted_particles_list) < smc_abc_params.number_particles:
            print(f"Aborting at tolerance {tol}. Found {len(accepted_particles)} particles.")
            if len(rmse_tracker_list) < 0:
                print(f"The lowest RMSE found in the last unsuccessful run was: {np.min(rmse_tracker, axis=0)}")
            print(f"Final accepted particles shape: {accepted_particles.shape}")

            # df = pd.DataFrame(init_particles, columns=models[0].parameter_names)
            # sns.pairplot(df, diag_kind="hist", plot_kws={"s": 10, "alpha": 0.5})
            # plt.title(f'Pairplot of {smc_abc_params.number_particles} accepted particles tolerance {tol}')
            # if save_fig:
            #     plt.savefig(save_dir + f'Pairplot_{smc_abc_params.number_particles}_p_tol_{tol}.png')
            #     plt.close()
            # else:
            #     plt.show()
            return init_particles, accepted_particles, True

    df = pd.DataFrame(accepted_particles, columns=models[0].parameter_names)
    sns.pairplot(df, diag_kind="hist", plot_kws={"s": 10, "alpha": 0.5})
    plt.title(f'Pairplot of {smc_abc_params.number_particles} accepted particles tolerance {tol}')
    if save_fig:
        plt.savefig(save_dir + f'Pairplot_{smc_abc_params.number_particles}_p_tol_{tol}.png')
        plt.close()
    else:
        plt.show()

    return init_particles, accepted_particles, False