import matplotlib.pyplot as plt
from multiprocessing import Manager, Process, Lock
import time
import copy
import pandas as pd
import seaborn as sns
from GEEM_model.fitting_functions.universal_functions import *
import platform


def generate_initial_particles_worker_multi_conds(model,
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
        for i in range(model.num_parameters_to_fit):
            if smc_abc_params.prior_type[i] == 'log-uniform':
                particle[i] = 10 ** particle[i]

        model_outputs = []
        sol_good = True
        for i in range(smc_abc_params.number_conditions):
            init_meta_conds, alternative_concentrations, mrna_concentrations_eq_dict = smc_abc_params.model_conditions[i]
            model._change_model_conditions(init_meta_conds, alternative_concentrations, mrna_concentrations_eq_dict)
            model.set_constants(copy.deepcopy(particle))
            kds = model.degradation_rates_metabolites[1:]
            if np.any(kds <= 1e-7) or np.any(0.1 <= kds):
                sol_good = False
                break
            sol = solve_ivp(model.fitting_derivatives, model.time_range, model.get_initial_conditions().ravel(),
                            t_eval=t_eval, method='LSODA')
            model._reset_concentrations()
            if not sol.success:
                sol_good = False
                break
            model_outputs.append(sol.y[eval_locs, :])
        if sol_good:
            model_outputs = np.concatenate(model_outputs, axis=0)
            rmse = MSE_loss_calc(y_t, model_outputs, norm=smc_abc_params.rsme_normalization)
            # rmse = RSE_loss(y_t[eval_locs, :], sol.y[eval_locs, :], norm=smc_abc_params.rsme_normalization)
            if rmse < smc_abc_params.tolerances[0]:
                with lock:
                    if len(accepted_particles) < smc_abc_params.number_particles:
                        accepted_particles.append(raw_particle)
                        rmse_tracker.append(rmse)
                        # print(len(accepted_particles))
                        if len(accepted_particles)%100 == 0:
                            # print(f'The rates at t0 are: {models.fitting_derivatives(0, models.get_initial_conditions())}')
                            print(len(accepted_particles))
                            # print(models.degradation_rates_metabolites[1:])
                    # print(len(accepted_particles))


def generate_initial_particles_multi_conds(model,
                              smc_abc_params,
                              eval_locs,
                              t_eval,
                              y_t,):
    # First we generate the initial set of params
    rmse_tracker=[]
    accepted_particles = []
    while True:
        raw_particle = np.random.uniform(low=smc_abc_params.lower_bounds, high=smc_abc_params.upper_bounds)
        # # Assuming that the prior type is constant
        # if smc_abc_params.prior_type == 'log-uniform':
        #     particle = 10 ** raw_particle
        # else:
        #     particle = raw_particle
        particle = raw_particle.copy()
        for i in range(model.num_parameters_to_fit):
            if smc_abc_params.prior_type[i] == 'log-uniform':
                particle[i] = 10 ** particle[i]
        model_outputs = []
        for i in range(smc_abc_params.number_conditions):
            init_meta_conds, alternative_concentrations, mrna_concentrations_eq_dict = smc_abc_params.model_conditions[
                i]
            model._change_model_conditions(init_meta_conds, alternative_concentrations, mrna_concentrations_eq_dict)
            model.set_constants(copy.deepcopy(particle))
            sol = solve_ivp(model.fitting_derivatives, model.time_range, model.get_initial_conditions().ravel(),
                            t_eval=t_eval, method='LSODA')
            model._reset_concentrations()
            model_outputs.append(sol.y[eval_locs, :])
        try:
            model_outputs = np.concatenate(model_outputs, axis=0)
        except:
            print(len(model_outputs[0]))
            print(len(model_outputs[1]))
        rmse = MSE_loss_calc(y_t, model_outputs, norm=smc_abc_params.rsme_normalization)
        if rmse < smc_abc_params.tolerances[0]:
            if len(accepted_particles) < smc_abc_params.number_particles:
                accepted_particles.append(raw_particle)
                rmse_tracker.append(rmse)
                if len(accepted_particles)%100 == 0:
                    print(len(accepted_particles))
                # print(len(accepted_particles))
                if len(accepted_particles) >= smc_abc_params.number_particles:
                    return accepted_particles, rmse_tracker


def generate_particles_worker_multi_conds(model,
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
        perturbed_particle = generate_perturbed_particle(model, smc_abc_params, init_particles, mvmu, ss)
        sol_good = True
        model_outputs = []
        for i in range(smc_abc_params.number_conditions):
            init_meta_conds, alternative_concentrations, mrna_concentrations_eq_dict = smc_abc_params.model_conditions[i]
            model._change_model_conditions(init_meta_conds, alternative_concentrations, mrna_concentrations_eq_dict)
            model.set_constants(copy.deepcopy(perturbed_particle))
            kds = model.degradation_rates_metabolites[1:]
            if np.any(kds <= 1e-7) or np.any(0.1 <= kds):
                sol_good = False
                break
            sol = solve_ivp(model.fitting_derivatives, model.time_range, model.get_initial_conditions().ravel(),
                            t_eval=t_eval, method='LSODA')
            model._reset_concentrations()
            if not sol.success:
                sol_good = False
                break
            model_outputs.append(sol.y[eval_locs, :])
        if sol_good:
            model_outputs = np.concatenate(model_outputs, axis=0)
            rmse = MSE_loss_calc(y_t, model_outputs, norm=smc_abc_params.rsme_normalization)
            # Compute RMSE
            # rmse = MSE_loss(y_t[eval_locs, :], sol.y[eval_locs, :], norm=smc_abc_params.rsme_normalization)

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


def generate_particles_multi_conds(model,
                              smc_abc_params,
                              eval_locs,
                              t_eval,
                              y_t,
                              ss,
                              tol,
                              init_particles,):
    """
        Worker process: repeatedly sample and evaluate a candidate particle until
        the global accepted_particles list has reached the required number.
    """
    # print(f'Worker {worker_id} started in generate particles worker')
    accepted_particles=[]
    rmse_tracker=[]
    local_counter = 0
    if smc_abc_params.sampling == 'multivar_gaussian':
        # mvmu = np.zeros(models.num_parameters_to_fit).ravel()
        mvmu = np.mean(init_particles, axis=0).ravel()
    while True:
        if local_counter >= smc_abc_params.max_unsuccessful_itters:
            break
        local_counter += 1
        # print(f'Worker {worker_id} started generating particles')
        # Rejection sampling and perturbation
        perturbed_particle = generate_perturbed_particle(model, smc_abc_params, init_particles, mvmu, ss)
        model_outputs = []
        for i in range(smc_abc_params.number_conditions):
            init_meta_conds, alternative_concentrations, mrna_concentrations_eq_dict = smc_abc_params.model_conditions[i]
            model._change_model_conditions(init_meta_conds, alternative_concentrations, mrna_concentrations_eq_dict)
            model.set_constants(copy.deepcopy(perturbed_particle))
            sol = solve_ivp(model.fitting_derivatives, model.time_range, model.get_initial_conditions().ravel(),
                            t_eval=t_eval, method='LSODA')
            model._reset_concentrations()
            model_outputs.append(sol.y[eval_locs, :])
        model_outputs = np.concatenate(model_outputs, axis=0)
        rmse = MSE_loss_calc(y_t, model_outputs, norm=smc_abc_params.rsme_normalization)
        # Compute RMSE
        # rmse = MSE_loss(y_t[eval_locs, :], sol.y[eval_locs, :], norm=smc_abc_params.rsme_normalization)

        # Accept the particle if the RMSE is below the tolerance
        if rmse < tol:
            if len(accepted_particles) < smc_abc_params.number_particles:
                perturbed_particle[smc_abc_params.log_priors] = np.log10(perturbed_particle[smc_abc_params.log_priors])
                accepted_particles.append(perturbed_particle)
                rmse_tracker.append(rmse)
            if len(accepted_particles) >= smc_abc_params.number_particles:
                return accepted_particles, rmse_tracker


def parallel_smc_abc_multi_cond(model, smc_abc_params, eval_locs, t_eval, y_t, save_fig=False, save_dir=''):
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

    tic = time.time()
    num_workers = smc_abc_params.num_workers
    if num_workers > 1:
        print('Mutli Thread Option')
        models = []
        for _ in range(num_workers):
            model_copy = copy.deepcopy(model)
            models.append(model_copy)

        processes = []
        print(f'Starting with initial particle generation with tolerance {smc_abc_params.tolerances[0]}')
        for worker_num in range(num_workers):
            p = Process(target=generate_initial_particles_worker_multi_conds, args=(
                models[worker_num], smc_abc_params, eval_locs, t_eval, y_t, lock,
                accepted_particles_list, rmse_tracker_list, worker_num))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    else:
        print('Single Thread Option')
        accepted_particles_list, rmse_tracker_list = generate_initial_particles_multi_conds(model, smc_abc_params,
                                                                                  eval_locs, t_eval, y_t)

    toc = time.time()
    print(f'The initial particle generation took {toc - tic} seconds')

    accepted_particles = np.array(accepted_particles_list)
    rmse_tracker = np.array(rmse_tracker_list)
    # df = pd.DataFrame(accepted_particles, columns=models.parameter_names)
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
            ss = np.cov(init_particles, rowvar=False) * (2.38**2)/model.num_parameters_to_fit

        print(f"Starting with tolerance: {tol}")
        if num_workers > 1:
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
                p = Process(target=generate_particles_worker_multi_conds, args=(models[worker_num],
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
        else:
            accepted_particles_list, rsme_tracker_list = generate_particles_multi_conds(model,
                              smc_abc_params,
                              eval_locs,
                              t_eval,
                              y_t,
                              ss,
                              tol,
                              init_particles)

        # Convert results back to numpy array
        accepted_particles = np.array(accepted_particles_list)
        rmse_tracker = np.array(rmse_tracker_list)

        # Check if we've reached the target number of accepted particles
        if len(accepted_particles_list) < smc_abc_params.number_particles:
            print(f"Aborting at tolerance {tol}. Found {len(accepted_particles)} particles.")
            if len(rmse_tracker_list) < 0:
                print(f"The lowest RMSE found in the last unsuccessful run was: {np.min(rmse_tracker, axis=0)}")
            print(f"Final accepted particles shape: {accepted_particles.shape}")

            # df = pd.DataFrame(init_particles, columns=models.parameter_names)
            # sns.pairplot(df, diag_kind="hist", plot_kws={"s": 10, "alpha": 0.5})
            # plt.title(f'Pairplot of {smc_abc_params.number_particles} accepted particles tolerance {tol}')
            # if save_fig:
            #     plt.savefig(save_dir + f'Pairplot_{smc_abc_params.number_particles}_p_tol_{tol}.png')
            #     plt.close()
            # else:
            #     plt.show()
            return init_particles, accepted_particles, True

    # df = pd.DataFrame(accepted_particles, columns=models.parameter_names)
    # sns.pairplot(df, diag_kind="hist", plot_kws={"s": 10, "alpha": 0.5})
    # plt.title(f'Pairplot of {smc_abc_params.number_particles} accepted particles tolerance {tol}')
    # if save_fig:
    #     plt.savefig(save_dir + f'Pairplot_{smc_abc_params.number_particles}_p_tol_{tol}.png')
    #     plt.close()
    # else:
    #     plt.show()

    return init_particles, accepted_particles, False


def generate_particles_worker(model,
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
                              worker_id, ):
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
        perturbed_particle = generate_perturbed_particle(model, smc_abc_params, init_particles, mvmu, ss)
        # print(f'Worker {worker_id} has generated a particle')
        # print('We have an acceoted particle')
        # Model simulation
        model.set_constants(copy.deepcopy(perturbed_particle))
        # print(f'Worker {worker_id} starting the fitting_functions')
        sol = solve_ivp(model.fitting_derivatives, model.time_range, model.get_initial_conditions().ravel(),
                        t_eval=t_eval, method='LSODA')
        # print(f'Worker {worker_id} ended the fitting_functions')
        # sol = solve_with_timeout(models, t_eval, timeout=90)
        # if sol is None:
        #     # print(f'[Worker {worker_id}] In the generate subsequent particles step we had an issue', flush=True)
        #     continue  # skip this particle
        model._reset_concentrations()

        # Compute RMSE
        rmse = MSE_loss_calc(y_t[eval_locs, :], sol.y[eval_locs, :], norm=smc_abc_params.rsme_normalization)

        # Accept the particle if the RMSE is below the tolerance
        if rmse < tol:
            with lock:
                if len(accepted_particles) < smc_abc_params.number_particles:
                    perturbed_particle[smc_abc_params.log_priors] = np.log10(
                        perturbed_particle[smc_abc_params.log_priors])
                    accepted_particles.append(perturbed_particle)
                    rmse_tracker.append(rmse)
                    # if len(accepted_particles)%100 == 0:
                    #     print(len(accepted_particles))
                    # print(len(accepted_particles))


def resample_parallel_smc_abc(model, smc_abc_params, eval_locs, t_eval, y_t, pdf, tol, save_fig=False, save_dir=''):
    system_type = platform.system().lower()
    print(system_type)
    if system_type == "linux":
        mp.set_start_method("spawn", force=True)
    else:
        # Windows and macOS: safer to use spawn
        mp.set_start_method("spawn", force=True)

    num_workers = smc_abc_params.num_workers
    models = []
    for _ in range(num_workers):
        model_copy = copy.deepcopy(model)
        models.append(model_copy)

    accepted_particles = pdf.dataset.T

    init_particles = accepted_particles  # New init particles based on accepted ones

    # Check results and calc the new covariance matrix
    init_normal_scale = init_particles.copy()
    init_normal_scale[:, smc_abc_params.log_priors] = 10 ** init_normal_scale[:, smc_abc_params.log_priors]

    # print(f"Mean of particles found (normal scale): {np.mean(init_normal_scale, axis=0)}")
    ss = (np.max(init_normal_scale, axis=0) - np.min(init_normal_scale, axis=0)) / smc_abc_params.ss_scaling

    if smc_abc_params.sampling == 'multivar_gaussian':
        # print('We do have a multivar gaussian')
        ss = np.cov(init_particles, rowvar=False) * (2.38 ** 2) / model.num_parameters_to_fit

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
        p = Process(target=generate_particles_worker, args=(models[worker_num],
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

        df = pd.DataFrame(init_particles, columns=model.parameter_names)
        sns.pairplot(df, diag_kind="hist", plot_kws={"s": 10, "alpha": 0.5})
        plt.title(f'Pairplot of {smc_abc_params.number_particles} accepted particles tolerance {tol}')
        if save_fig:
            plt.savefig(save_dir + f'Pairplot_{smc_abc_params.number_particles}_p_tol_{tol}.png')
            plt.close()
        else:
            plt.show()
        return init_particles, accepted_particles, True

    return init_particles, accepted_particles, False