import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd


def plot_set_formating(SMALL_SIZE=12, MEDIUM_SIZE=15, BIGGER_SIZE=20, BIGGEST_SIZE=27):
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGEST_SIZE)  # fontsize of the figure title


def plotting_with_model(unfit_model, synth_model, t_sim, t_eval):
    plot_set_formating()
    t_range_eval = np.arange(t_sim[0], t_sim[1], 1)
    subs_dict = unfit_model.metabolite_dictionary


    output = solve_ivp(unfit_model.fitting_derivatives, t_sim, unfit_model.get_initial_conditions(), t_eval=t_range_eval)
    real_data = solve_ivp(synth_model.fitting_derivatives, t_sim, synth_model.get_initial_conditions(),
                       t_eval=t_eval)
    solutions = output.y
    input_data = real_data.y

    check_solutions = solve_ivp(unfit_model.fitting_derivatives, t_sim, unfit_model.get_initial_conditions(),
                       t_eval=t_eval)

    print('The difference between the fitted models and the synthetic models is {0}'.format(input_data-check_solutions.y))

    experiment_data = pd.DataFrame(input_data.T, index=t_eval, columns=list(subs_dict.values())[1:])
    sol_df = pd.DataFrame(solutions.T, index=t_range_eval, columns=list(subs_dict.values())[1:])

    plt.figure(figsize=(12, 8))
    experiment_data = pd.melt(experiment_data, ignore_index=False, var_name='Metabolite', value_name='Concentration')
    experiment_data = experiment_data.reset_index().rename(columns={'index': 'Time'})
    synth_df = pd.melt(sol_df, ignore_index=False, var_name='Metabolite', value_name='Concentration')
    synth_df = synth_df.reset_index().rename(columns={'index': 'Time'})

    palette = {
        '3-indolylmthyl GLS glucobrassicin': 'b',
        '1-hydroxy-3-indolylmethyl GSL': 'y',
        '1-methoxy-3-indolylmethyl GSL': 'g',
        '4-hydroxy-3-indolylmethyl GSL': 'm',
        '4-methoxy-3-indolylmethyl GSL': 'r'
    }

    sns.scatterplot(data=experiment_data, x='Time', y='Concentration', hue='Metabolite', palette=palette, legend=False)
    palette = {
        '3-indolylmthyl GLS glucobrassicin': 'b',
        '1-hydroxy-3-indolylmethyl GSL': 'y',
        '1-methoxy-3-indolylmethyl GSL': 'g',
        '4-hydroxy-3-indolylmethyl GSL': 'm',
        '4-methoxy-3-indolylmethyl GSL': 'r'
    }
    sns.lineplot(data=synth_df, x='Time', y='Concentration', hue='Metabolite', palette=palette)
    base_dir = '../../results/model_structure/'
    figure_dir = base_dir + 'figures/abc_fit_results/'
    create_dirs([figure_dir])
    filename = 'abc_right_half_equation_fit_attempt_sim'
    plt.savefig(figure_dir + filename + '_fig.png')
    # plt.show()


def plotting_with_data(unfit_model, yt, t_sim, t_eval, fig_dir, filename, save_figs=False):
    plot_set_formating()
    t_range_eval = np.arange(t_sim[0], t_sim[1], 1)
    subs_dict = unfit_model.metabolite_dictionary


    output = solve_ivp(unfit_model.fitting_derivatives, t_sim, unfit_model.get_initial_conditions().ravel(),
                       t_eval=t_range_eval)
    solutions = output.y
    unfit_model._reset_concentrations()
    check_solutions = solve_ivp(unfit_model.fitting_derivatives, t_sim, unfit_model.get_initial_conditions(),
                                t_eval=t_eval)
    discrete_solutions = check_solutions.y
    if unfit_model.variable_enzymes:
        solutions = solutions[:unfit_model.number_metabolites-1, :]
        discrete_solutions = discrete_solutions[:unfit_model.number_metabolites-1, :]
    print('The difference between the fitted models and the synthetic models is {0}'.format(yt-discrete_solutions))

    experiment_data = pd.DataFrame(yt.T, index=t_eval, columns=list(subs_dict.values())[1:])
    sol_df = pd.DataFrame(solutions.T, index=t_range_eval, columns=list(subs_dict.values())[1:])

    plt.figure(figsize=(12, 8))
    experiment_data = pd.melt(experiment_data, ignore_index=False, var_name='Metabolite', value_name='Concentration')
    experiment_data = experiment_data.reset_index().rename(columns={'index': 'Time'})
    synth_df = pd.melt(sol_df, ignore_index=False, var_name='Metabolite', value_name='Concentration')
    synth_df = synth_df.reset_index().rename(columns={'index': 'Time'})

    palette = {
        '3-indolylmthyl GLS glucobrassicin': 'b',
        '1-hydroxy-3-indolylmethyl GSL': 'y',
        '1-methoxy-3-indolylmethyl GSL': 'g',
        '4-hydroxy-3-indolylmethyl GSL': 'm',
        '4-methoxy-3-indolylmethyl GSL': 'r'
    }

    sns.scatterplot(data=experiment_data, x='Time', y='Concentration', hue='Metabolite', palette=palette, legend=False)
    palette = {
        '3-indolylmthyl GLS glucobrassicin': 'b',
        '1-hydroxy-3-indolylmethyl GSL': 'y',
        '1-methoxy-3-indolylmethyl GSL': 'g',
        '4-hydroxy-3-indolylmethyl GSL': 'm',
        '4-methoxy-3-indolylmethyl GSL': 'r'
    }
    sns.lineplot(data=synth_df, x='Time', y='Concentration', hue='Metabolite', palette=palette)
    if save_figs:
        create_dirs([fig_dir])
        plt.savefig(fig_dir+filename+'_fig.png')
        plt.close()
    else:
        plt.show()


def distribution_plotting_with_data(unfit_model, yt, t_sim, t_eval, pdf, smc_params, fig_dir=None, filename=None,
                                    num_sims=1000, exp_conds='Control', save_figs=False, sample_df=None):
    plot_set_formating()
    if exp_conds == 'control':
        exp_conds='Mock'
    elif exp_conds == 'meJA':
        exp_conds='JA'
    else:
        exp_conds = exp_conds
    t_range_eval = np.arange(t_sim[0], t_sim[1], 1)
    subs_dict = unfit_model.metabolite_dictionary
    experiment_data = pd.DataFrame(yt.T, index=t_eval, columns=list(subs_dict.values())[1:])
    plt.figure(figsize=(12, 8))
    experiment_data = pd.melt(experiment_data, ignore_index=False, var_name='Metabolite', value_name='Concentration')
    experiment_data = experiment_data.reset_index().rename(columns={'index': 'Time'})
    palette = {
        '3-indolylmthyl GLS glucobrassicin': 'b',
        '1-hydroxy-3-indolylmethyl GSL': 'y',
        '1-methoxy-3-indolylmethyl GSL': 'g',
        '4-hydroxy-3-indolylmethyl GSL': 'm',
        '4-methoxy-3-indolylmethyl GSL': 'r'
    }
    sns.set_style("whitegrid")
    if sample_df is None:
        sns.scatterplot(data=experiment_data, x='Time', y='Concentration', hue='Metabolite', palette=palette,
                        legend=False)
    else:
        # df_melt = sample_df.melt(
        #     id_vars="Time",
        #     var_name="Metabolite",
        #     value_name="Concentration"
        # )
        sns.scatterplot(sample_df, x='Time', y='Concentration', hue='Glucosinolate', palette=palette,
                        legend=False)
        # If I want to plot the errorbar plot instead of a scatterplot
        # summary_df = (
        #     sample_df
        #     .groupby(["Time", "Glucosinolate"])
        #     .agg(
        #         Mean=("Concentration", "mean"),
        #         Std=("Concentration", "std"),
        #         N=("Concentration", "count")  # optional: number of replicates
        #     )
        #     .reset_index()
        # )
        # for g, df in summary_df.groupby("Glucosinolate"):
        #     plt.errorbar(df["Time"], df["Mean"], yerr=df["Std"], fmt="o", label=g, color=palette[g])
        # sns.lineplot(data=sample_df, x='Time', y='Concentration', hue='Glucosinolate', palette=palette,
        #                    errorbar="sd", estimator="mean", marker="o", linestyle="none", legend=False)
    palette = {
        '3-indolylmthyl GLS glucobrassicin': 'b',
        '1-hydroxy-3-indolylmethyl GSL': 'y',
        '1-methoxy-3-indolylmethyl GSL': 'g',
        '4-hydroxy-3-indolylmethyl GSL': 'm',
        '4-methoxy-3-indolylmethyl GSL': 'r'
    }
    var_names = list(subs_dict.values())[1:]  # adjust for your models
    timepoints = t_range_eval  # your time vector
    # Pre-allocate storage: (n_sims, n_time, n_vars)
    all_solutions = np.zeros((num_sims, len(timepoints), len(var_names)))
    enzyme_solutions = np.zeros((num_sims, len(timepoints), unfit_model.number_enzymes))
    # breakpoint()
    for i in range(num_sims):
        # print(i)
        params = pdf.dataset.T[i, :].ravel()
        params[smc_params.log_priors] = 10 ** params[smc_params.log_priors]
        unfit_model.set_constants(params)
        output = solve_ivp(
            unfit_model.fitting_derivatives,
            t_sim,
            unfit_model.get_initial_conditions(),
            t_eval=timepoints,
            method='LSODA'
        )
        unfit_model._reset_concentrations()
        # Store transpose so axis=1 is variables
        all_solutions[i] = output.y[:unfit_model.number_metabolites-1, :].T
        if unfit_model.variable_enzymes:
            enzyme_solutions[i]=output.y[unfit_model.number_metabolites-1:, :].T

    # Turn into wide DataFrame
    m_df_wide = pd.DataFrame(
        all_solutions.reshape(-1, len(var_names)),
        columns=var_names
    )
    m_df_wide["Time"] = np.tile(timepoints, num_sims)
    # df_wide["Simulation"] = np.repeat(np.arange(n_sims), len(timepoints))

    # Convert to long when needed
    m_synth_df = m_df_wide.melt(
        id_vars="Time",
        var_name="Metabolite",
        value_name="Concentration"
    )
    sns.set_style("whitegrid")
    sns.lineplot(data=m_synth_df, x='Time', y='Concentration', hue='Metabolite', palette=palette, errorbar=('ci'), estimator='mean')
    plt.xlabel('Time (hour)')
    plt.ylabel(r'Concentration $\frac{\mu Mole}{gdw}$')
    if exp_conds == 'Mock' or 'JA':
        plt.title(f'Glucosinolates over Time {exp_conds.capitalize()} Treatment')
    else:
        plt.title(f'Glucosinolates over Time {exp_conds}')
    if save_figs:
        create_dirs([fig_dir])
        plt.savefig(fig_dir+filename+'_m_many.png')
        plt.close()
    else:
        plt.show()
    if unfit_model.variable_enzymes:
        # Turn into wide DataFrame
        e_df_wide = pd.DataFrame(
            enzyme_solutions.reshape(-1, len(unfit_model.enzyme_dictionary.values())),
            columns=unfit_model.enzyme_dictionary.values()
        )
        e_df_wide["Time"] = np.tile(timepoints, num_sims)
        e_df_wide["Simulation"] = np.repeat(np.arange(num_sims), len(timepoints))

        # Convert to long when needed
        e_synth_df = e_df_wide.melt(
            id_vars=["Time", "Simulation"],
            var_name="Enzyme",
            value_name="Concentration"
        )
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 8))
        # sns.lineplot(data=e_synth_df, x='Time', y='Concentration', hue='Enzyme', errorbar=('sd'), estimator='mean')
        sns.lineplot(data=e_synth_df, x='Time', y='Concentration', hue='Enzyme')
        # sns.lineplot(data=e_synth_df, x='Time', y='Concentration', hue='Enzyme', units='Simulation', estimator=None,
        #              lw=0.8, alpha=0.5)
        plt.xlabel('Time (hour)')
        plt.ylabel(r'Concentration $\frac{\mu Mole}{gdw}$')
        if exp_conds == 'Mock' or 'JA':
            plt.title(f'Enzymes over Time {exp_conds.capitalize()} Treatment')
        else:
            plt.title(f'Glucosinolates over Time {exp_conds}')
        if save_figs:
            create_dirs([fig_dir])
            plt.savefig(fig_dir+filename+'_e_many.png')
            plt.close()
        else:
            plt.show()


def distribution_plotting_enzyme(unfit_model, yt, t_sim, t_eval, pdf, smc_params, fig_dir=None, filename=None,
                                    num_sims=1000, exp_conds='Control', save_figs=False):
    plot_set_formating()
    t_range_eval = np.arange(t_sim[0], t_sim[1], 1)
    subs_dict = unfit_model.metabolite_dictionary
    experiment_data = pd.DataFrame(yt.T, index=t_eval, columns=list(subs_dict.values())[1:])
    plt.figure(figsize=(12, 8))
    experiment_data = pd.melt(experiment_data, ignore_index=False, var_name='Metabolite', value_name='Concentration')
    experiment_data = experiment_data.reset_index().rename(columns={'index': 'Time'})
    palette = {
        '3-indolylmthyl GLS glucobrassicin': 'b',
        '1-hydroxy-3-indolylmethyl GSL': 'y',
        '1-methoxy-3-indolylmethyl GSL': 'g',
        '4-hydroxy-3-indolylmethyl GSL': 'm',
        '4-methoxy-3-indolylmethyl GSL': 'r'
    }

    sns.scatterplot(data=experiment_data, x='Time', y='Concentration', hue='Metabolite', palette=palette, legend=False)
    palette = {
        '3-indolylmthyl GLS glucobrassicin': 'b',
        '1-hydroxy-3-indolylmethyl GSL': 'y',
        '1-methoxy-3-indolylmethyl GSL': 'g',
        '4-hydroxy-3-indolylmethyl GSL': 'm',
        '4-methoxy-3-indolylmethyl GSL': 'r'
    }
    var_names = list(subs_dict.values())[1:]  # adjust for your models
    timepoints = t_range_eval  # your time vector
    # Pre-allocate storage: (n_sims, n_time, n_vars)
    all_solutions = np.zeros((num_sims, len(timepoints), len(var_names)))

    for i in range(num_sims):
        # print(i)
        params = pdf.dataset.T[i, :].ravel()
        params[smc_params.log_priors] = 10 ** params[smc_params.log_priors]
        unfit_model.set_constants(params)
        output = solve_ivp(
            unfit_model.fitting_derivatives,
            t_sim,
            unfit_model.get_initial_conditions(),
            t_eval=timepoints,
            method='LSODA'
        )
        unfit_model._reset_concentrations()
        # Store transpose so axis=1 is variables
        all_solutions[i] = output.y[:unfit_model.number_metabolites-1, :].T

        # Turn into wide DataFrame
    df_wide = pd.DataFrame(
        all_solutions.reshape(-1, len(var_names)),
        columns=var_names
    )
    df_wide["Time"] = np.tile(timepoints, num_sims)
    # df_wide["Simulation"] = np.repeat(np.arange(n_sims), len(timepoints))

    # Convert to long when needed
    synth_df = df_wide.melt(
        id_vars="Time",
        var_name="Metabolite",
        value_name="Concentration"
    )

    plt.xlabel('Time (hour)')
    plt.ylabel(r'Concentration $\frac{\mu Mole}{gdw}$')
    plt.title(f'Enzymes over Time {exp_conds.capitalize()} Treatment')

    sns.lineplot(data=synth_df, x='Time', y='Concentration', hue='Metabolite', palette=palette, errorbar=('sd'), estimator='mean')
    if save_figs:
        create_dirs([fig_dir])
        plt.savefig(fig_dir+filename+'_manyshots_fig.png')
        plt.close()
    else:
        plt.show()