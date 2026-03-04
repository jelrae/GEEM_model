import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
from GEEM_model.utils import io as GEEM_io

import matplotlib as mpl

# ── PLOS Computational Biology figure standards ──────────────────────────────
# Single column: 5.2 in (13.2 cm) | Full page: 7.5 in (19.05 cm)
# Resolution: 300–600 dpi | Font: Arial 8–12pt | Format: TIFF

PLOS_SINGLE_COL_WIDTH = 5.2   # inches
PLOS_FULL_PAGE_WIDTH  = 7.5   # inches
PLOS_MAX_HEIGHT       = 8.75  # inches
PLOS_DPI              = 300   # use 600 for figures with many small elements


def get_plos_figsize(column="single", aspect_ratio=0.75, n_panels_vertical=1):
    """
    Returns a (width, height) tuple conforming to PLOS CompBiol figure rules.

    Parameters
    ----------
    column : str
        "single" (≤5.2 in, aligns with text column) or
        "double" (full page width, 7.5 in)
    aspect_ratio : float
        Height as a fraction of width. Default 0.75 works well for
        line/violin plots. Increase for taller figures.
    n_panels_vertical : int
        Number of stacked panels. Scales height accordingly.
    """
    width = PLOS_SINGLE_COL_WIDTH if column == "single" else PLOS_FULL_PAGE_WIDTH
    height = min(width * aspect_ratio * n_panels_vertical, PLOS_MAX_HEIGHT)
    return (width, height)


def set_plos_style():
    """
    Sets global matplotlib/seaborn style to comply with PLOS figure requirements.
    Call once at the top of your script.
    """
    sns.set_theme(style="ticks")  # clean base; avoids heavy gridlines

    mpl.rcParams.update({
        # ── Fonts ─────────────────────────────────────────────────────────────
        "font.family":        "Arial",
        "font.size":          9,        # safe default within 8–12pt range
        "axes.titlesize":     10,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,

        # ── Lines & ticks ─────────────────────────────────────────────────────
        "axes.linewidth":     0.8,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "lines.linewidth":    1.2,      # clean for line plots at small sizes

        # ── Layout ────────────────────────────────────────────────────────────
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "figure.autolayout":  True,     # prevents label clipping on save

        # ── Output ────────────────────────────────────────────────────────────
        "savefig.format":     "tiff",
        "savefig.bbox":       "tight",  # removes excess whitespace on save
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
    })


def save_plos_figure(fig, filename, dpi=PLOS_DPI):
    """
    Saves a figure as a TIFF with a 2pt white border, per PLOS requirements.

    Parameters
    ----------
    fig : matplotlib Figure object
    filename : str
        Output filename. '.tiff' will be appended if not present.
    dpi : int
        300 for most plots; use 600 for dense/small-text figures.
    """
    if not filename.endswith(".tiff"):
        filename += ".tiff"
    fig.savefig(
        filename,
        dpi=dpi,
        format="tiff",
        bbox_inches="tight",
        pad_inches=0.028,   # ~2pt white border (2pt / 72pt per inch ≈ 0.028)
        facecolor="white"
    )
    print(f"Saved: {filename} at {dpi} dpi")


# ── Call this once at the top of your script ─────────────────────────────────
set_plos_style()


# def plot_set_formating(SMALL_SIZE=12, MEDIUM_SIZE=15, BIGGER_SIZE=20, BIGGEST_SIZE=27):
#     plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
#     plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
#     plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
#     plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
#     plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
#     plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
#     plt.rc('figure', titlesize=BIGGEST_SIZE)  # fontsize of the figure title


def plotting_with_model(unfit_model, synth_model, t_sim, t_eval):
    # set_plos_style()
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
    # set_plos_style()
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
        GEEM_io.create_dirs([fig_dir])
        plt.savefig(fig_dir+filename+'_fig.png')
        plt.close()
    else:
        plt.show()


def distribution_plotting_with_data(unfit_model, yt, t_sim, t_eval, pdf, smc_params, fig_dir=None, filename=None,
                                    num_sims=1000, exp_conds='Control', save_figs=False, sample_df=None):
    # set_plos_style()
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
        GEEM_io.create_dirs([fig_dir])
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
            GEEM_io.create_dirs([fig_dir])
            plt.savefig(fig_dir+filename+'_e_many.png')
            plt.close()
        else:
            plt.show()


def distribution_plotting_enzyme(unfit_model, yt, t_sim, t_eval, pdf, smc_params, fig_dir=None, filename=None,
                                    num_sims=1000, exp_conds='Control', save_figs=False):
    # set_plos_style()
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
        GEEM_io.create_dirs([fig_dir])
        plt.savefig(fig_dir+filename+'_manyshots_fig.png')
        plt.close()
    else:
        plt.show()