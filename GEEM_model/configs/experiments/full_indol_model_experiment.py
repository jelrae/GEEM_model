import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from GEEM_model.data import indole_data_loaders
from GEEM_model.configs.fitting.fitting_params_config.full_model_full_fit_wide_bounds import make_params_to_fit


def build_control_meja_experiment(data_path, mrna_dictionary):
    glucs_to_fit = [
        '3-indolylmthyl GLS glucobrassicin',
        '1-hydroxy-3-indolylmethyl GSL',
        '1-methoxy-3-indolylmethyl GSL',
        '4-hydroxy-3-indolylmethyl GSL',
        '4-methoxy-3-indolylmethyl GSL'
    ]

    # Load in the control conditions
    meta_data = indole_data_loaders.load_and_process_indole_data_metabolomics(
        data_path, exp_type='control'
    )

    name_conv, experiment_data, t_eval = meta_data

    experiment_data = experiment_data[glucs_to_fit]
    init_conds = experiment_data.loc[0, glucs_to_fit].to_numpy().T
    meta_spl = InterpolatedUnivariateSpline(
        experiment_data.index,
        experiment_data['3-indolylmthyl GLS glucobrassicin'],
        k=1
    )

    gene_conv_dict, _, mrna_equations_full = indole_data_loaders.load_and_process_indole_data_transcripts(
        data_path, exp_type='A')

    mrna_equations = {
        loc: mrna_equations_full[name] for loc, name in mrna_dictionary.items()
    }

    model_conditions = []
    yt = []

    model_conditions.append([
        init_conds[:, None],
        {0: meta_spl},
        mrna_equations
    ])

    yt.append(experiment_data.to_numpy()[1:, 1:].T)

    # Load in the meJA conditions
    meta_data = indole_data_loaders.load_and_process_indole_data_metabolomics(
        data_path, exp_type='meJA'
    )

    name_conv, experiment_data, t_eval = meta_data

    experiment_data = experiment_data[glucs_to_fit]
    init_conds = experiment_data.loc[0, glucs_to_fit].to_numpy().T
    meta_spl = InterpolatedUnivariateSpline(
        experiment_data.index,
        experiment_data['3-indolylmthyl GLS glucobrassicin'],
        k=1
    )

    gene_conv_dict, _, mrna_equations_full = indole_data_loaders.load_and_process_indole_data_transcripts(
        data_path, exp_type='B')

    mrna_equations = {
        loc: mrna_equations_full[name] for loc, name in mrna_dictionary.items()
    }

    model_conditions.append([
        init_conds[:, None],
        {0: meta_spl},
        mrna_equations
    ])

    yt.append(experiment_data.to_numpy()[1:, 1:].T)

    return {
        "model_conditions": model_conditions,
        "yt": yt,
        "t_eval": t_eval,
        "name_conversion": name_conv
    }