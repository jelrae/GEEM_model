import argparse
import os.path

from GEEM_model.registry import (
    MODELS,
    EXPERIMENTS,
    FITTING_CONFIGS,
    SMC_CONFIGS,
)

from GEEM_model.mich_mens_models.MichMentenModel import MichMentenModel
from GEEM_model.fitting_functions.SMC_ABC_multi_conds import (
    parallel_smc_abc_multi_cond
)

import numpy as np

from GEEM_model.utils import io as GEEM_io
from GEEM_model.utils import plotting as GEEM_plotting


def run_pipeline(
        model_name,
        experiment_name,
        fitting_name,
        smc_name,
        data_path,
        number_particles=1000,
        save_figs=False,
        fig_dir="./figures/",
):
    model_configs_dir = './model_configs/'
    model_spec = MODELS[model_name]
    experiment_builder = EXPERIMENTS[experiment_name]
    make_params_to_fit = FITTING_CONFIGS[fitting_name]
    make_smc_params = SMC_CONFIGS[smc_name]

    # --- Experiment ---
    exp = experiment_builder.build_control_meja_experiment(
        data_path,
        model_spec.MRNA_DICT,
        model_spec.META_DICT.values(),
    )

    model_conditions = exp["model_conditions"]
    yt = np.concatenate(exp["yt"], axis=0)
    t_eval = exp["t_eval"]

    # --- Params ---
    leaf_g_fw = 35e-3
    fw_to_dw = 7.45
    leaf_g_dw = leaf_g_fw / fw_to_dw

    PARAMS_TO_FIT = make_params_to_fit(leaf_g_fw)
    params_to_fit_dict = {k: v[0] for k, v in PARAMS_TO_FIT.items()}

    model_save_name = model_configs_dir + 'models/' + model_name + '.pkl'
    smc_save_name = model_configs_dir + 'fit_params/' + fitting_name + '.pkl'

    if os.path.isfile(model_save_name):
        model = GEEM_io.load_params(model_save_name)
    else:
        # --- Model ---
        model = MichMentenModel(
            model_spec.META_DICT,
            model_spec.ENZ_DICT,
            model_spec.REACT_DICT,
            [],
            model_conditions[0][0],
            model_spec.ENZ_CONC,
            (0, max(t_eval)),
            k_cat=model_spec.K_CAT,
            km=model_spec.K_M,
            degradation_rates_metabolites=model_spec.KDM,
            variable_enzymes=True,
            alternative_concentrations=model_conditions[0][1],
            mrna_dictionary=model_spec.MRNA_DICT,
            mrna_reaction_dictionary=model_spec.MRNA_REACT_DICT,
            mrna_concentrations_eq_dict=model_conditions[0][2],
            enzyme_synth_rates=model_spec.KSE,
            enzyme_degradation_rates=model_spec.KDE,
            parameters_to_fit=params_to_fit_dict,
            fit_v_enzyme_synth=model_spec.fit_VES,
            v_synth_rates=model_spec.VS,
            leaf_g_dw=leaf_g_dw,
            constrained_metabolite_decay=model_spec.constrain_KDM,
        )
        GEEM_io.save_params(model_save_name, model)

    if os.path.isfile(smc_save_name):
        smc_params = GEEM_io.load_params(smc_save_name)
    else:
        smc_params = make_smc_params(
            number_particles,
            model_conditions,
            PARAMS_TO_FIT,
        )
        GEEM_io.save_params(smc_save_name, smc_params)

    fit_results = parallel_smc_abc_multi_cond(
        model,
        smc_params,
        [0, 1, 2, 3],
        t_eval,
        yt,
        save_figs,
        fig_dir,
    )




def main(args):
    run_pipeline(
        model_name=args.model,
        experiment_name=args.experiment,
        fitting_name=args.fitting,
        smc_name=args.smc,
        data_path=args.data_path,
        number_particles=args.num_particles,
        save_figs=args.save_figs,
        fig_dir=args.fig_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SMC-ABC pipeline"
    )

    parser.add_argument("--model", type=str, default='full_indol_model_var_enz')
    parser.add_argument("--experiment", type=str, default='full_indol_model_exp')
    parser.add_argument("--fitting", type=str, default='fm_ff_ai')
    parser.add_argument("--smc", type=str, default='smc_abc_full_indole')
    parser.add_argument("--data-path", type=str, default="../data/")

    parser.add_argument("--num-particles", type=int, default=1000)
    parser.add_argument("--save-figs", action="store_true")
    parser.add_argument("--fig-dir", default="./figures/")

    args = parser.parse_args()
    main(args)
