# MODELS
from GEEM_model.configs.models import full_indole_model_variable_enzyme

# EXPERIMENTS
from GEEM_model.configs.experiments import full_indol_model_experiment

# FITTING CONFIGS
from GEEM_model.configs.fitting.smc_abc_full_indole_model_variable_enzyme import (
    make_smc_params
)
from GEEM_model.configs.fitting.fitting_params_config import(
    full_model_full_fit_wide_bounds as make_params_ff_wb,
    full_model_full_fit_AI_bounds as make_params_ff_aib,
)

MODELS = {
    "full_indol_model_var_enz": full_indole_model_variable_enzyme,
}

EXPERIMENTS = {
    "full_indol_model_exp": full_indol_model_experiment,
}

FITTING_CONFIGS = {
    "fm_ff_wb": make_params_ff_wb.make_params_to_fit,
    "fm_ff_ai": make_params_ff_aib.make_params_to_fit,
}

SMC_CONFIGS = {
    "smc_abc_full_indole": make_smc_params,
}