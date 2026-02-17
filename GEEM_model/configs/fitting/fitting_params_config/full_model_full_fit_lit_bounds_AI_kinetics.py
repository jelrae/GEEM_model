def make_params_to_fit(leaf_g_fw):

    # In units of mol/h-gFW originally, then transformed into micro moles /h
    v_synth_lb = 9.5e-13 * leaf_g_fw * 1e6
    v_synth_ub = 2.9e-9 * leaf_g_fw * 1e6
    v_synth_vals = (v_synth_lb, v_synth_ub)

    # In units of 1/day originally then transformed into 1/hour
    kde_lb = 0.05 / 24  # low from secondary metabolism
    kde_ub = 0.5 / 24  # High from stress
    kds_vals = (kde_lb, kde_ub)

    PARAMS_TO_FIT = {
        'v_synth_rates': [{
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
        }, v_synth_vals, 'log-uniform'],
        'enzyme_degradation_rates': [{
            0: 7,
            1: 8,
            2: 9,
            3: 10,
            4: 11,
            5: 12,
            6: 13,
        }, kds_vals, 'uniform'],
    }

    return PARAMS_TO_FIT