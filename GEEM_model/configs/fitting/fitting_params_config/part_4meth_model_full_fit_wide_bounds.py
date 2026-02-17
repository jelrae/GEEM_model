def make_params_to_fit(leaf_g_fw):

    # rates here taken from table from 2024 LR enyzme, origionally in 1/s now in 1/hr
    lb_kc_val = 0.0010 * 3600  # s/hr conversion
    ub_kc_val = 500 * 3600  # The highest pred AI value was 200, so doing a bit over double that
    k_cat_vals = (lb_kc_val, ub_kc_val)

    # In units of micro moles, should be in the same range as the metabolites themselves according to evolution
    lb_km_val = 1e-3
    ub_km_val = 500 # The highest pred AI value was 200, so doing a bit over double that
    k_m_vals = (lb_km_val, ub_km_val)

    # In units of mol/h-gFW originally, then transformed into micro moles /h
    v_synth_lb = 9.5e-13 * leaf_g_fw * 1e6
    v_synth_ub = 2.9e-9 * leaf_g_fw * 1e6
    v_synth_vals = (v_synth_lb, v_synth_ub)

    # In units of 1/day originally then transformed into 1/hour
    kde_lb = 0.05 / 24  # low from secondary metabolism
    kde_ub = 0.5 / 24  # High from stress
    kde_vals = (kde_lb, kde_ub)

    PARAMS_TO_FIT = {
        'k_cat': [{
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
        }, k_cat_vals, 'log-uniform'],
        'km': [{
            0: 5,
            1: 6,
            2: 7,
            3: 8,
            4: 9,
        }, k_m_vals, 'log-uniform'],
        'v_synth_rates': [{
            0: 10,
            1: 11,
            2: 12,
            3: 13,
            4: 14,
        }, v_synth_vals, 'log-uniform'],
        'enzyme_degradation_rates': [{
            0: 15,
            1: 16,
            2: 17,
            3: 18,
            4: 19,
        }, kde_vals, 'uniform']
    }

    return PARAMS_TO_FIT