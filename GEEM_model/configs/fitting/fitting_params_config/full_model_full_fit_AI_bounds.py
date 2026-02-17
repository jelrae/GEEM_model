def make_params_to_fit(leaf_g_fw):

    # From the AI predictions in 1/hr
    kc_ai = [250.77, 245.82, 1139.04, 928.03, 563.13, 429.62, 1039.98, 195.11, 187.05, 295.31, 283.12, 260.32]
    # In units of micro moles, should be in the same range as the metabolites themselves according to evolution
    km_ai = [5.3, 5.30, 6.36, 6.36, 6.09, 6.09, 5.93, 42.83, 114.10, 38.83, 130.10, 50.68]

    lb_kc = [kc_i*1e-1 for kc_i in kc_ai]
    ub_kc = [kc_i*1e1 for kc_i in kc_ai]
    k_cat_vals = [lb_kc, ub_kc]

    lb_km = [km_i*1e-1 for km_i in km_ai]
    ub_km = [km_i*1e1 for km_i in km_ai]
    k_m_vals = [lb_km, ub_km]

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
            5: 5,
            6: 6,
            7: 7,
            8: 8,
            9: 9,
            10: 10,
            11: 11,
        }, k_cat_vals, 'log-uniform'],
        'km': [{
            0: 12,
            1: 13,
            2: 14,
            3: 15,
            4: 16,
            5: 17,
            6: 18,
            7: 19,
            8: 20,
            9: 21,
            10: 22,
            11: 23,
        }, k_m_vals, 'log-uniform'],
        'v_synth_rates': [{
            0: 24,
            1: 25,
            2: 26,
            3: 27,
            4: 28,
            5: 29,
            6: 30,
        }, v_synth_vals, 'log-uniform'],
        'enzyme_degradation_rates': [{
            0: 31,
            1: 32,
            2: 33,
            3: 34,
            4: 35,
            5: 36,
            6: 37,
        }, kde_vals, 'uniform']
    }

    return PARAMS_TO_FIT