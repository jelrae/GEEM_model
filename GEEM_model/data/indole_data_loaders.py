import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy as sp
import numpy as np


def load_and_process_indole_data_transcripts(data_loc_fp, leaf_mg_dw=4.697, exp_type='A', lowest_scaling=1, init_conds='Control', lowest_val_swap=False):
    # load mrna data
    # The mRNA data time is originally in minutes
    # mrna_fp = data_loc_fp+'julia_mrna_data/mRNA_counts/Ath/processed_data/mean_counts.csv'
    mrna_fp = data_loc_fp+'julia_mrna_data/mRNA_counts/Ath/processed_data/normalized_mean_counts_min_counts.csv'
    # Old Files
    # mrna_fp = data_loc_fp+'julia_mrna_data/mRNA_counts/Ath/processed_data/normalized_mean_counts.csv'
    mrna_df = pd.read_csv(mrna_fp)
    mrna_df.head()

    leaf_g_dw = leaf_mg_dw/1000

    min_time_rows = mrna_df.loc[mrna_df.groupby(['Gene', 'Type'])['Time'].idxmin()]
    # Step 2: Modify the Time to 12960
    new_rows = min_time_rows.copy()
    new_rows['Time'] = 1440.0

    # Step 3: Append the new rows to the original DataFrame
    mrna_df = pd.concat([mrna_df, new_rows], ignore_index=True)
    new_rows['Time'] = 12960.0
    mrna_df = pd.concat([mrna_df, new_rows], ignore_index=True)

    gene_conversion = {
        'AT4G37430': 'CYP81F1',
        'AT5G57220': 'CYP81F2',
        'AT4G37400': 'CYP81F3',
        'AT4G37410': 'CYP81F4',
        'AT1G21100': 'IGMT1',
        'AT1G21120': 'IGMT2',
        'AT1G76790': 'IGMT5'
    }

    mrna_df.head()
    mrna_fraction_df = mrna_df[mrna_df['Gene'].isin(gene_conversion.keys())].copy()
    mrna_fraction_df['Gene'] = mrna_fraction_df['Gene'].map(gene_conversion)
    mrna_fraction_control_df = mrna_fraction_df[mrna_fraction_df.Type == 'A']
    mrna_fraction_df = mrna_fraction_df[mrna_fraction_df.Type == exp_type]
    if init_conds == 'Control':
        mrna_fraction_df.loc[mrna_fraction_df.Time == 0, 'Counts'] = mrna_fraction_control_df.loc[mrna_fraction_control_df.Time == 0, 'Counts'].values
    # mrna_fraction_df = mrna_fraction_df.pivot(index='Gene', columns='Time', values='Counts')

    # Need to get a linear interpoloation of each of these

    mrna_concentration_equations = {}
    for gene in gene_conversion.values():
        # Fraction of counts * number of mrna in a cell, then * number of cells in a leaf to get the total mRNA present
        mrna_counts_df = mrna_fraction_df[mrna_fraction_df['Gene'] == gene].copy()
        if lowest_val_swap:
            lowest_value = mrna_counts_df[mrna_counts_df.Counts != 0].Counts.min()
            mrna_counts_df['Counts'] = mrna_counts_df['Counts'].replace(0, lowest_value/lowest_scaling)
            print(f'Lowest value of {gene}: {lowest_value}')
        mrna_times = mrna_counts_df.Time/60 # Put it from minutes into hours
        n_mrna_per_cell = 250000
        n_cells_per_leaf = 764000
        # the old version of the conversion from fract counts to micro mol / gDW
        # mrna_counts_gene = (mrna_counts_df.Counts) * 1e6 * (1.25 * 121e6)
        # mrna_concentration_gene = (mrna_counts_gene / (sp.constants.N_A * leaf_g_dw)) * 1e9  # Units are micro_mol/gFWo
        mrna_counts_gene = (mrna_counts_df.Counts) * n_mrna_per_cell * n_cells_per_leaf
        mrna_concentration_gene = (mrna_counts_gene / (sp.constants.N_A * leaf_g_dw)) # This makes the units in terms of mol/mgDW
        mrna_concentration_gene = mrna_concentration_gene * 1e6 # This converts to micro-mol/mgDW
        mrna_concentration_spl = InterpolatedUnivariateSpline(mrna_times, mrna_concentration_gene, k=1)

        mrna_concentration_equations[gene] = mrna_concentration_spl
    #     sns.lineplot(x=mrna_times, y=mrna_concentration_gene, label=gene)
    # plt.show()
    mrna_ordered_functs = list(mrna_concentration_equations.values())

    # Returns the mrna_concentration_equations in terms of the enzyme we want rather than gene

    return gene_conversion, mrna_ordered_functs, mrna_concentration_equations


def load_and_process_indole_data_metabolomics(data_loc_fp, exp_type='control', init_conds='Control'):
    # load metabolic data
    metabolite_fp = data_loc_fp+'anna_data/processed_data/metabolite_mole_concentrations_umol_gDW_average.csv'
    metabolite_std_fp = data_loc_fp+'anna_data/processed_data/metabolite_mole_concentrations_umol_gDW_std.csv'
    # Old files
    # metabolite_fp = data_loc_fp+'anna_data/processed_data/old_data_version/metabolite_molar_concentrations_average.csv'
    # metabolite_std_fp = data_loc_fp+'anna_data/processed_data/old_data_version/metabolite_molar_concentrations_std.csv'
    # breakpoint()
    metabolite_df = pd.read_csv(metabolite_fp)
    metabolite_std_df = pd.read_csv(metabolite_std_fp)

    # metabolite_df.Time = metabolite_df.Time * 60
    # Trying with minutes
    metabolite_df.Time = metabolite_df.Time


    # Convert the metabolite names from the models names to the dataset names
    name_conversion_dict = {
        '3-indolylmthyl GLS glucobrassicin': 'Glucobrassicin',
        # '1-hydroxy-3-indolylmethyl GSL' : None,
        '1-methoxy-3-indolylmethyl GSL': 'Neoglucobrassicin',
        '4-hydroxy-3-indolylmethyl GSL': '4-hydroxyglucobrassicin',
        '4-methoxy-3-indolylmethyl GSL': '4-methoxyglucobrassicin'
    }

    inv_name_conversion_dict = {y: x for x, y in name_conversion_dict.items()}

    data_df = metabolite_df[(metabolite_df['Treatment'] == exp_type) & (metabolite_df['Temp'] == 21)].copy()
    if init_conds == 'Control':
        control_df = metabolite_df[(metabolite_df['Treatment'] == 'control') & (metabolite_df['Temp'] == 21)].copy()
        data_df.loc[data_df.Time == 0, :] = control_df.loc[control_df.Time == 0, :].values
        data_df.loc[data_df.Time == 0, 'Treatment'] = exp_type
    data_df.drop(columns=['Treatment', 'Temp'], inplace=True)
    data_df.set_index('Time', inplace=True)
    data_df.drop(columns=['Sinigrin', 'Sinalbin'], inplace=True)

    # Convert to the units micromol/gdw
    data_df.rename(columns=inv_name_conversion_dict, inplace=True)
    experiment_data = data_df[name_conversion_dict.keys()].copy()

    # we use minutes instead
    t_eval = experiment_data.loc[120/60:, :].index.to_numpy()
    IH4_i = experiment_data.loc[0,'4-hydroxy-3-indolylmethyl GSL']
    MH4_i = experiment_data.loc[0,'4-methoxy-3-indolylmethyl GSL']
    MH1_i = experiment_data.loc[0,'1-methoxy-3-indolylmethyl GSL']

    hm_ratio = IH4_i/MH4_i
    IH1_i = MH1_i * hm_ratio
    experiment_data.loc[:, '1-hydroxy-3-indolylmethyl GSL'] = IH1_i

    return name_conversion_dict, experiment_data, t_eval