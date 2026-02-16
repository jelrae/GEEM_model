import numpy as np
from GEEM_model.utils.io import set_seeds


class MichMentenModel:
    """
    This class is an implementation of the michaelis menten models using matrix operations.
        Given a Metabolite concentration vector (Mc),  Enzyme concentration vector (Ec), reactions vector (R), substrate mask matrix (Ms),
        Enzyme mask matrix (Me), interaction matrix (I), k_cat vector (Kc), km vector (Km), degradation vector (kdm)  where:
            S = (n x 1) (n = number of compounds)
            E = (e x 1) (e = number of enzymes)
            R = (r x 1) (r = number of reactions)
            Mc = (r x s)
            Ec = (r x e)
            I = (n x r)
            Kc = (r x 1)
            Km = (r x 1)
            kdm = (S x 1) (0 if there is no degradation or the constant if there is degradation)
        The Class will store the functions to perform the calcs of the rates for the MM models
    """

    def __init__(self,
                 metabolite_dictionary,
                 enzyme_dictionary,
                 reaction_dictionary,
                 const_metabolites,
                 init_meta_conc,
                 init_enzy_conc,
                 time_range,
                 # react_code=None,
                 interaction_matrix=None,
                 mask_m=None,
                 mask_e=None,
                 k_cat=None,
                 km=None,
                 degradation_rates_metabolites=None,
                 alternative_concentrations=None,
                 variable_enzymes=False,
                 mrna_dictionary=None,
                 mrna_reaction_dictionary=None,
                 mrna_concentrations_eq_dict=None,
                 # mrna_concentrations=None,
                 enzyme_synth_rates=None,
                 enzyme_degradation_rates=None,
                 parameters_to_fit=None,
                 parameter_search_space=None,
                 fit_v_enzyme_synth=False,
                 v_synth_rates=None,
                 leaf_g_dw=None,
                 day_night_cycle=False,
                 day_night_type=None,
                 day_v_synthesis_rates=np.array([1]),
                 night_v_synthesis_rates=np.array([1]),
                 constrained_metabolite_decay=False,
                 ):

        # Storing basic information
        self.metabolite_dictionary = metabolite_dictionary
        self.enzyme_dictionary = enzyme_dictionary
        self.reaction_dictionary = reaction_dictionary
        self.number_metabolites = len(self.metabolite_dictionary)
        self.number_enzymes = len(self.enzyme_dictionary)
        self.number_reactions = len(self.reaction_dictionary)
        self.variable_enzymes = variable_enzymes
        self.time_range = time_range
        # self.t_eval_dict = t_eval_dict

        # Which substrates are constant in the formulas
        self.const_subs = const_metabolites

        # It is assumed that alternative concentrations is a dictionary with structure:
        # {location, function} the function should only take time as an argument
        self.alternative_concentrations = alternative_concentrations
        if alternative_concentrations:
            self.extract_alternative_concentrations(alternative_concentrations)
        else:
            self.alternative_concentrations_locs = None
            self.alternative_concentrations_funcs = None

        # Storing the initial concentration and setting the concentration to the initial concentration
        self.initial_conditions = init_meta_conc.copy()
        self.metabolite_concentration = init_meta_conc.copy()

        self.initial_enzyme_concentration = init_enzy_conc.copy()
        self.enzyme_concentration = init_enzy_conc.copy()

        # Setting up the degradation of the substrates
        self.degradation_rates_metabolites = degradation_rates_metabolites

        # Creating the masks and interaction matrix
        self.interaction_matrix = None
        self.mask_metabolites = None
        self.mask_enzymes = None
        self.setup_mask_matrices_michaelis_menten(interaction_matrix, mask_m, mask_e)

        # Storing the K's for the system
        self.init_k_cat = k_cat.copy()
        self.k_cat = k_cat.copy()
        self.init_km = km.copy()
        self.km = km.copy()

        self.mrna_dictionary = mrna_dictionary
        self.mrna_reaction_dictionary = mrna_reaction_dictionary
        self.mrna_concentrations_eq_dict = mrna_concentrations_eq_dict
        self.enzyme_synthesis_rates = enzyme_synth_rates
        self.enzyme_degradation_rates = enzyme_degradation_rates

        if self.variable_enzymes:
            # Enzyme Synthesis and degradation based on the mrna concentrations
            # Currently we assume the synthesis rate and degradation rate are in a linear form (ks*mRNA), (kd*E)
            # mrna reaction dict is mrna, target enzyme
            self.number_mrna = len(self.mrna_dictionary)
            self.extract_mrna_data(mrna_concentrations_eq_dict)

        self.parameter_search_space = parameter_search_space

        # for if we want to try fitting_functions the v_synth as well
        self.fit_vs = fit_v_enzyme_synth
        self.v_synth_rates = v_synth_rates
        self.leaf_g_dw = leaf_g_dw

        self.day_night_cycle = day_night_cycle
        if self.day_night_cycle:
            self.day_night_type = day_night_type
            self.day_v_synthesis_rates = day_v_synthesis_rates
            self.night_v_synthesis_rates = night_v_synthesis_rates
            if day_night_type == 'Square':
                self.set_ek_synth_day_night()

        self.constrained_metabolite_decay = constrained_metabolite_decay

        if self.constrained_metabolite_decay:
            self.calc_metabolite_decay_rate()

        self.parameters_to_fit = parameters_to_fit
        self.num_parameters_to_fit = len(self.get_constants())
        self.parameter_names = self.make_parameter_names()

    def extract_mrna_data(self, mrna_concentrations_eq_dict):
        self.mrna_concentrations_eq_dict = mrna_concentrations_eq_dict
        self.mrna_concentrations_locs = list(mrna_concentrations_eq_dict.keys())
        self.mrna_concentrations_funcs = list(mrna_concentrations_eq_dict.values())
        self.initial_mrna_concentration = np.fromiter((f(0) for f in self.mrna_concentrations_funcs),
                                                      dtype=float, count=self.number_mrna).reshape(-1, 1)

    def extract_alternative_concentrations(self, alternative_concentrations):
        self.alternative_concentrations = alternative_concentrations
        self.alternative_concentrations_locs = list(alternative_concentrations.keys())
        self.alternative_concentrations_funcs = list(alternative_concentrations.values())

    def make_parameter_names(self):
        parameter_names = [
            parameter.replace('degradation_rates_metabolites', 'dgr') + str(param_num) for parameter, access_dict in
            self.parameters_to_fit.items()
            for param_num, param_ind in access_dict.items()
        ]
        parameter_names = [parameter.replace('v_synth_rates', 'vsr') for parameter in parameter_names]
        parameter_names = [parameter.replace('enzyme_degradation_rates', 'edr') for parameter in parameter_names]
        parameter_names = [parameter.replace('day_v_synthesis_rate', 'dvsr') for parameter in parameter_names]
        parameter_names = [parameter.replace('night_v_synthesis_rate', 'nvsr') for parameter in parameter_names]
        return parameter_names

    def create_mask_matrices_michaelis_menten(self):
        mask_metabolites = np.zeros((self.number_reactions, self.number_metabolites))
        mask_enzymes = np.zeros((self.number_reactions, self.number_enzymes))
        interaction_matrix = np.zeros((self.number_metabolites, self.number_reactions))

        for reaction_number, reaction_info in self.reaction_dictionary.items():
            sub, prod, enzyme = reaction_info
            mask_metabolites[reaction_number, sub] = 1
            mask_enzymes[reaction_number, enzyme] = 1
            interaction_matrix[[sub, prod], reaction_number] = [-1, 1]

        for metabolite in self.const_subs:
            interaction_matrix[metabolite, :] = 0

        if self.degradation_rates_metabolites is not None:
            degradation_interactions = np.diag(self.degradation_rates_metabolites.squeeze())
            degradation_interactions[degradation_interactions != 0] = -1
            for metabolite in self.const_subs:
                degradation_interactions[metabolite, :] = 0
            interaction_matrix = np.hstack((interaction_matrix, degradation_interactions))

        return interaction_matrix, mask_metabolites, mask_enzymes

    def setup_mask_matrices_michaelis_menten(self, interaction_matrix, mask_m, mask_e):
        interaction_m, metabolite_m, enzyme_m = self.create_mask_matrices_michaelis_menten()
        if interaction_matrix is not None:
            # Storing the interaction matrix
            self.interaction_matrix = interaction_matrix
        else:
            self.interaction_matrix = interaction_m

        if mask_m is not None:
            if mask_m.shape[0] != self.number_reactions or mask_m.shape[1] != self.number_metabolites:
                raise ValueError('Metabolite Mask matrix must be shape number of reactions x number of metabolites')
            # Storing the masks for the substrates and enzymes which will be in the reaction
            self.mask_metabolites = mask_m.reshape((self.number_reactions, self.number_metabolites))
        else:
            self.mask_metabolites = metabolite_m

        if mask_e is not None:
            if mask_e.shape[0] != self.number_reactions or mask_e.shape[1] != self.number_enzymes:
                raise ValueError('Enzyme Mask matrix must be shape number of reactions x number of enzymes')
            self.mask_enzymes = mask_e.reshape((self.number_reactions, self.number_enzymes))
        else:
            self.mask_enzymes = enzyme_m

    def fitting_derivatives(self, ti, y0):
        # if self.alternative_concentrations:
        #     for location, function in self.alternative_concentrations.items():
        #         y0[location, :] = function(ti)
        if self.variable_enzymes:
            return self.fitting_derivatives_var_enzyme(ti, y0)
        else:
            return self.fitting_derivatives_const_enzyme(ti, y0)

    def fitting_derivatives_const_enzyme(self, ti, y0):
        if self.alternative_concentrations:
            function_driven_concentrations = [fun(ti) for fun in self.alternative_concentrations_funcs]
            y0 = np.insert(y0, self.alternative_concentrations_locs, function_driven_concentrations)
        self.metabolite_concentration = y0[:, None]
        rates = self.interaction_matrix.dot(self.calc_rates_metabolites(ti)).ravel()
        if self.alternative_concentrations:
            rates = np.delete(rates, self.alternative_concentrations_locs)
        return rates

    def fitting_derivatives_var_enzyme(self, ti, y0):
        if self.alternative_concentrations_funcs:
            function_driven_concentrations = [fun(ti) for fun in self.alternative_concentrations_funcs]
            y0 = np.insert(y0, self.alternative_concentrations_locs, function_driven_concentrations)
        self.metabolite_concentration = y0[:self.number_metabolites, None]
        self.enzyme_concentration = y0[self.number_metabolites:, None]
        mrna_concentration = self.calc_mrna_levels(ti)
        rates_metabolites = self.interaction_matrix.dot(self.calc_rates_metabolites(ti))
        if self.alternative_concentrations:
            rates_metabolites = np.delete(rates_metabolites, self.alternative_concentrations_locs)
        rates_enzymes = self.calc_rates_enzymes(mrna_concentration, ti % 24)
        rates = np.concatenate((rates_metabolites.ravel(), rates_enzymes.ravel()))
        return rates

    def model_derivatives(self, ti):
        return self.interaction_matrix.dot(self.calc_rates_metabolites(ti))

    def calc_rates_metabolites(self, ti):
        # dS/dt = E Kcat S / Km + S
        meta_dot = self.mask_metabolites.dot(self.metabolite_concentration)
        rates_meta = (self.mask_enzymes.dot(self.enzyme_concentration) * self.k_cat * meta_dot) / (self.km + meta_dot)
        if self.degradation_rates_metabolites is not None:
            # Metabolite Degradation rate k * S
            rates_meta_deg = self.degradation_rates_metabolites * self.metabolite_concentration
            rates = np.zeros((self.number_metabolites + rates_meta.shape[0], 1))
            rates[:rates_meta.shape[0], 0] = rates_meta.ravel()
            rates[rates_meta.shape[0]:, 0] = rates_meta_deg.ravel()
            # rates = np.vstack([rates, rates_meta_deg])
        else:
            rates = rates_meta
        return rates

    def calc_rates_enzymes(self, mrna_concentrations, cycle_time=0):
        # Structure for this is dE/dt = k_synth * mRNA - k_deg * E
        if self.day_night_cycle:
            self.set_ek_synth_time(cycle_time)
        rates = ((mrna_concentrations * self.enzyme_synthesis_rates)
                 - (self.enzyme_degradation_rates * self.enzyme_concentration))
        # return rates * 0.00000001
        return rates

    def calc_ek_synth(self, cycle_time):
        # calculate the ksn assuming that over the course of the day the enzyme level difference is 0
        if 22 <= cycle_time or cycle_time <= 8:
            # this is for the day part of the cycle
            k_synth = (((self.enzyme_synthesis_rates * self.initial_enzyme_concentration)
                       - (14/24)*self.night_v_synthesis_rates)) * (24/10) / self.initial_enzyme_concentration
        else:
            # this is for the night part of the cycle
            k_synth = (((self.enzyme_synthesis_rates * self.initial_enzyme_concentration)
                        - (10 / 24) * self.day_v_synthesis_rates)) * (24 / 14) / self.initial_enzyme_concentration
        return k_synth

    def set_ek_synth_day_night(self):
        self.enzyme_synthesis_rates_day = self.calc_ek_synth(0)
        self.enzyme_synthesis_rates_night = self.calc_ek_synth(10)

    def set_ek_synth_time(self, cycle_time):
        if 22 <= cycle_time or cycle_time <= 8:
            self.enzyme_synthesis_rates = self.enzyme_synthesis_rates_day
        else:
            self.enzyme_synthesis_rates = self.enzyme_synthesis_rates_night

    # def calc_v_synthesis_rates_day_night(self, ti):
    #     # Determines what the value of the synthesis rate would be in the dat night cycle
    #     cycle_time = ti % 24
    #     if self.day_night_type == 'Sine':
    #         # The setup for the sine wave function, since the window is uneven, we have to have 2 functions
    #         peak_amplitude_d = self.day_v_synthesis_rates * (np.pi / 2)
    #         peak_amplitude_n = self.night_v_synthesis_rates * (np.pi / 2)
    #         vert_shift = (peak_amplitude_d + peak_amplitude_n) / 2
    #         if cycle_time < 8:
    #             # Day side
    #             phase_shift = -2
    #             v_synth_time = peak_amplitude_d * np.sin(
    #                 ((1 / (2 * 10)) * 2 * np.pi) * (cycle_time - phase_shift)) + vert_shift
    #         elif 22 <= cycle_time < 24:
    #             # Day side
    #             phase_shift = +22
    #             v_synth_time = peak_amplitude_d * np.sin(
    #                 ((1 / (2 * 10)) * 2 * np.pi) * (cycle_time - phase_shift)) + vert_shift
    #         else:
    #             phase_shift = (14 + 8)
    #             v_synth_time = peak_amplitude_n * np.sin(
    #                 ((1 / (2 * 14)) * 2 * np.pi) * (cycle_time - phase_shift)) + vert_shift
    #     elif self.day_night_type == 'Square':
    #         if 22 <= cycle_time or cycle_time <= 8:
    #             v_synth_time = self.day_v_synthesis_rates
    #         else:
    #             v_synth_time =  self.night_v_synthesis_rates
    #     else:
    #         # Assume square if nothing given
    #         if 22 <= cycle_time or cycle_time <= 8:
    #             v_synth_time = self.day_v_synthesis_rates
    #         else:
    #             v_synth_time = self.night_v_synthesis_rates
    #     return v_synth_time

    def calc_mrna_levels(self, ti):
        # mrna_concentrations = np.zeros((self.number_mrna, 1))
        # for ind, mrna in enumerate(self.mrna_dictionary.values()):
        #     mrna_concentrations[ind, 0] = self.mrna_concentrations_eq_dict[mrna](ti)
        # return mrna_concentrations
        # funcs = list(self.mrna_concentrations_eq_dict.values())
        mrna_concentration = np.fromiter((f(ti) for f in self.mrna_concentrations_funcs), dtype=float,
                                         count=self.number_mrna).reshape(-1, 1)
        return mrna_concentration

    def set_constants(self, params):
        params = np.array(params).ravel()
        for param, locations in self.parameters_to_fit.items():
            param_to_update = getattr(self, param)
            param_to_update[list(locations.keys()), 0] = params[list(locations.values())]
        # param_names = self.parameters_to_fit.keys()
        if self.variable_enzymes and not self.day_night_cycle:
            self.calc_enzyme_parameters()
        if self.constrained_metabolite_decay:
            self.calc_metabolite_decay_rate()
        if self.day_night_cycle:
            self.set_ek_synth_day_night()

    def calc_enzyme_parameters(self):
        self.initial_enzyme_concentration = self.v_synth_rates / self.enzyme_degradation_rates
        self.enzyme_concentrations = self.initial_enzyme_concentration.copy()
        self.enzyme_synthesis_rates = self.v_synth_rates / self.initial_mrna_concentration

    def calc_metabolite_decay_rate(self):
        # first set the degredation rates to 0 for the metabolites
        self.degradation_rates_metabolites.fill(0)
        # then calculate the rates of the full system, since those will be good.
        rates = self.fitting_derivatives(0, self.get_initial_conditions())
        # subselect on the desired places and divide by the initial conditions
        self.degradation_rates_metabolites[1:, 0] = rates[:self.number_metabolites-1]/self.initial_conditions[1:, 0]

    def get_constants(self):
        constants_list = []
        for key, value in self.parameters_to_fit.items():
            param_to_update = getattr(self, key)
            constants_list.extend(param_to_update[list(value.keys()), 0].tolist())
        return np.array(constants_list).ravel()

    def get_initial_conditions(self):
        if self.variable_enzymes:
            ic = np.concatenate((self.initial_conditions, self.initial_enzyme_concentration), axis=0).copy()
        else:
            ic = self.initial_conditions.copy()
        if self.alternative_concentrations_funcs:
            ic = np.delete(ic, self.alternative_concentrations_locs)
        return ic

    def _reset_model(self):
        self.k_cat = self.init_k_cat.copy()
        self.km = self.init_km.copy()
        self.concentrations = self.initial_conditions.copy()
        self.enzyme_concentrations = self.initial_enzyme_concentration.copy()

    def _reset_concentrations(self):
        self.concentrations = self.initial_conditions
        self.enzyme_concentrations = self.initial_enzyme_concentration.copy()

    def _change_model_conditions(self, init_meta_conds, alternative_concentrations, mrna_concentrations_eq_dict):
        self.initial_conditions = init_meta_conds.copy()
        self.metabolite_concentration = init_meta_conds.copy()
        self.extract_alternative_concentrations(alternative_concentrations)
        self.extract_mrna_data(mrna_concentrations_eq_dict)

    def print_shapes(self):
        print('Shape of metabolite_concentration: {0}'.format(self.metabolite_concentration.shape))
        print('Shape of enzyme concentrations: {0}'.format(self.enzyme_concentration.shape))
        print('Shape of degradation rates of the metabolites: {0}'.format(self.degradation_rates_metabolites.shape))
        print('Shape of interaction matrix: \n{0}'.format(self.interaction_matrix.shape))
        print('Shape of mask_metabolites:\n {0}'.format(self.mask_metabolites.shape))
        print('Shape of mask_enzymes:\n {0}'.format(self.mask_enzymes.shape))
        print('Shape of kcat: {0}'.format(self.k_cat.shape))
        print('Shape of km: {0}'.format(self.km.shape))

    def print_values(self):
        print('Vals of metabolite_concentration: {0}'.format(self.metabolite_concentration))
        print('Vals of enzyme concentration: {0}'.format(self.enzyme_concentration))
        print('Vals of degradation rates of the metabolites: {0}'.format(self.degradation_rates_metabolites))
        print('Vals of interaction matrix:\n {0}'.format(self.interaction_matrix))
        print('Vals of mask metabolites:\n {0}'.format(self.mask_metabolites))
        print('Vals of mask_enzymes:\n {0}'.format(self.mask_enzymes))
        print('Vals of kcat: {0}'.format(self.k_cat))
        print('Vals of km: {0}'.format(self.km))


def main():
    # # Basic Structure of the test network 1s 1p from before
    # subs_dict = {0: '3-indolylmthyl GLS glucobrassicin',
    #              1: '1-hydroxy-3-indolylmethyl GSL',
    #              }
    # subs = np.array([100, 0])
    # enz_dict = {
    #     0: 'CYP81F1',
    # }
    # enz = np.ones(len(enz_dict))
    #
    # # Reaction Dictionary defines the reactions for MM dynamics. key is reaction number, value is a set containing substrate num, product num, enzyme num
    # react_dict = {
    #     0: (0, 1, 0),
    # }
    #
    # k_cat = np.array([150.0])
    # km = np.array([50.0])
    #
    # models = MichMentenModel(subs_dict, enz_dict, react_dict, [], subs, enz, k_cat=k_cat, km=km)
    # # models.print_values()
    # print(models.calc_rates())
    # breakpoint()

    # Test Structure of the Indol chain network from Anna data

    subs_dict = {0: '3-indolylmthyl GLS glucobrassicin',
                 1: '1-hydroxy-3-indolylmethyl GSL',
                 2: '1-metoxy-3-indolylmethyl GSL',
                 3: '4-hydroxy-3-indolylmethyl GSL',
                 4: '4-metoxy-3-indolylmethyl GSL'
                 }
    subs = np.ones(len(subs_dict))
    enz_dict = {
        0: 'CYP81F1',
        1: 'CYP81F2',
        2: 'CYP81F3',
        3: 'CYP81F4',
        4: 'IGMT1',
        5: 'IGMT2',
        6: 'IGMT5',
    }
    enz = np.ones(len(enz_dict))

    # Reaction Dictionary defines the reactions for MM dynamics. key is reaction number, value is a set containing substrate num, product num, enzyme num
    react_dict = {
        0: (0, 1, 0),
        1: (0, 1, 1),
        2: (0, 1, 2),
        3: (0, 1, 3),
        4: (1, 2, 4),
        5: (1, 2, 5),
        6: (1, 2, 6),
        7: (0, 3, 0),
        8: (0, 3, 1),
        9: (0, 3, 2),
        10: (3, 4, 4),
        11: (3, 4, 5)
    }

    degradation_rates_metabolites = np.zeros(len(subs_dict))
    degradation_rates_metabolites[2] = 1
    degradation_rates_metabolites[3] = 100

    k_cat = np.ones(len(react_dict))
    km = np.ones(len(react_dict))

    # Enzyme Synthesis and Degradation parts of the models

    mRNA_dict = {
        0: 'CYP81F1',
        1: 'CYP81F2',
        2: 'CYP81F3',
        3: 'CYP81F4',
        4: 'IGMT1',
        5: 'IGMT2',
        6: 'IGMT5'
    }

    # Structure of the mrna reaction dict is mrna, target enzyme
    mrna_reacts = {
        0: (0, 0),
        1: (1, 1),
        2: (2, 2),
        3: (3, 3),
        4: (4, 4),
        5: (5, 5),
        6: (6, 6)
    }

    enzyme_synth_consts = np.ones(len(mrna_reacts))

    enzyme_deg_constants = np.ones(len(enz_dict)) * 0.18 / 24 / 60  # 1/d -> 1d / 24 h -> 1h / 60 min

    model = MichMentenModel(subs_dict, enz_dict, react_dict, [], subs, enz, k_cat=k_cat, km=km)
    model.print_values()
    print(model.calc_rates_metabolites(0.01))


if __name__ == "__main__":
    main()
