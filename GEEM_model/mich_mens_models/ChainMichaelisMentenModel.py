import numpy as np
from GEEM_model.utils.io import set_seeds


class ChainMichaelisMentenModel:

    def __init__(self,
                 n_substrates,
                 n_products,
                 const_subs,
                 init_concentrations,
                 parameters_to_fit=None,
                 time_range=(0, 2)):
        """
        :param n_substrates: Number of additional substrates to be used in the models
        :param n_products:  Number of products to be used in the models
        :param const_subs: The position in the concentration array of the constant substrates
        :param init_concentrations: The initial concentrations of the metabolites in the models
        """
        if n_products > n_substrates:
            raise ValueError("n_substrates must be larger than number of products")
        # global n_rs, s_rs
        # n_rs, s_rs = set_seeds(42)
        self.ns = n_substrates
        self.np = n_products
        self.const_subs = const_subs
        self.initial_conditions = init_concentrations
        self.concentrations = init_concentrations
        self.interaction_network = None
        self.constants = None
        self.parameters_to_fit = parameters_to_fit
        self.num_parameters_to_fit = len(self.parameters_to_fit)
        self._gen_interaction_network()
        self._gen_constants()
        self.time_range = time_range

    def _gen_interaction_network(self):
        # Interaction network is the stoch. matrix, where rows are species and columns are reactions
        # The reactions go in order of: Sub -> sub reactions, sub -> prod reactions, deg reactions
        # S0->S1, S1-> S2 ... S1 ->P1, S2 -> p2 ...., p1deg, p2 deg .....
        if self.ns == self.np:
            self.interaction_network = np.zeros([self.ns + 1 + self.np, self.np + ((self.ns * 2) - 1) + 1])
        else:
            self.interaction_network = np.zeros([self.ns + 1 + self.np, self.np + ((self.ns * 2) - 1) + 1])
            self.interaction_network[self.ns, -1] = -1
        # The first row will always be 0, since this will remain constant, so starting at 1
        # For the sake of the analysis I will set it to degrade though with the first rate
        self.interaction_network[0, 0] = -1
        for i in range(1, self.ns + 1):
            i_a = [1]
            if i < self.ns:
                i_a.append(-1)
            elif self.ns > 1:
                i_a.append(0)
            if self.ns > 2:
                i_a += [0 for j in range(self.ns - 2)]
            if i <= self.np:
                i_a.append(-1)
            self.interaction_network[i, i - 1:i + len(i_a) - 1] = i_a
        for i in range(self.ns + 1, self.ns + self.np + 1):
            i_a = [1]
            if self.np - 1 > 0:
                i_a += [0 for j in range(self.np - 1)]
            i_a.append(-1)
            self.interaction_network[i, i - 1:i + len(i_a) - 1] = i_a
        if self.const_subs:
            for i in self.const_subs:
                # These are held at a constant value so that will be all 0's
                self.interaction_network[i, :] = 0

    def _gen_constants(self, g_shape=5, g_scale=1):
        # The constants go in order of Km, Vmax for first the substrates, then the products, the kd for degregation
        # Example: S1, S2, P1, P2 would have a vector of Vm1, Km1, Vm2, Km2 .... , kdp1, kdp2, .... kdsn
        # (if we have more S's than p's)
        a_size = (2 * self.ns) + (2 * self.np) + self.np
        if self.ns == 1 and self.np == 0:
            a_size += 1
        # self.constants = n_rs.gamma(g_shape, g_scale, [a_size, 1])
        init_constants = np.random.gamma(g_shape, g_scale, [a_size, 1])
        self.constants = init_constants.copy()
        self.init_constants = init_constants.copy()
        self.bounds = np.array([(1e-8, None) for i in self.constants])

    def _gen_fit_constants(self, means, deviation, itf):
        a_size = len(itf)
        # self.constants = n_rs.gamma(g_shape, g_scale, [a_size, 1])
        init_constants = np.random.normal(means[:,None], deviation[:,None], [a_size, 1])
        self.constants[itf] = init_constants.copy()
        self.init_constants[itf] = init_constants.copy()
        self.bounds = np.array([(1e-8, None) for i in self.constants])

    def _create_rates_vector(self):
        rates = np.zeros([self.interaction_network.shape[1], 1])
        decay_start = (self.np * 2) + (self.ns - self.np - 1) + 1
        rates[0] = (self.constants[0] * self.concentrations[0]) / (self.constants[1] + self.concentrations[0])
        # print(decay_start)
        # this does all the synthesis reactions
        for i in range(1, self.ns + 1):
            con_sub = self.concentrations[i]
            # print('For substrate {0} we stored:'.format(con_sub - 1))
            vsi = i * 2
            vpi = (self.ns * 2) + ((i - 1) * 2)
            # print("Locations in the constants array being used (start point) are: {0}, {1}".format(vsi, vpi))
            if i != self.ns:
                # print("Substrate creation")
                # print(i)
                rates[i] = (self.constants[vsi] * con_sub) / (self.constants[vsi + 1] + con_sub)
            if i <= self.np:
                # print("Product creation")
                # print(i + self.ns - 1)
                rates[i + self.ns - 1] = (self.constants[vpi] * con_sub) / (self.constants[vpi + 1] + con_sub)

        # Does the case of having more substrates than products
        if self.ns > self.np:
            # print('hit check of subs')
            for i in range(1, 1 + (self.ns - self.np)):
                # Degregation rate of the substrate
                end_pos = i * -1
                rates[end_pos] = (self.constants[end_pos] * self.concentrations[self.ns + 1 + end_pos])

        # print(rates)
        # print("Done with first loop")

        # This does all of the decay reactions
        for i in range(decay_start, decay_start + self.np):
            prod_id = self.ns + 1 + i - decay_start
            con_sub = self.concentrations[prod_id]
            # print('For substrate {0} we stored:'.format(con_sub - 1))
            vsi = (decay_start * 2) + (i - decay_start) - (self.ns - self.np)
            # print(i)
            # print("Using the variables at")
            # print(vsi)
            # print(self.constants[vsi])
            # print(con_sub)
            rates[i] = (self.constants[vsi] * con_sub)
        # print(rates)
        # print(self.interaction_network.shape)
        return rates

    def fitting_derivatives(self, t, y0):
        self.concentrations = y0.reshape(len(y0), 1)
        return self.interaction_network.dot(self._create_rates_vector()).flatten()

    def model_derivatives(self):
        return self.interaction_network.dot(self._create_rates_vector())

    def model_step(self):
        # print(self.concentrations)
        self.concentrations += self.model_derivatives()
        # print(self.concentrations)

    def run_simulation(self):
        for i in range(0, 1000):
            self.model_step()

    def print_shapes(self):
        print(self.interaction_network.shape)
        # print(self.interaction_network)
        print(self.constants.shape)
        # print(self.constants)
        print(self.concentrations.shape)
        # print(self.concentrations)

    def print_setup_info(self):
        print('The interaction network for {0} extra sub and {1} extra product is'.format(self.ns, self.np))
        print(self.interaction_network.shape)
        print(self.interaction_network)
        print('The concentrations for the network are:')
        print(self.concentrations.shape)
        print(self.concentrations)
        print('The constants for the network are:')
        print(self.constants.shape)
        print(self.constants)

    def _reset_model(self):
        self.constants = self.init_constants.copy()
        self.concentrations = self.initial_conditions

    def _reset_concentrations(self):
        self.concentrations = self.initial_conditions

    def set_constants(self, params):
        self.constants[self.parameters_to_fit, 0] = params

    def get_constants(self):
        return self.constants[self.parameters_to_fit, 0].flatten()

    def objective_function(self):
        loss = 1
        return loss

    def get_initial_conditions(self):
        return self.initial_conditions


def main():
    global n_rs, s_rs
    n_rs, s_rs = set_seeds(42)
    n_s = 1
    n_p = 0
    ic = np.array([i for i in range(1, n_s + n_p + 2)]).astype('float64')
    ic = ic.reshape([ic.shape[0], 1])
    const_subs = []
    test_mm = ChainMichaelisMentenModel(n_s, n_p, const_subs, ic)
    test_mm.print_shapes()
    test_mm.model_step()


if __name__ == "__main__":
    main()
