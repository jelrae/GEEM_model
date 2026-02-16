from scipy.optimize import Bounds


def bounds_prior_generator(PARAMS_TO_FIT):
    lbs = []
    ubs = []
    priors = []
    for name, params in PARAMS_TO_FIT.items():
        num_params = len(params[0])
        if type(params[1]) == tuple:
            lbs += [params[1][0] for _ in range(num_params)]
            ubs += [params[1][1] for _ in range(num_params)]
        elif type(params[1]) == list:
            lbs += params[1][0]
            ubs += params[1][1]
        else:
            raise TypeError
        priors += [params[2] for _ in range(num_params)]

    bounds = Bounds(
        lb=lbs,
        ub=ubs
    )

    return bounds, priors
