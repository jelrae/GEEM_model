import numpy as np
# Setup the skeleton of the models

# Starting with the metabolites, enzymes involved and the reactions

META_DICT = {0: '3-indolylmthyl GLS glucobrassicin',
             1: '1-hydroxy-3-indolylmethyl GSL',
             2: '1-methoxy-3-indolylmethyl GSL',
             3: '4-hydroxy-3-indolylmethyl GSL',
             4: '4-methoxy-3-indolylmethyl GSL'
             }

ENZ_DICT = {
    0: 'CYP81F1',
    1: 'CYP81F2',
    2: 'CYP81F3',
    3: 'CYP81F4',
    4: 'IGMT1',
    5: 'IGMT2',
    6: 'IGMT5'
}

ENZ_CONC = np.ones((len(ENZ_DICT), 1))

# Reaction Dictionary defines the reactions for MM dynamics.
# key is reaction number, value is a set containing substrate num, product num, enzyme num
REACT_DICT = {
    0: (0, 1, 0),
    1: (0, 3, 0),
    2: (0, 1, 1),
    3: (0, 3, 1),
    4: (0, 1, 2),
    5: (0, 3, 2),
    6: (0, 1, 3),
    7: (1, 2, 4),
    8: (3, 4, 4),
    9: (1, 2, 5),
    10: (3, 4, 5),
    11: (1, 2, 6),
}

# From the AI predictions in 1/hr
kc_ai = [250.77, 245.82, 1139.04, 928.03, 563.13, 429.62, 1039.98, 195.11, 187.05, 295.31, 283.12, 260.32]
# In units of micro moles, should be in the same range as the metabolites themselves according to evolution
km_ai = [5.3, 5.30, 6.36, 6.36, 6.09, 6.09, 5.93, 42.83, 114.10, 38.83, 130.10, 50.68]


K_CAT = np.array(kc_ai)[:, None]
K_M = np.array(km_ai)[:, None]

KDM = np.ones((len(META_DICT), 1))

# Then we look at the mRNA which are used and the reactions which they are used in

MRNA_DICT = {
    0: 'CYP81F1',
    1: 'CYP81F2',
    2: 'CYP81F3',
    3: 'CYP81F4',
    4: 'IGMT1',
    5: 'IGMT2',
    6: 'IGMT5',
}

MRNA_REACT_DICT = {  # mrna to enzyme
    0: (0, 0),
    1: (1, 1),
    2: (2, 2),
    3: (3, 3),
    4: (4, 4),
    5: (5, 5),
    6: (6, 6),
}

KSE = np.ones((len(ENZ_DICT), 1))
KDE = np.ones((len(ENZ_DICT), 1))
VS = np.ones((len(ENZ_DICT), 1))

fit_VES = True
constrain_KDM = True