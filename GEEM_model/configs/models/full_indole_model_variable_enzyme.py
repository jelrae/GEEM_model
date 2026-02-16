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

K_CAT = np.ones((len(REACT_DICT), 1))
K_M = np.ones((len(REACT_DICT), 1))

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