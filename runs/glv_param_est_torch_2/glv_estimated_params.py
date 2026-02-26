"""
gLV parameter estimates from AD-EnKF (glv_param_est_run).
Same format as Data/gLV/glv_data_generator.py: R = growth rates, A = interaction matrix.
Convention: A[i,j] = effect of species j on species i's per-capita growth rate.
"""

import numpy as np

# Intrinsic growth rates (r_1 .. r_5)
R_EST = np.array([0.976364, 0.798143, -0.00405991, -0.235441, -0.0488877])

# Interaction matrix A[i,j] = effect of j on i (same layout as A_TRUE in glv_data_generator)
# fmt: off
A_EST = np.array([
    [ -0.3940,   0.0000,  -0.2194,   0.0000,   0.0000],  # prod1:    self-reg; loss to herb3
    [  0.0000,  -0.3345,   0.0000,  -0.3718,   0.0000],  # prod2:    self-reg; loss to herb4
    [  0.2155,   0.0000,  -0.1185,   0.0000,  -0.1622],  # herb3:    gain from prod1; loss to predator
    [  0.0000,   0.2271,   0.0000,  -0.1926,  -0.1914],  # herb4:    gain from prod2; loss to predator
    [  0.0000,   0.0000,   0.2220,   0.1360,  -0.0644],  # predator: gain from herb3 and herb4
])
# fmt: on
