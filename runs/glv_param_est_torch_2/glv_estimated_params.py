"""
gLV parameter estimates from AD-EnKF (glv_param_est_run).
Same format as Data/gLV/glv_data_generator.py: R = growth rates, A = interaction matrix.
Convention: A[i,j] = effect of species j on species i's per-capita growth rate.
"""

import numpy as np

# Intrinsic growth rates (r_1 .. r_5)
R_EST = np.array([0.965295, 0.812981, -0.0256479, -0.264681, -0.0730122])

# Interaction matrix A[i,j] = effect of j on i (same layout as A_TRUE in glv_data_generator)
# fmt: off
A_EST = np.array([
    [ -0.4115,   0.0000,  -0.2135,   0.0000,   0.0000],  # prod1:    self-reg; loss to herb3
    [  0.0000,  -0.3293,   0.0000,  -0.3067,   0.0000],  # prod2:    self-reg; loss to herb4
    [  0.1874,   0.0000,  -0.0840,   0.0000,  -0.2366],  # herb3:    gain from prod1; loss to predator
    [  0.0000,   0.2108,   0.0000,  -0.2044,  -0.1813],  # herb4:    gain from prod2; loss to predator
    [  0.0000,   0.0000,   0.2115,   0.1926,  -0.1212],  # predator: gain from herb3 and herb4
])
# fmt: on
