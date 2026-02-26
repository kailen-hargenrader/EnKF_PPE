"""
gLV parameter estimates from AD-EnKF (glv_param_est_run).
Same format as Data/gLV/glv_data_generator.py: R = growth rates, A = interaction matrix.
Convention: A[i,j] = effect of species j on species i's per-capita growth rate.
"""

import numpy as np

# Intrinsic growth rates (r_1 .. r_5)
R_EST = np.array([1.19812, 1.11081, -0.00998604, -0.346596, -0.245295])

# Interaction matrix A[i,j] = effect of j on i (same layout as A_TRUE in glv_data_generator)
# fmt: off
A_EST = np.array([
    [ -0.4791,   0.0000,  -0.2942,   0.0000,   0.0000],  # prod1:    self-reg; loss to herb3
    [  0.0000,  -0.4730,   0.0000,  -0.8792,   0.0000],  # prod2:    self-reg; loss to herb4
    [  0.0480,   0.0000,  -0.1013,   0.0000,  -0.4399],  # herb3:    gain from prod1; loss to predator
    [  0.0000,   0.1366,   0.0000,  -0.5957,  -0.3136],  # herb4:    gain from prod2; loss to predator
    [  0.0000,   0.0000,   0.2115,   0.3128,  -0.2347],  # predator: gain from herb3 and herb4
])
# fmt: on
