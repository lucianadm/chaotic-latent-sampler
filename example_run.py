import numpy as np
from chaotic_sampler import chaotic_latent_sampler

import numpy as np

# Example: Rössler latent data
data = np.loadtxt("rossler_trajectory.txt")

#Single-seed
res = chaotic_sampler(data, mode="single")
XX_single = res["XX"]

#Multi-seed
#res = chaotic_sampler(data, mode="multi")
#SAL2 = res["SAL2"]
#mat_nueva = np.transpose(SAL2, (1, 2, 0))
#XX_multi = mat_nueva[:, :, 1]  