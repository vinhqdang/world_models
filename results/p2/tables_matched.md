
### lorenz: compute-matched controls (seeds [0, 1, 2, 3]; SCN uses 10000 optimiser steps in total)

| method | VPT | blow-up frac | energy dist. | off-manifold | one-step RMSE |
|---|---|---|---|---|---|
| scn (10k) | 10.3 ± 1.3 | 0.0156 ± 0.05 | 0.127 ± 0.1 | 0.102 ± 0.017 | 0.0815 ± 0.017 |
| iso_10k | 17.7 ± 1.9 | 0.219 ± 0.33 | 1.48 ± 1.5 | 0.0772 ± 0.017 | 0.0374 ± 0.0031 |
| normal_10k | 17.8 ± 2.1 | 0.109 ± 0.15 | 2.02 ± 0.26 | 0.0731 ± 0.021 | 0.0344 ± 0.0028 |
| ss_10k | 35.1 ± 2.3 | 0.797 ± 0.39 | 0.802 ± 6.1 | 0.796 ± 6.3 | 0.0498 ± 0.0036 |

- paired VPT scn - iso_10k: -7.4 ± 3.0 (n=4)

- paired VPT scn - normal_10k: -7.5 ± 2.3 (n=4)

- paired VPT scn - ss_10k: -24.8 ± 2.5 (n=4)

### vdp: compute-matched controls (seeds [0, 1, 2, 3]; SCN uses 10000 optimiser steps in total)

| method | VPT | blow-up frac | energy dist. | off-manifold | one-step RMSE |
|---|---|---|---|---|---|
| scn (10k) | 487 ± 5.9e+02 | 0 ± 0 | 0.00525 ± 0.0091 | 0.00636 ± 0.00091 | 0.0021 ± 0.00031 |
| iso_10k | 150 ± 1.6e+02 | 0 ± 0 | 0.0097 ± 0.0061 | 0.00429 ± 0.00031 | 0.00158 ± 0.00019 |
| normal_10k | 154 ± 70 | 0 ± 0 | 0.00879 ± 0.01 | 0.00406 ± 0.00046 | 0.00138 ± 0.00016 |
| ss_10k | 431 ± 2e+02 | 0 ± 0 | 0.00301 ± 0.0051 | 0.0112 ± 0.0036 | 0.00238 ± 0.00031 |

- paired VPT scn - iso_10k: +337.7 ± 616.0 (n=4)

- paired VPT scn - normal_10k: +333.1 ± 583.1 (n=4)

- paired VPT scn - ss_10k: +56.0 ± 596.7 (n=4)