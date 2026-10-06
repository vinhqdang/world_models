
### lorenz (test seeds [0, 1, 2, 3, 4], mean ± 95% t-CI over seeds)

| method | VPT (steps, thr 0.3) | VPT (Lyapunov times) | blow-up frac (3000 steps) | energy dist. to attractor | off-manifold dist | one-step RMSE |
|---|---|---|---|---|---|---|
| tf | 9.43 ± 0.86 | 0.171 ± 0.016 | 1 ± 0 | n/a | n/a | 0.0325 ± 0.0017 |
| tf+aeproj | 10.2 ± 1.2 | 0.184 ± 0.022 | 0.775 ± 0.29 | 4.49 ± 1.6 | 0.52 ± 1 | 0.0325 ± 0.0017 |
| iso | 16 ± 1.4 | 0.29 ± 0.026 | 0.212 ± 0.26 | 2.67 ± 1.9 | 0.104 ± 0.035 | 0.0442 ± 0.0023 |
| iso+aeproj | 11.5 ± 0.97 | 0.209 ± 0.018 | 0.025 ± 0.069 | 0.585 ± 0.8 | 0.178 ± 0.059 | 0.0442 ± 0.0023 |
| normal | 15.7 ± 0.99 | 0.284 ± 0.018 | 0.25 ± 0.29 | 1.75 ± 0.89 | 0.0959 ± 0.018 | 0.0418 ± 0.0022 |
| ss | 25.3 ± 1.8 | 0.459 ± 0.033 | 0.725 ± 0.44 | 3.09 ± 11 | 2.07 ± 5.1 | 0.0644 ± 0.0022 |
| scn | 11.3 ± 2.8 | 0.204 ± 0.051 | 0.0125 ± 0.035 | 0.216 ± 0.26 | 0.0984 ± 0.015 | 0.0772 ± 0.017 |
| noisy_oracle@tf_eps | 38.9 ± 1.6 | - | n/a | n/a | n/a | n/a |
| noisy_oracle@tf_eps/2 | 60.8 ± 6.3 | - | n/a | n/a | n/a | n/a |

Paired VPT differences (seed-matched), scn minus X:

- scn - tf: +1.84 ± 2.32 (n=5)
- scn - iso: -4.71 ± 2.91 (n=5)
- scn - normal: -4.39 ± 2.70 (n=5)
- scn - ss: -14.08 ± 4.63 (n=5)
- scn - iso+aeproj: -0.24 ± 2.78 (n=5)

### vdp (test seeds [0, 1, 2, 3, 4], mean ± 95% t-CI over seeds)

| method | VPT (steps, thr 0.3) | VPT (Lyapunov times) | blow-up frac (3000 steps) | energy dist. to attractor | off-manifold dist | one-step RMSE |
|---|---|---|---|---|---|---|
| tf | 42.3 ± 11 | - | 1 ± 0 | n/a | n/a | 0.00405 ± 0.00031 |
| tf+aeproj | 22.6 ± 4.4 | - | 0 ± 0 | 0.0296 ± 0.026 | 0.0928 ± 0.044 | 0.00405 ± 0.00031 |
| iso | 228 ± 1.7e+02 | - | 0 ± 0 | 0.00574 ± 0.0073 | 0.0111 ± 0.0019 | 0.0039 ± 0.00049 |
| iso+aeproj | 25.6 ± 5.3 | - | 0 ± 0 | 0.00693 ± 0.012 | 0.051 ± 0.0097 | 0.0039 ± 0.00049 |
| normal | 451 ± 4.3e+02 | - | 0 ± 0 | 0.00287 ± 0.0055 | 0.0117 ± 0.002 | 0.00378 ± 0.00053 |
| ss | 267 ± 1.6e+02 | - | 0 ± 0 | 0.00437 ± 0.0077 | 0.0347 ± 0.022 | 0.00525 ± 0.00033 |
| scn | 436 ± 4.2e+02 | - | 0 ± 0 | 0.00385 ± 0.0073 | 0.00623 ± 0.00071 | 0.00205 ± 0.00025 |
| noisy_oracle@tf_eps | 90.8 ± 19 | - | n/a | n/a | n/a | n/a |
| noisy_oracle@tf_eps/2 | 224 ± 64 | - | n/a | n/a | n/a | n/a |

Paired VPT differences (seed-matched), scn minus X:

- scn - tf: +394.01 ± 432.80 (n=5)
- scn - iso: +208.03 ± 452.67 (n=5)
- scn - normal: -15.15 ± 346.10 (n=5)
- scn - ss: +169.72 ± 415.43 (n=5)
- scn - iso+aeproj: +410.75 ± 423.28 (n=5)