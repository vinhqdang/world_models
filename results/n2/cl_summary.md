| arm | runs | episodes | success | fall | timeout |
|---|---|---|---|---|---|
| crn_chain3 | 1 | 52 | 0.827 [0.731, 0.923] | 0.038 [0.000, 0.096] | 0.135 [0.058, 0.231] |
| dir_oa | 1 | 45 | 0.778 [0.644, 0.889] | 0.000 [0.000, 0.000] | 0.222 [0.111, 0.356] |
| indep | 1 | 43 | 0.767 [0.628, 0.884] | 0.047 [0.000, 0.116] | 0.186 [0.070, 0.302] |

Paired vs `indep` (matched (model, seed, slot, episode-index) episodes; McNemar exact on success, and on fall):

| arm | n matched | success diff | McNemar p (succ) | fall diff | McNemar p (fall) | timeout diff |
|---|---|---|---|---|---|---|
| crn_chain3 | 41 | +0.049 | 0.727 | -0.024 | 1.000 | -0.024 |
| dir_oa | 40 | +0.000 | 1.000 | -0.050 | 0.500 | +0.050 |

Paired vs `crn_chain3` (matched (model, seed, slot, episode-index) episodes; McNemar exact on success, and on fall):

| arm | n matched | success diff | McNemar p (succ) | fall diff | McNemar p (fall) | timeout diff |
|---|---|---|---|---|---|---|
| dir_oa | 44 | -0.045 | 0.754 | -0.023 | 1.000 | +0.068 |
| indep | 41 | -0.049 | 0.727 | +0.024 | 1.000 | +0.024 |
