| arm | runs | episodes | success | fall | timeout |
|---|---|---|---|---|---|
| crn_chain3 | 3 | 161 | 0.839 [0.776, 0.894] | 0.031 [0.006, 0.062] | 0.130 [0.081, 0.186] |
| dir_oa | 3 | 142 | 0.803 [0.739, 0.866] | 0.028 [0.007, 0.056] | 0.169 [0.113, 0.225] |
| indep | 3 | 141 | 0.738 [0.667, 0.809] | 0.085 [0.043, 0.135] | 0.177 [0.113, 0.241] |

Paired vs `indep` (matched (model, seed, slot, episode-index) episodes; McNemar exact on success, and on fall):

| arm | n matched | success diff | McNemar p (succ) | fall diff | McNemar p (fall) | timeout diff |
|---|---|---|---|---|---|---|
| crn_chain3 | 131 | +0.076 | 0.076 | -0.053 | 0.092 | -0.023 |
| dir_oa | 127 | +0.063 | 0.243 | -0.055 | 0.092 | -0.008 |

Paired vs `crn_chain3` (matched (model, seed, slot, episode-index) episodes; McNemar exact on success, and on fall):

| arm | n matched | success diff | McNemar p (succ) | fall diff | McNemar p (fall) | timeout diff |
|---|---|---|---|---|---|---|
| dir_oa | 138 | -0.007 | 1.000 | -0.007 | 1.000 | +0.014 |
| indep | 131 | -0.076 | 0.076 | +0.053 | 0.092 | +0.023 |
