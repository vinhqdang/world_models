# LeWM planning-budget measurements (TwoRoom, released checkpoint quentinll/lewm-tworooms)

CEM, horizon 5 blocks x frameskip 5, elite fraction 0.1, 30 iterations, 50 episodes, 1 seed.
Success is the benchmark's own criterion (proprio distance to goal below a threshold of roughly 16 units).

| goal offset | N=30 | N=100 | N=300 | N=1000 |
|---|---|---|---|---|
| 25 | 76% | 92% | 94% | - |
| 50 | 46% | 54% | 54% | 52% |
| 100 | 10% | 12% | 12% | not completed |

Observations (single seed, wide intervals; treat as indicative):

- Success saturates by N of about 100 and does not decrease at larger N.
- At offset 100 the mean true progress per executed chunk (proprio distance to goal, before minus after) is about zero
  for every N, so the controller is stuck regardless of search effort.
- The model-predicted terminal latent cost of the executed plan is far below the realised one
  (predicted about 60-140, realised about 330-350), i.e. the predicted latent progress is overstated roughly 6-10 times,
  while its rank correlation with true progress across rounds is high (0.7-0.9 at offset 25, small n).
- Conclusion: on this benchmark the limit is the planning objective / horizon, not the amount of search.
  This is consistent with arXiv:2608.12959 and arXiv:2609.30036.
