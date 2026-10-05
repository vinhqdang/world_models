# Verification of paper/refs.bib

Date of verification: 2026-10-05. Entries: 73. Verified: 73. Unverified (marked `UNVERIFIED`): 0.

## Method

- arXiv entries (65): every ID was looked up with the arXiv API (`id_list`); title, first author, author list and submission year in refs.bib are taken from the API record and were re-checked against it after the file was generated. The arXiv API returned no `journal_ref` for any of them.
- Non-arXiv entries and published versions (11 DOIs): retrieved from the Crossref API (title, authors, venue, volume, issue, pages, year). Hoeffding Races (no DOI) was checked on the NeurIPS proceedings page metadata (papers.nips.cc, 1993, vol. 6).
- Published versions are used only where a record was retrieved: DreamerV3 (Nature 640, 2025), Engression (JRSS-B 87(3), 2025), Particle MPC (IROS 2021), Janson-Schmerling-Pavone (ISRR 2015, Springer 2017). These keep their arXiv eprint field. All other arXiv-only entries are cited as `@misc` preprints. Venue versions of PlaNet, Dreamer, DreamerV2, PETS, MOPO, IRIS, DIAMOND, TD-MPC2, DINO-WM, ACI, Gao et al. and others exist (ICML/ICLR/NeurIPS) but were not retrieved, so they are not claimed. Replace them before camera-ready.
- Content claims in related.tex about close prior work were checked against full text (pdftotext of arxiv.org/pdf): VJEPA (2601.14354, Sec. 6.4-6.7 and Thm. 3, experiments are a linear-Gaussian toy), Branch-JEPA (2607.05238, energy-score training, Argoverse 2 offline forecasting, "not closed-loop control metrics"), Flow-JEPA (2608.29029, LeWM baseline, CEM with 300 samples, one fresh generated sample per candidate, terminal latent cost), JEDI (2605.13013, online MBRL without look-ahead planning), FlowWM (2606.29059, forecasting/perception benchmarks, no planning), Li et al. (2607.23602, proposal overgeneration), D-JEPA (2609.24749, expected-value and CVaR controls with 16 sampled futures per candidate). Other entries were characterised from their arXiv abstracts only.

## Unverified entries

None.

## IDs in the task description: checks

All 39 listed arXiv IDs exist and match the intended papers. Points to note (no ID was wrong):

- 2609.30036 is titled "Aim Short to Reach Far: Your Frozen World Model Can Plan Better Than You Think"; "Anchored Planning (AP)" is the method name inside it.
- 2605.22164 (TRM) is "World Model Control by Trajectory Reachability Metrics", first author Liangyu Li.
- 2607.23602 (Li et al.) is "Action from Adjacent Set in Physical Space Outperforms the Best Prediction in World Models" (names the effect "proposal overgeneration"; method ASAR); same first author as TRM.
- 2607.05238 has title Branch-JEPA in v3; v2 was titled "MoP-JEPA: Hard-Assigned Predictor Mixtures for Stochastic JEPA World Models". Cite as Branch-JEPA, but expect searches to turn up the old title.
- 2608.00591 (Dong) is "Why Does the Future Branch? Identifiable Closure Tests for Stochastic Physical World Models", single author Yibin Dong.
- 2609.05461 (ARC-Bench) has submission date 2026-08-12 although its ID is 2609 (ID/date mismatch on the arXiv side; the record is valid).
- 1609.05399 (Schmerling & Pavone) is "Evaluating Trajectory Collision Probability through Adaptive Importance Sampling for Safe Motion Planning". The Janson-Schmerling-Pavone paper is a separate arXiv entry, 1504.08053, cited via its published Springer version.
- SIGReg is introduced in LeJEPA (2511.08544, Balestriero and LeCun).
- Particle MPC is Dyro, Harrison, Sharma, Pavone, arXiv 2104.02213 (IROS 2021). The "risk-aware MPPI with CVaR" paper 2209.12842 is Yin, Zhang, Tsiotras.
- Ranking and selection / common random numbers: cited Kim and Nelson 2001 (ACM TOMACS), Maron and Moore 1993 (Hoeffding races) and Glasserman and Yao 1992 (Management Science), all retrieved.

## Close prior work the authors should read and cite carefully

1. Var-JEPA (2603.20111), D-JEPA (2609.24749): D-JEPA's baseline table implements "expected-value planning" and "CVaR planning" over 16 sampled futures per candidate from a Var-JEPA-inspired stochastic JEPA, with scores 78.9 vs 79.7 (PushT) and 74.2 vs 75.8 (Reacher): a CVaR-versus-mean comparison already exists, with no gain from CVaR.
2. VJEPA (2601.14354): writes the stochastic MPC objective, the sampling-based latent MPC algorithm, and Theorem 3 (mean planning is optimal under quadratic cost with action-independent covariance). Any theory about "when mean planning suffices" must be positioned against this theorem.
3. Flow-JEPA (2608.29029): generative predictor on LeWM planned with CEM; it scores one sample per candidate (as described), so particle-averaging is the difference.
4. Branch-JEPA (2607.05238): energy-score-trained JEPA predictor, but offline forecasting only.
5. Li et al. 2607.23602: proposal overgeneration (larger proposal budget lowers feasibility of the min-cost candidate) overlaps directly with any claim about planning-budget optimism or selection bias; the optimizer's-curse framing is the distinguishing link, not the phenomenon.
6. 2605.15960 (Imperfect World Models are Exploitable), 2610.00921 (In CEM, a World Model Is Also a Proposal Mechanism, 2026-10-01) are adjacent on exploitation and CEM selection error.

Seen but not cited (exist per arXiv API, optional): 2608.14125 Traj-LeWM, 2609.03294 LEAP, 2608.29998 The Intervention Gap, 2608.24855 LeFlow, 2609.11445 FARM, 2605.04732 (common random numbers for rollout planning, not latent world models).

## Search coverage behind the "to our knowledge" sentence

arXiv API title/abstract/all-field queries (energy score / CRPS with world model; stochastic JEPA planning; CVaR or risk with latent world model; importance sampling and failure probability with world model; optimizer's/winner's curse with world model; common random numbers with planning; LeWM with stochastic or ensemble) and two web searches, up to 2026-10-05. The arXiv API handles phrase queries poorly (several returned zero hits), so the absence of a hit is weak evidence. The sentence in related.tex is limited to the specific combination (proper-scoring-rule predictor plus particle-averaged closed-loop planning against a matched deterministic predictor; systematic study of the listed add-ons).

## Entries
| key | arXiv id / DOI | first author | year | source |
|---|---|---|---|---|
| ha2018world | 1803.10122 | David Ha | 2018 | arXiv API |
| hafner2019planet | 1811.04551 | Danijar Hafner | 2018 | arXiv API |
| hafner2020dreamer | 1912.01603 | Danijar Hafner | 2019 | arXiv API |
| hafner2021dreamerv2 | 2010.02193 | Danijar Hafner | 2020 | arXiv API |
| hafner2025dreamerv3 | 2301.04104 | Hafner, Danijar | 2025 | arXiv API + Crossref DOI |
| hafner2025dreamer4 | 2509.24527 | Danijar Hafner | 2025 | arXiv API |
| hansen2024tdmpc2 | 2310.16828 | Nicklas Hansen | 2023 | arXiv API |
| hansen2022tdmpc | 2203.04955 | Nicklas Hansen | 2022 | arXiv API |
| micheli2023iris | 2209.00588 | Vincent Micheli | 2022 | arXiv API |
| alonso2024diamond | 2405.12399 | Eloi Alonso | 2024 | arXiv API |
| janner2019mbpo | 1906.08253 | Michael Janner | 2019 | arXiv API |
| balestriero2025lejepa | 2511.08544 | Randall Balestriero | 2025 | arXiv API |
| maes2026lewm | 2603.19312 | Lucas Maes | 2026 | arXiv API |
| zhou2024dinowm | 2411.04983 | Gaoyue Zhou | 2024 | arXiv API |
| sobal2025pldm | 2502.14819 | Vlad Sobal | 2025 | arXiv API |
| assran2025vjepa2 | 2506.09985 | Mido Assran | 2025 | arXiv API |
| terver2025jepawms | 2512.24497 | Basile Terver | 2025 | arXiv API |
| maes2026swm | 2602.08968 | Lucas Maes | 2026 | arXiv API |
| assran2023ijepa | 2301.08243 | Mahmoud Assran | 2023 | arXiv API |
| bardes2024vjepa | 2404.08471 | Adrien Bardes | 2024 | arXiv API |
| song2026branchjepa | 2607.05238 | Zhi Song | 2026 | arXiv API |
| huang2026vjepa | 2601.14354 | Yongchao Huang | 2026 | arXiv API |
| huo2026flowjepa | 2608.29029 | Yanchen Huo | 2026 | arXiv API |
| porcher2026flowwm | 2606.29059 | Francois Porcher | 2026 | arXiv API |
| lim2026jedi | 2605.13013 | Jing Yu Lim | 2026 | arXiv API |
| dong2026closure | 2608.00591 | Yibin Dong | 2026 | arXiv API |
| gogl2026varjepa | 2603.20111 | Moritz Gögl | 2026 | arXiv API |
| singh2026objective | 2608.12959 | Joyjeet Singh | 2026 | arXiv API |
| liu2026anchored | 2609.30036 | Xvyuan Liu | 2026 | arXiv API |
| li2026trm | 2605.22164 | Liangyu Li | 2026 | arXiv API |
| bai2026tdjepa | 2607.25337 | Jiaxin Bai | 2026 | arXiv API |
| cheng2026sage | 2607.17973 | Letian Cheng | 2026 | arXiv API |
| alrasheed2026limits | 2609.39235 | Ali Alrasheed | 2026 | arXiv API |
| zhang2026arcbench | 2609.05461 | Zhengshu Zhang | 2026 | arXiv API |
| wang2026decisionmetric | 2608.18746 | Jiawei Wang | 2026 | arXiv API |
| liu2026djepa | 2609.24749 | Shuaijun Liu | 2026 | arXiv API |
| fu2026cgs | 2609.35603 | Ziang Fu | 2026 | arXiv API |
| chen2026atm | 2606.09028 | Jiaheng Chen | 2026 | arXiv API |
| li2026overgeneration | 2607.23602 | Liangyu Li | 2026 | arXiv API |
| bhamidipaty2026exploitable | 2605.15960 | Logan Mondal Bhamidipaty | 2026 | arXiv API |
| chua2018pets | 1805.12114 | Kurtland Chua | 2018 | arXiv API |
| yu2020mopo | 2005.13239 | Tianhe Yu | 2020 | arXiv API |
| yin2022ramppi | 2209.12842 | Ji Yin | 2022 | arXiv API |
| dyro2021pmpc | 2104.02213 | Dyro, Robert | 2021 | arXiv API + Crossref DOI |
| williams2016mppi | 1509.01149 | Grady Williams | 2015 | arXiv API |
| lakshminarayanan2017ensembles | 1612.01474 | Balaji Lakshminarayanan | 2016 | arXiv API |
| nath2026sls | 2606.15594 | Devesh Nath | 2026 | arXiv API |
| enwerem2026riskbelief | 2604.03868 | Clinton Enwerem | 2026 | arXiv API |
| janson2017mcmp | 1504.08053 | Janson, Lucas | 2017 | arXiv API + Crossref DOI |
| schmerling2017adaptiveis | 1609.05399 | Edward Schmerling | 2016 | arXiv API |
| shen2025engression | 2307.00835 | Shen, Xinwei | 2025 | arXiv API + Crossref DOI |
| lang2024aifscrps | 2412.15832 | Simon Lang | 2024 | arXiv API |
| wang2026adajepa | 2606.32026 | Ying Wang | 2026 | arXiv API |
| zhang2026jepattt | 2610.00722 | Zheyuan Zhang | 2026 | arXiv API |
| soni2026sandwich | 2609.21740 | Krishnam Soni | 2026 | arXiv API |
| sun2020ttt | 1909.13231 | Yu Sun | 2019 | arXiv API |
| wang2021tent | 2006.10726 | Dequan Wang | 2020 | arXiv API |
| cao2026worldmodelslie | 2609.34300 | John Cao | 2026 | arXiv API |
| lekeufack2024cdt | 2310.05921 | Jordan Lekeufack | 2023 | arXiv API |
| gibbs2021aci | 2106.00170 | Isaac Gibbs | 2021 | arXiv API |
| gao2023overopt | 2210.10760 | Leo Gao | 2022 | arXiv API |
| khalaf2025inferencehacking | 2506.19248 | Hadi Khalaf | 2025 | arXiv API |
| zidan2026survey | 2606.00133 | Arif Hassan Zidan | 2026 | arXiv API |
| hou2026robotsurvey | 2605.00080 | Bohan Hou | 2026 | arXiv API |
| huang2024safedreamer | 2307.07176 | Weidong Huang | 2023 | arXiv API |
| gneiting2007scoring | 10.1198/016214506000001437 | Gneiting, Tilmann | 2007 | Crossref API |
| smith2006optimizers | 10.1287/mnsc.1050.0451 | Smith, James E. | 2006 | Crossref API |
| rubinstein1999cem | 10.1023/A:1010091220143 | Rubinstein, Reuven | 1999 | Crossref API |
| rockafellar2000cvar | 10.21314/JOR.2000.038 | Rockafellar, R. Tyrrell | 2000 | Crossref API |
| efron1973stein | 10.1080/01621459.1973.10481350 | Efron, Bradley | 1973 | Crossref API |
| kim2001fully | 10.1145/502109.502111 | Kim, Seong-Hee | 2001 | Crossref API |
| glasserman1992crn | 10.1287/mnsc.38.6.884 | Glasserman, Paul | 1992 | Crossref API |
| maron1993hoeffding | NeurIPS proceedings page | Maron, Oded | 1993 | papers.nips.cc page metadata |
