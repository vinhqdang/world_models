# Environment used for Stages 0-2

- Machine: 4 CPU cores, 15 GB RAM, no GPU, Linux 6.18 (x86_64); no EGL / display (rendering is not used: state level only).
- Python 3.11.18; numpy 2.4.6; scipy 1.17.1; torch 2.15.0+cpu (torch.set_num_threads(2) in all fits, at most two heavy jobs
  in parallel apart from the single-threaded simulator jobs).
- Simulators (the authors' pins from requirements.txt where installable): mujoco 3.11.0, dm_control 1.0.44, gymnasium 1.2.3,
  gymnasium-robotics 1.4.2 (as pinned), stable-baselines3 2.9.2 and huggingface_hub 2.4.0 (Ball-in-Cup behaviour policy: the
  pinned checkpoint qgallouedec/ddpg-BallInCupDMC-v0 at revision d4cab42 downloaded and loaded unchanged), pyyaml 6.0.1.
  numpy is 2.4.6 rather than the authors' pin; HandReach data nevertheless reproduce the paper's published oracle / ideal
  residual / copy values to the printed digit at the default population sizes (STAGE0.md).
- Upstream code: /home/user/cwm_upstream, https://github.com/galilai-group/counterfactual-WM, commit 5c84bba92b2ff19d41175070a61238fa1396d10d
  (MIT licence), cloned outside the repository and imported through cwk/upstream.py; not modified. cwk/sim.py subclasses its
  NativeSimulator; cwk/worlds.py re-implements cwm/sims/population.py (with exact MjData snapshots) and cwk/metrics.py follows
  cwm/metrics.py (credited in the module docstrings).
- Not installed / not used: SoftGym (PassWater), highway-env (Racetrack), pymunk/LeRobot (Push-T), BlueROV2 simulator: not
  needed for Stages 0-2 (HandReach, Ball-in-Cup and FetchReach only). No Colab GPU was used.
