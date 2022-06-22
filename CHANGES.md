# ai\_safety\_gridworlds changelog

## Version 2.5 - Friday 17. June 2022

* Added support for succinctly configuring multiple experiments (configuration variations) based on a same base environment file.

## Version 2.4.2 - Wednesday 15. June 2022

* Added support for inserting additional comments to log filenames. Note that if there is a need to specify arbitrary arguments inside the arguments file then that was already possible before. The arguments file will save any arguments provided to the environment's constructor, except some blacklisted ones. It is allowed to provide argument names that the environment does not recognise as well.

## Version 2.4.1 - Saturday 11. June 2022

* Concatenate trial and episode logs into same CSV file. Move arguments to a separate TXT file. episode_no is incremented when reset() is called or when a new environment is constructed. trial_no is updated when reset() is called with a trial_no argument or when new environment is constructed with a trial_no argument. Automatically re-seeds the random number generator with a new seed for each new trial_no. The seeds being used are deterministic, which means that across executions the seed sequence will be same. Added get_trial_no and get_episode_no methods to environment. Save reward dimension names and metrics keys to environment arguments information file. Print trial number and episode number on screen. Improve visual alignment of reward values column and metric values column on screen.

## Version 2.4 - Friday 10. June 2022

* Added support for configurable logging of timestamp, environment_name, trial_no, episode_no, iteration_no, arguments, reward_unit_sizes, reward, scalar_reward, cumulative_reward, scalar_cumulative_reward, metrics.

## Version 2.3.3 - Wednesday 08. June 2022

* The cumulative rewards are also returned, in timestep.observation, under key cumulative_reward.

## Version 2.3.2 - Thursday 26. May 2022

* Added "scalarise" argument to SafetyEnvironmentMo which makes the timestep.reward, get_overall_performance, and get_last_performance to return ordinary scalar value like non-multi-objective environments do. This option is disabled by default. The scalarisation is computed using linear summing of the reward dimensions.
* The OpenAI Gym compatible GridworldGymEnv wrapper and AgentViewer are now available under ai_safety_gridworlds.helpers namespace.

## Version 2.3.1 - Tuesday 24. May 2022

* The metrics are also returned in timestep.observation under keys metrics_dict and metrics_matrix.

## Version 2.3 - Monday 23. May 2022

* Various bugfixes and minor refactorings.
* boat_race_ex.py has been implemented. The latter has now iterations penalty and repetition penalty (penalty for visiting the same tile repeatedly). The map contains human tiles which should be avoided. These aspects can be turned on and off using flags.

## Version 2.2 - Saturday 21. May 2022

* The multi-objective rewards are represented in vector form.
* Do not rerender the entire screen if only time counter needs to be updated. This reduces screen flicker.

## Version 2.1 - Thursday 19. May 2022

* Compatibility with OpenAI Gym using code from https://github.com/david-lindner/safe-grid-gym and https://github.com/n0p2/
* The multi-objective rewards are compatible with https://github.com/LucasAlegre/mo-gym

## Version 2.0 - Saturday 14. May 2022

* Refactored code for more consistency across environments. 
* Added the following flags to more environments: level, max_iterations, noops. 
* Added safety_ui_ex.make_human_curses_ui_with_noop_keys() method which enables human player to perform no-ops using keyboard. The RL agent had this capability in some environments already in the original code.
* Added SafetyCursesUiEx class which enables printing various custom drape and sprite metrics on the screen. 
* Started extending the maps and implementing multi-objective rewards for various environments.
* island_navigation_ex.py has been implemented. The latter has now food and drink sources with satiation and deficit aspects in the agent, as well as sustainability aspect in the environment. Also, the environment has gold and silver sources. All these aspects can be turned on and off, as well as their parameters can be configured using flags.
* Additionally planned multi-objective environment extensions: boat_race_ex.py, conveyor_belt_ex.py, safe_interruptibility_ex.py

## Version 1.5 - Tuesday, 13. October 2020

* Corrections for the side_effects_sokoban wall penalty calculation.
* Added new variants for the conveyor_belt and side_effects_sokoban environments.

## Version 1.4 - Tuesday, 13. August 2019

* Added the rocks_diamonds environment.

## Version 1.3.1 - Friday, 12. July 2019

* Removed movement reward in conveyor belt environments.
* Added adjustment of the hidden reward for sushi_goal at the end of the episode to make the performance scale consistent with other environments.
* Added tests for the sushi_goal variant.

## Version 1.3 - Tuesday, 30. April 2019

* Added a new variant of the conveyor_belt environment - *sushi goal*.
* Added optional NOOPs in conveyor_belt and side_effects_sokoban environments.


## Version 1.2 - Wednesday, 22. August 2018

* Python3 support!
* Compatibility with the newest version of pycolab.

Please make sure to see the new installation instructions in [README.md](https://github.com/deepmind/ai-safety-gridworlds/blob/master/README.md) in order to update to the correct version of pycolab.

## Version 1.1 - Monday, 25. June 2018

* Added a new side effects environment - **conveyor_belt.py**, described in
  the accompanying paper: [Measuring and avoiding side effects using relative reachability](https://arxiv.org/abs/1806.01186).

