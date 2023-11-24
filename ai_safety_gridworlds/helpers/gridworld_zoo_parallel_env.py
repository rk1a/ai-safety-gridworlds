# Copyright 2022-2023 Roland Pihlakas. https://github.com/levitation-opensource/multiobjective-ai-safety-gridworlds
# Copyright 2018 n0p2 https://github.com/n0p2/gym_ai_safety_gridworlds
"""
The GridworldZooParallelEnv implements the Zoo Parallel interface for the ai_safety_gridworlds.

GridworldZooParallelEnv is derived from an GridworldGymEnv implementation by n0p2.
The original repo can be found at https://github.com/n0p2/gym_ai_safety_gridworlds
"""

import importlib
import random

import pettingzoo
from pettingzoo import ParallelEnv

import copy
import numpy as np

try:
  import gymnasium as gym
  from gymnasium import error
  from gymnasium.utils import seeding
  gym_v26 = True
except:
  import gym
  from gym import error
  from gym.utils import seeding
  gym_v26 = False

# from ai_safety_gridworlds.environments.shared.safety_game_mp import METRICS_DICT, METRICS_MATRIX
# from ai_safety_gridworlds.environments.shared.safety_game import EXTRA_OBSERVATIONS, HIDDEN_REWARD
from ai_safety_gridworlds.environments.shared.safety_game import HIDDEN_REWARD as INFO_HIDDEN_REWARD
from ai_safety_gridworlds.environments.shared.safety_game_ma import Actions   # used as export
from ai_safety_gridworlds.environments.shared.rl.pycolab_interface_ma import INFO_OBSERVATION_DIRECTION, INFO_ACTION_DIRECTION
from ai_safety_gridworlds.environments.shared import safety_game_ma
from ai_safety_gridworlds.environments.shared import safety_game_moma
from ai_safety_gridworlds.helpers import factory
from ai_safety_gridworlds.helpers.agent_viewer import AgentViewer

from pycolab import rendering

#from safe_grid_gym_orig.envs.common.interface import (
#    INFO_HIDDEN_REWARD,
#    INFO_OBSERVED_REWARD,
#    INFO_DISCOUNT,
#)
#INFO_HIDDEN_REWARD = "hidden_reward"
INFO_OBSERVED_REWARD = "observed_reward"
INFO_DISCOUNT = "discount"
INFO_OBSERVATION_COORDINATES = "info_observation_coordinates"
INFO_OBSERVATION_LAYERS_DICT = "info_observation_layers_dict"
INFO_OBSERVATION_LAYERS_CUBE = "info_observation_layers_cube"
INFO_AGENT_OBSERVATIONS = "info_agent_observations"
INFO_AGENT_OBSERVATION_COORDINATES = "info_agent_observation_coordinates"
INFO_AGENT_OBSERVATION_LAYERS_DICT = "info_agent_observation_layers_dict"
INFO_AGENT_OBSERVATION_LAYERS_CUBE = "info_agent_observation_layers_cube"


class GridworldZooParallelEnv(ParallelEnv):
    """ An Farama Foundation PettingZoo environment wrapping the AI safety gridworlds created by DeepMind.

    Parameters:
    env_name (str): defines the safety gridworld to load. can take all values
                    defined in ai_safety_gridworlds.helpers.factory._environment_classes:
                        - 'boat_race'
                        - 'boat_race_ex'
                        - 'conveyor_belt'
                        - 'conveyor_belt_ex'
                        - 'distributional_shift'
                        - 'firemaker_ex_ma'
                        - 'friend_foe'
                        - 'island_navigation'
                        - 'island_navigation_ex'
                        - 'island_navigation_ex_ma'
                        - 'rocks_diamonds'
                        - 'safe_interruptibility'
                        - 'safe_interruptibility_ex'
                        - 'side_effects_sokoban'
                        - 'tomato_watering'
                        - 'tomato_crmdp'
                        - 'absent_supervisor'
                        - 'whisky_gold'
    use_transitions (bool): If set to true the state will be the concatenation
                            of the board at time t-1 and at time t
    render_animation_delay (float): is passed through to the AgentViewer
                                    and defines the speed of the animation in
                                    render mode "human"
    """

    metadata = {"render.modes": ["human", "ansi", "rgb_array"]}

    def __init__(self, env_name, 
                 use_transitions=False, 
                 render_animation_delay=0.1, 
                 flatten_observations=False, 

                 ascii_observation_format=True, 
                 object_coordinates_in_observation=True, 
                 layers_in_observation=True, 
                 occlusion_in_layers=False, 
                 layers_order_in_cube=[], 
                 layers_order_in_cube_per_agent:dict[str, list[str]]={}, 

                 ascii_attributes_format=False, 
                 attribute_coordinates_in_observation=True, 
                 layers_in_attribute_observation=False, 
                 occlusion_in_atribute_layers=False, 
                 observable_attribute_categories=["expression", "action_direction", "observation_direction", "numeric_message", "public_metrics"], 
                 # observable_attribute_value_mapping:dict[str, dict[str, float]]={}, 
                 observable_attribute_value_mapping:dict[str, float]={},  

                 *args, **kwargs
                ):

        self._env_name = env_name
        self._render_animation_delay = render_animation_delay
        self._viewer = None
        self._env = factory.get_environment_obj(env_name, *args, **kwargs)
        self._rgb = None
        self._use_transitions = use_transitions
        self._flatten_observations = flatten_observations
        self._ascii_observation_format = ascii_observation_format
        self._object_coordinates_in_observation = object_coordinates_in_observation
        self._layers_in_observation = layers_in_observation
        self._occlusion_in_layers = occlusion_in_layers
        self._layers_order_in_cube = layers_order_in_cube
        self._layers_order_in_cube_per_agent = layers_order_in_cube_per_agent
        self._observable_attribute_categories = observable_attribute_categories
        self._observable_attribute_value_mapping = observable_attribute_value_mapping

        self._last_board = None
        self._last_observation = None
        self._last_observation_coordinates = None
        self._last_observation_layers_cube = None
        self._last_agent_observations = None
        self._last_agent_observations_coordinates = None
        self._last_agent_observations_layers_cubes = None

        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
            self._env.set_observable_attribute_categories(observable_attribute_categories, observable_attribute_value_mapping)  # TODO
            self._env.reset() # apply _observable_attribute_categories

            agents = safety_game_ma.get_players(self._env.environment_data)
            # num_agents = len(agents)
            self.possible_agents = [f"agent_{agent.character}" for agent in agents]  # TODO: make it readonly
            self.agent_name_mapping = dict(
                zip(self.possible_agents, [agent.character for agent in agents])
            )
        else:
            #if len(observable_attribute_categories) > 0:
            #    raise ValueError("observable_attribute_categories")
            num_agents = 1
            self.possible_agents = [f"agent_{r}" for r in range(0, num_agents)]  # TODO: make it readonly
            self.agent_name_mapping = dict(   # TODO: read agent char from environment
                zip(self.possible_agents, [str(r) for r in range(0, num_agents)])
            )
            
        self._state = None
        self._dones = {agent_name: False for agent_name in self.possible_agents} 

        self._last_hidden_reward = { agent: 0 for agent in self.possible_agents }

        self.action_spaces = {  # TODO: make it readonly
            agent: GridworldsActionSpace(self._env) for agent in self.possible_agents
        }  
        self.observation_spaces = {  # TODO: make it readonly
            agent: GridworldsObservationSpace(self._env, use_transitions, flatten_observations) for agent in self.possible_agents
        }

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


    @property
    def agents(self):
        return [agent for agent in self.possible_agents if not self._dones[agent]]

    @property
    def num_agents(self):
        return sum(1 for agent in self.possible_agents if not self._dones[agent])

    @property
    def max_num_agents(self):
        return len(self.possible_agents)

    @property
    def state(self):
        return self._state

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]


    def _process_observation(self, obs):

        self._last_observation = obs
        self._rgb = obs["RGB"]

        if self._object_coordinates_in_observation and hasattr(self._env, "calculate_observation_coordinates"):   # original Gridworlds environments do not support this method currently   # TODO
            self._last_observation_coordinates = self._env.calculate_observation_coordinates(obs, use_layers=not self._occlusion_in_layers, ascii=self._ascii_observation_format)

        if self._layers_order_in_cube is not None and hasattr(self._env, "calculate_observation_layers_cube"):
            self._last_observation_layers_cube = self._env.calculate_observation_layers_cube(obs, use_layers=not self._occlusion_in_layers, layers_order=self._layers_order_in_cube)

        if hasattr(self._env, "_agent_perspectives") and self._env._agent_perspectives is not None: 
            # TODO: for step() method, calculate observations and coordinates only for current agent 

            agent_observations = self._env.agent_perspectives_with_layers(obs, include_layers=not self._occlusion_in_layers, ascii=self._ascii_observation_format)
            self._last_agent_observations = { agent_name: agent_observations[agent_chr] for agent_name, agent_chr in self.agent_name_mapping.items() }

            if self._object_coordinates_in_observation:
                agent_observations_coordinates = self._env.calculate_agents_observation_coordinates(obs, agent_observations, use_layers=not self._occlusion_in_layers, ascii=self._ascii_observation_format)
                self._last_agent_observations_coordinates = { agent_name: agent_observations_coordinates[agent_chr] for agent_name, agent_chr in self.agent_name_mapping.items() }

            if self._layers_order_in_cube_per_agent is not None:
                self._last_agent_observations_layers_cubes = { agent_name: self._env.calculate_observation_layers_cube(agent_observations[agent_chr], use_layers=not self._occlusion_in_layers, layers_order=self._layers_order_in_cube_per_agent.get(agent_name, [])) for agent_name, agent_chr in self.agent_name_mapping.items() }

    #/ def _process_observation(self, obs):


    def _compute_infos(self, obs):

        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):

            infos = { 
                      agent: 
                      {
                          INFO_OBSERVATION_DIRECTION: obs.get(INFO_OBSERVATION_DIRECTION, {}).get(self.agent_name_mapping[agent]),
                          INFO_ACTION_DIRECTION: obs.get(INFO_ACTION_DIRECTION, {}).get(self.agent_name_mapping[agent]),
                      }
                      for agent in self.possible_agents
                   }

            if self._object_coordinates_in_observation and hasattr(self._env, "calculate_observation_coordinates"):
                for agent in self.possible_agents:
                    infos[agent][INFO_OBSERVATION_COORDINATES] = self._last_observation_coordinates   # shared global observation must be returned via agent keys

            if self._layers_in_observation and "layers" in obs: # only multi-objective or multi-agent environments have layers in observation available
                for agent in self.possible_agents:
                    infos[agent][INFO_OBSERVATION_LAYERS_DICT] = obs["layers"]   # shared global observation must be returned via agent keys

            if self._layers_order_in_cube is not None and hasattr(self._env, "calculate_observation_layers_cube"):
                for agent in self.possible_agents:
                    infos[agent][INFO_OBSERVATION_LAYERS_CUBE] = self._last_observation_layers_cube   # shared global observation must be returned via agent keys

            if hasattr(self._env, "_agent_perspectives") and self._env._agent_perspectives is not None:
                for agent in self.possible_agents:
                    infos[agent][INFO_AGENT_OBSERVATIONS] = self._last_agent_observations[agent].board

                    if self._layers_in_observation:
                        infos[agent][INFO_OBSERVATION_LAYERS_DICT] = self._last_agent_observations[agent].layers

                    if self._object_coordinates_in_observation:
                        infos[agent][INFO_AGENT_OBSERVATION_COORDINATES] = self._last_agent_observations_coordinates[agent]

                    if self._layers_order_in_cube_per_agent is not None:
                        infos[agent][INFO_AGENT_OBSERVATION_LAYERS_CUBE] = self._last_agent_observations_layers_cubes[agent]

        else:   # if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):

            infos = {
                agent: {
                    INFO_OBSERVATION_DIRECTION: obs.get(INFO_OBSERVATION_DIRECTION),
                    INFO_ACTION_DIRECTION: obs.get(INFO_ACTION_DIRECTION),
                }
                for agent in self.possible_agents
            }

            if self._object_coordinates_in_observation and hasattr(self._env, "calculate_observation_coordinates"):
                for agent in self.possible_agents:
                    infos[agent][INFO_OBSERVATION_COORDINATES] = self._last_observation_coordinates   # shared global observation must be returned via agent keys

            if self._layers_order_in_cube is not None and hasattr(self._env, "calculate_observation_layers_cube"):
                for agent in self.possible_agents:
                    infos[agent][INFO_OBSERVATION_LAYERS_CUBE] = self._last_observation_layers_cube   # shared global observation must be returned via agent keys

        #/ if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):


        for k, v in obs.items():
            if k not in ("board", "RGB", "layers"):
                for agent in self.possible_agents:
                    infos[agent][k] = v   # shared global observation must be returned via agent keys


        return infos

    #/ def _compute_infos(self, obs):


    def step(self, actions, *args, **kwargs):                    # CHANGED: added *args, **kwargs 
        """ Perform an action in the gridworld environment.

        Returns:
            - the board as a numpy array
            - the observed reward
            - if the episode ended
            - an info dict containing:
                - the observed reward with key INFO_OBSERVED_REWARD
                - the hidden reward with key INFO_HIDDEN_REWARD
                - the discount factor of the last step with key INFO_DISCOUNT
                - any additional information in the pycolab observation object,
                  excluding the RGB array. This includes in particular
                  the "extra_observations"
        """
        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
            action = { self.agent_name_mapping[agent_name]: agent_action for agent_name, agent_action in actions.items() }
        else:
            action = next(iter(actions.values()))   # take the single value in dict

        timestep = self._env.step(action, *args, **kwargs)      # CHANGED: added *args, **kwargs 

        obs = timestep.observation
        self._process_observation(obs)
        infos = self._compute_infos(obs)


        reward = { agent: 0.0 for agent in self.agent_name_mapping.values() } if timestep.reward is None else timestep.reward

        cumulative_hidden_reward = self._env._get_hidden_reward(default_reward=None)
        if cumulative_hidden_reward is not None:
            if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
                hidden_reward = {}
                for agent in self.possible_agents:
                    hidden_reward[agent] = cumulative_hidden_reward[self.agent_name_mapping[agent]] - self._last_hidden_reward[agent]
                    self._last_hidden_reward[agent] = cumulative_hidden_reward[self.agent_name_mapping[agent]]
            else:
                hidden_reward = cumulative_hidden_reward - self._last_hidden_reward[self.possible_agents[0]]
                self._last_hidden_reward[self.possible_agents[0]] = cumulative_hidden_reward
        else:
            hidden_reward = None


        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
            # TODO
            for agent in self.possible_agents:
                infos[agent].update({
                          INFO_HIDDEN_REWARD: hidden_reward[agent] if hidden_reward is not None else None,
                          INFO_OBSERVED_REWARD: reward[self.agent_name_mapping[agent]],
                          INFO_DISCOUNT: timestep.discount, # [agent],    # TODO: agent-based discount
                      })

        else:   # if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
            for agent in self.possible_agents:
                infos[agent].update({
                    INFO_HIDDEN_REWARD: hidden_reward,
                    INFO_OBSERVED_REWARD: reward,
                    INFO_DISCOUNT: timestep.discount,                
                })


        board = copy.deepcopy(obs["board"])   # TODO: option to return observation as character array

        # TODO: apply transitions to agent observations as well
        if self._use_transitions:
            state = np.stack([self._last_board, board], axis=0)
            self._last_board = board
        else:
            state = board[np.newaxis, :]

        # TODO: apply flatten to agent observations as well
        if self._flatten_observations:
            state = state.flatten()

        self._state = state
            

        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
            dones = { agent_name: timestep.step_type[agent_char].last() for agent_name, agent_char in self.agent_name_mapping.items() }
        else:
            dones = { agent_name: timestep.step_type.last() for agent_name in self.agent_name_mapping.keys() }

        self._dones = dones


        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):

            # TODO

            if gym_v26:
                # https://gymnasium.farama.org/content/migration-guide/
                # For users wishing to update, in most cases, replacing done with terminated and truncated=False in step() should address most issues. 
                # TODO: However, environments that have reasons for episode truncation rather than termination should read through the associated PR https://github.com/openai/gym/pull/2752
                terminateds = dones
                truncateds = {agent: False for agent in self.possible_agents}    # TODO        
                return (state, reward, terminateds, truncateds, infos)
            else:
                return (state, reward, dones, infos)

        else:   #/ if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):

            agent_name = self.possible_agents[0]
            if gym_v26:
                # https://gymnasium.farama.org/content/migration-guide/
                # For users wishing to update, in most cases, replacing done with terminated and truncated=False in step() should address most issues. 
                # TODO: However, environments that have reasons for episode truncation rather than termination should read through the associated PR https://github.com/openai/gym/pull/2752
                terminateds = dones
                truncateds = {agent: False for agent in self.possible_agents}    # TODO                
                return ({agent_name: state}, {agent_name: reward}, terminateds, truncateds, infos)
            else:
                return ({agent_name: state}, {agent_name: reward}, dones, infos)

        #/ if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):

    def reset(self, *args, **kwargs):                     # CHANGED: added *args, **kwargs
        timestep = self._env.reset(*args, **kwargs)       # CHANGED: added *args, **kwargs      

        if self._viewer is not None:
            self._viewer.reset_time()

        board = copy.deepcopy(timestep.observation["board"])   # TODO: option to return observation as character array

        obs = timestep.observation
        self._process_observation(obs)
        infos = self._compute_infos(obs)


        # TODO: apply transitions to agent observations as well
        if self._use_transitions:
            state = np.stack([np.zeros_like(board), board], axis=0)
            self._last_board = board
        else:
            state = board[np.newaxis, :]

        # TODO: apply flatten to agent observations as well
        if self._flatten_observations:
            state = state.flatten()

        self._state = state

        self._dones = {agent_name: False for agent_name in self.possible_agents} 

        obs = {agent_name: state for agent_name in self.possible_agents}    
        return obs, infos

    def get_reward_unit_space(self):                    # ADDED
        return self._env.get_reward_unit_space()

    def get_trial_no(self):                             # ADDED
        return self._env.get_trial_no()

    def get_episode_no(self):                           # ADDED
        return self._env.get_episode_no()

    # gym does not support additional arguments to .step() method so we need to use a separate method. See also https://github.com/openai/gym/issues/2399
    def set_current_q_value_per_action(self, current_agent: dict):                           # ADDED
        current_agent = next(iter(current_agent.items()))[1]
        return self._env.set_current_agent(current_agent)

    # gym does not support additional arguments to .step() method so we need to use a separate method. See also https://github.com/openai/gym/issues/2399
    #def set_current_agent(self, current_agent: dict):                           # ADDED
    #    current_agent = next(iter(current_agent.items()))[1]
    #    return self._env.set_current_agent(current_agent)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human"):
        """ Implements the gym render modes "rgb_array", "ansi" and "human".

        - "rgb_array" just passes through the RGB array provided by pycolab in each state
        - "ansi" gets an ASCII art from pycolab and returns is as a string
        - "human" uses the ai-safety-gridworlds-viewer to show an animation of the
          gridworld in a terminal
        """
        if mode == "rgb_array":
            if self._rgb is None:
                error.Error("environment has to be reset before rendering")
            else:
                return self._rgb
        elif mode == "ansi":
            if self._env._current_game is None:
                error.Error("environment has to be reset before rendering")
            else:
                ascii_np_array = self._env._current_game._board.board
                ansi_string = "\n".join(
                    [
                        " ".join([chr(i) for i in ascii_np_array[j]])
                        for j in range(ascii_np_array.shape[0])
                    ]
                )
                return ansi_string
        elif mode == "human":
            if self._viewer is None:
                self._viewer = init_viewer(self._env_name, self._render_animation_delay)
                self._viewer.display(self._env)
            else:
                self._viewer.display(self._env)
        else:
            super(GridworldEnv, self).render(mode=mode)  # just raise an exception


class GridworldsActionSpace(gym.Space):

    def __init__(self, env):
        action_spec = env.action_spec()
        assert action_spec.name == "discrete"
        assert action_spec.dtype == "int32"
        assert len(action_spec.shape) == 1 and action_spec.shape[0] == 1
        self.min_action = int(action_spec.minimum)
        self.max_action = int(action_spec.maximum)
        self.n = (self.max_action - self.min_action) + 1
        super(GridworldsActionSpace, self).__init__(
            shape=action_spec.shape, dtype=action_spec.dtype
        )

    def sample(self):
        return random.randint(self.min_action, self.max_action)

    def contains(self, x):
        """
        Return True is x is a valid action. Note, that this does not use the
        pycolab validate function, because that expects a numpy array and not
        an individual action.
        """
        return self.min_action <= x <= self.max_action


class GridworldsObservationSpace(gym.Space):

    def __init__(self, env, use_transitions, flatten_observations):
        self.observation_spec_dict = env.observation_spec()
        self.use_transitions = use_transitions
        self.flatten_observations = flatten_observations

        if flatten_observations:
            if self.use_transitions:
                shape = (2, np.prod(self.observation_spec_dict["board"].shape))
            else:
                shape = (np.prod(self.observation_spec_dict["board"].shape), )
        else:
            if self.use_transitions:
                shape = (2, *self.observation_spec_dict["board"].shape)
            else:
                shape = (1, *self.observation_spec_dict["board"].shape)

        dtype = self.observation_spec_dict["board"].dtype
        super(GridworldsObservationSpace, self).__init__(shape=shape, dtype=dtype)

    def sample(self):
        """
        Use pycolab to generate an example observation. Note that this is not a
        random sample, but might return the same observation for every call.
        """
        if self.use_transitions:
            raise NotImplementedError(
                "Sampling from transition-based envs not yet supported."
            )
        observation = {}
        for key, spec in self.observation_spec_dict.items():  # TODO: this loop has no purpose, since only "board" key is used at the end
            if spec == {}:
                observation[key] = {}
            else:
                if isinstance(spec, dict):
                    observation[key] = {}
                    for subkey, subkey_spec in spec.items():
                        if subkey_spec == {}:
                            observation[key][subkey] = {}
                        else:
                            observation[key][subkey] = subkey_spec.generate_value()
                else:
                    observation[key] = spec.generate_value()
        result = observation["board"][np.newaxis, :]    # TODO: add object coordinates and agent perspectives?
        if self.flatten_observations:
            result = result.flatten()
        return result

    def contains(self, x):
        if "board" in self.observation_spec_dict.keys():
            try:
                self.observation_spec_dict["board"].validate(x[0, ...])
                if self.use_transitions:
                    self.observation_spec_dict["board"].validate(x[1, ...])
                return True
            except ValueError:
                return False
        else:
            return False


def init_viewer(env_name, pause):
    (color_bg, color_fg) = get_color_map(env_name)
    av = AgentViewer(pause, color_bg=color_bg, color_fg=color_fg)
    return av


def get_color_map(env_name):
    module_prefix = "ai_safety_gridworlds.environments."
    env_module_name = module_prefix + env_name
    env_module = importlib.import_module(env_module_name)
    color_bg = env_module.GAME_BG_COLOURS
    color_fg = env_module.GAME_FG_COLOURS
    return (color_bg, color_fg)
