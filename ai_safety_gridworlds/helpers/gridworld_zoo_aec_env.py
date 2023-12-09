# Copyright 2022-2023 Roland Pihlakas. https://github.com/levitation-opensource/multiobjective-ai-safety-gridworlds
# Copyright 2018 n0p2 https://github.com/n0p2/gym_ai_safety_gridworlds
"""
The GridworldZooAecEnv implements the Zoo AEC interface for the ai_safety_gridworlds.

GridworldZooAecEnv is derived from an GridworldGymEnv implementation by n0p2.
The original repo can be found at https://github.com/n0p2/gym_ai_safety_gridworlds
"""

import importlib
import random

import pettingzoo
from pettingzoo import AECEnv

from typing import Dict, List, Optional, NamedTuple, Tuple

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


class GridworldZooAecEnv(AECEnv):
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
                 
                 agents_stepping_order=None,  # stepping order is specified as a list of map characters

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
        self._last_observed_agent_board = {}
        self._last_observation = None
        self._last_observation_coordinates = None
        self._last_observation_layers_cube = None
        self._last_agent_observations_after_some_agents_step = None
        self._last_agent_observation_coordinates = None
        self._last_agent_observations_layers_cubes = None

        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
            self._env.set_observable_attribute_categories(observable_attribute_categories, observable_attribute_value_mapping)
            self._env.reset() # apply _observable_attribute_categories

            agents = safety_game_ma.get_players(self._env.environment_data)
            # num_agents = len(agents)
            if agents_stepping_order is None:
              self.possible_agents = [f"agent_{agent.character}" for agent in agents]  # TODO: make it readonly
              self.agent_name_mapping = dict(
                  zip(self.possible_agents, [agent.character for agent in agents])
              )
            else:
              self.possible_agents = [f"agent_{character}" for character in agents_stepping_order]  # TODO: make it readonly
              self.agent_name_mapping = dict(
                  zip(self.possible_agents, list(agents_stepping_order))
              )
        else:
            #if len(observable_attribute_categories) > 0:
            #    raise ValueError("observable_attribute_categories")
            num_agents = 1
            self.possible_agents = [f"agent_{r}" for r in range(0, num_agents)]  # TODO: make it readonly
            self.agent_name_mapping = dict(   # TODO: read agent char from environment
                zip(self.possible_agents, [str(r) for r in range(0, num_agents)])
            )

        self._last_hidden_reward = { agent: 0 for agent in self.possible_agents }

        self._next_agent = self.possible_agents[0]
        self._next_agent_index = 0
        self._all_agents_done = False

        state = None    # TODO?
        reward = None
        info = None

        self.rewards = { agent: reward for agent in self.possible_agents }  # TODO: make it readonly for callers
        self.infos = { agent: info for agent in self.possible_agents }  # TODO: make it readonly for callers

        if gym_v26:
            self._given_agents_last_step_result = { agent: (state, reward, False, False, info) for agent in self.possible_agents }
            self.terminations = { agent: False for agent in self.possible_agents }  # TODO: make it readonly for callers
            self.truncations = { agent: False for agent in self.possible_agents }  # TODO: make it readonly for callers
        else:
            self._given_agents_last_step_result = { agent: (state, reward, False, info) for agent in self.possible_agents }
            self.dones = { agent: False for agent in self.possible_agents }  # TODO: make it readonly for callers

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
        if gym_v26:
            return [agent for agent in self.possible_agents if not self.terminations[agent] and not self.truncations[agent]]
        else:
            return [agent for agent in self.possible_agents if not self._dones[agent]]

    @property
    def num_agents(self):
        if gym_v26:
            return sum(1 for agent in self.possible_agents if not self.terminations[agent] and not self.truncations[agent])
        else:
            return sum(1 for agent in self.possible_agents if not self._dones[agent])

    @property
    def max_num_agents(self):
        return len(self.possible_agents)

    @property
    def agent_selection(self):
        return self._next_agent

    @property
    def state(self):
        return self._state

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]


    def agent_iter(self, max_iter = 2**63):     
        """ This iter returns current agent and keeps count of total steps. This iter DOES NOT change active agent while iterating. Active agent is changed by the .step() function of the AEC env.
        See https://pettingzoo.farama.org/content/basic_usage/#interacting-with-environments  
        """
        return GridworldZooAecEnv.ZooAECAgentIter(self, max_iter)

    #def _set_next_agent(self, agent):
    #    self._next_agent = self.agent_name_mapping[agent] if agent is not None else None

    def _move_to_next_agent(self):  # https://pettingzoo.farama.org/content/basic_usage/#interacting-with-environments      

        continue_search_for_non_done_agent = True
        search_loops_count = 0

        while continue_search_for_non_done_agent:

            self._next_agent_index = (self._next_agent_index + 1) % len(self.possible_agents) # loop over agents repeatedly     # https://pettingzoo.farama.org/content/basic_usage/#interacting-with-environments  
            agent = self.possible_agents[self._next_agent_index]                

            if gym_v26:
                done = self.terminations[agent] or self.truncations[agent]
            else:
                done = self.dones[agent]

            continue_search_for_non_done_agent = done

            search_loops_count += 1
            if continue_search_for_non_done_agent and search_loops_count == len(self.possible_agents):   # all agents are done     # https://pettingzoo.farama.org/content/basic_usage/#interacting-with-environments  
                self._next_agent_index = -1
                self._next_agent = None
                self._all_agents_done = True
                return

        #/ while search_for_non_done_agent:

        self._next_agent = agent

    #/ def _move_to_next_agent(self):  


    class ZooAECAgentIter:

        def __init__(self, env, max_iter):
            self.env = env
            self.num_iterations = 0
            self.max_iter = max_iter

        def __iter__(self):
            return self

        def __next__(self):
            if self.num_iterations < self.max_iter and not self.env._all_agents_done:
                self.num_iterations += 1
                return self.env._next_agent
            else: 
                raise StopIteration

    # TODO: function to observe with info
    def observe(self, agent, transition_from_agents_last_step_result=False):

        # get board observation from latest step, regardless whether the latest step was made by current agent or some other. If agent perspectives are available, we get current agent's perspective computed after that latest step made by any agent.
        if hasattr(self._env, "agent_perspectives"): # are agent perspectives enabled and available?
            board = self._last_agent_observations_after_some_agents_step[agent].board
        else:
            board = copy.deepcopy(self._last_observation["board"])

        if self._use_transitions:
            if not transition_from_agents_last_step_result:
                # transition from previous observation by current agent
                last_agent_board = self._last_observed_agent_board.get(agent)
                if last_agent_board is None:
                    last_agent_board = np.zeros_like(board)
                state = np.stack([last_agent_board, board], axis=0) # TODO
                self._last_observed_agent_board[agent] = board
            else:
                # transition from last step result by current agent
                last_agent_board = self._given_agents_last_step_result[agent][0][-1] # -1 takes last element from agent state. State may contain two elements in case self._use_transitions == True
                state = np.stack([last_agent_board, board], axis=0)
                self._last_observed_agent_board[agent] = board
        else:
            state = board[np.newaxis, :]

        if self._flatten_observations:
            state = state.flatten()

        return state


    def observe_info(self, agent):

        # get board observation from latest step, regardless whether the latest step was made by current agent or some other. If agent perspectives are available, _compute_info will use current agent's perspective computed after that latest step made by any agent.
        obs = self._last_observation
        info = self._compute_info(obs, agent)
        return info


    # def observe_from_location(self, agent_name, agent_coordinates, agents_observation_directions: Dict):
    def observe_infos_from_location(self, agents_coordinates: Dict, agents_observation_directions: Dict = None):
        """This method is read-only (does not change the actual state of the environment nor the actual state of agents).
        Each given agent observes the environment as well as itself as if it was in the given location."""
        
        #timestep = self._env.observe_from_location(agents_coordinates, agents_observation_directions)       # CHANGED: added *args, **kwargs      

        #board = copy.deepcopy(timestep.observation["board"])   # TODO: option to return observation as character array

        #obs = timestep.observation
        obs = self._last_observation
        # board = obs["board"]
        agents_coordinates2 = { self.agent_name_mapping[agent_name]: coordinate for agent_name, coordinate in agents_coordinates.items() }
        agents_observation_directions2 = { self.agent_name_mapping[agent_name]: direction for agent_name, direction in agents_observation_directions.items() }

        self._process_observation(obs, agents_coordinates2, agents_observation_directions2)

        infos = {}
        for agent in agents_coordinates.keys():
            info = self._compute_info(obs, agent)
            infos[agent] = info

        ## TODO: apply transitions to agent observations as well
        #if self._use_transitions:
        #    state = np.stack([np.zeros_like(board), board], axis=0)
        #    self._last_board = board
        #else:
        #    state = board[np.newaxis, :]

        ## TODO: apply flatten to agent observations as well
        #if self._flatten_observations:
        #    state = state.flatten()

        # self._state = state

        #return (state, info)
        return infos


    def last_for_agent(self, agent = None, observe = True):    
        """Returns observation, cumulative reward, terminated, truncated, info for the specified agent.
        
        If observe flag is True then CURRENT board state is observed. If observe flag is False then the observation that was made after given agent's latest move is returned.
        """
        if agent is None:
            agent = self._next_agent

        if observe:
            state = self.observe(agent)
        else:
            state = self._given_agents_last_step_result[agent][0] # take state part from _given_agents_last_step_result tuple
            if self._flatten_observations:
                state = state.flatten()

        # TODO: update coordinates and agent perspectives info if observe == True

        if gym_v26:
            (_, reward, terminated, truncated, info) = self._given_agents_last_step_result[agent]
            return (state, reward, terminated, truncated, info)
        else:
            (_, reward, done, info) = self._given_agents_last_step_result[agent]
            return (state, reward, done, info)

    def last(self, observe = True):
        """Returns observation, cumulative reward, terminated, truncated, info for the current agent (specified by self.agent_selection).
        
        If observe flag is True then current board state is observed. If observe flag is False then the observation that was made after current agent's latest move is returned."""
        result = self.last_for_agent(self._next_agent, observe)

        if gym_v26:
            (state, reward, terminated, truncated, info) = result
            if observe == False:      # thats how Zoo API test requires it to be
                state = None
            return (state, reward, terminated, truncated, info)
        else:
            (state, reward, done, info) = result
            if observe == False:      # thats how Zoo API test requires it to be
                state = None
            return (state, reward, done, info)


    def _process_observation(self, obs, observe_from_agent_coordinates = None, observe_from_agent_directions = None):
        """Computes observation perspectives."""

        self._last_observation = obs
        self._rgb = obs["RGB"]

        if self._object_coordinates_in_observation and hasattr(self._env, "calculate_observation_coordinates"):   # original Gridworlds environments do not support this method currently   # TODO
            self._last_observation_coordinates = self._env.calculate_observation_coordinates(obs, occlusion_in_layers=self._occlusion_in_layers, ascii=self._ascii_observation_format, agent_coordinates_override=observe_from_agent_coordinates)

        if observe_from_agent_coordinates is None and observe_from_agent_directions is None:
            if self._layers_order_in_cube is not None and hasattr(self._env, "calculate_observation_layers_cube"):
                self._last_observation_layers_cube = self._env.calculate_observation_layers_cube(obs, occlusion_in_layers=self._occlusion_in_layers, layers_order=self._layers_order_in_cube)

        if hasattr(self._env, "agent_perspectives"): 
            # TODO: for step() method, calculate observations and coordinates only for current agent 

            agent_observations = self._env.agent_perspectives_with_layers(obs, include_layers=not self._occlusion_in_layers, ascii=self._ascii_observation_format, observe_from_agent_coordinates=observe_from_agent_coordinates, observe_from_agent_directions=observe_from_agent_directions)
            self._last_agent_observations_after_some_agents_step = { agent_name: agent_observations[agent_chr] for agent_name, agent_chr in self.agent_name_mapping.items() }

            if self._object_coordinates_in_observation:
                agent_observations_coordinates = self._env.calculate_agents_observation_coordinates(obs, agent_observations, occlusion_in_layers=self._occlusion_in_layers, ascii=self._ascii_observation_format, observe_from_agent_coordinates=observe_from_agent_coordinates, observe_from_agent_directions=observe_from_agent_directions)
                self._last_agent_observations_coordinates = { agent_name: agent_observations_coordinates[agent_chr] for agent_name, agent_chr in self.agent_name_mapping.items() }

            if self._layers_order_in_cube_per_agent is not None:
                self._last_agent_observations_layers_cubes = { agent_name: self._env.calculate_observation_layers_cube(agent_observations[agent_chr], occlusion_in_layers=self._occlusion_in_layers, layers_order=self._layers_order_in_cube_per_agent.get(agent_name, [])) for agent_name, agent_chr in self.agent_name_mapping.items() }

    #/ def _process_observation(self, obs):


    def _compute_info(self, obs, agent_name):
        """Stores the observation into info fields."""
      
        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
            observation_direction = obs.get(INFO_OBSERVATION_DIRECTION, {}).get(self.agent_name_mapping[agent_name])
            action_direction = obs.get(INFO_ACTION_DIRECTION, {}).get(self.agent_name_mapping[agent_name])
        else:
            observation_direction = obs.get(INFO_OBSERVATION_DIRECTION)
            action_direction = obs.get(INFO_ACTION_DIRECTION)

        info = {
            INFO_OBSERVATION_DIRECTION: observation_direction,
            INFO_ACTION_DIRECTION: action_direction,
        }

        if self._object_coordinates_in_observation and hasattr(self._env, "calculate_observation_coordinates"):
            info[INFO_OBSERVATION_COORDINATES] = self._last_observation_coordinates

        if self._layers_in_observation and "layers" in obs: # only multi-objective or multi-agent environments have layers in observation available
            info[INFO_OBSERVATION_LAYERS_DICT] = obs["layers"]

        if self._layers_order_in_cube is not None and hasattr(self._env, "calculate_observation_layers_cube"):
            info[INFO_OBSERVATION_LAYERS_CUBE] = self._last_observation_layers_cube

        if hasattr(self._env, "agent_perspectives"):
            info[INFO_AGENT_OBSERVATIONS] = self._last_agent_observations_after_some_agents_step[agent_name].board

            if self._layers_in_observation:
                info[INFO_AGENT_OBSERVATION_LAYERS_DICT] = self._last_agent_observations_after_some_agents_step[agent_name].layers

            if self._object_coordinates_in_observation:
                info[INFO_AGENT_OBSERVATION_COORDINATES] = self._last_agent_observations_coordinates[agent_name]

            if self._layers_order_in_cube_per_agent is not None:
                info[INFO_AGENT_OBSERVATION_LAYERS_CUBE] = self._last_agent_observations_layers_cubes[agent_name]


        for k, v in obs.items():
            if k not in ("board", "RGB", "layers"):
                info[k] = v


        return info

    #/ def _compute_info(self, obs, agent_name):


    def step(self, action, *args, **kwargs):                    # CHANGED: added *args, **kwargs 
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
            timestep = self._env.step({ self.agent_name_mapping[self._next_agent]: action }, *args, **kwargs)      # CHANGED: added *args, **kwargs 
        else:
            timestep = self._env.step(action, *args, **kwargs)
            
        obs = timestep.observation
        self._process_observation(obs)
        info = self._compute_info(obs, self._next_agent)
        self.infos[self._next_agent] = info


        rewards = { agent_name: 0.0 if timestep.reward is None else timestep.reward[agent_chr] 
                    for agent_name, agent_chr in self.agent_name_mapping.items() }

        cumulative_hidden_reward = self._env._get_hidden_reward(default_reward=None)
        if cumulative_hidden_reward is not None:
            if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
                hidden_reward = cumulative_hidden_reward[self.agent_name_mapping[self._next_agent]] - self._last_hidden_reward[self._next_agent]
                self._last_hidden_reward[self._next_agent] = cumulative_hidden_reward[self.agent_name_mapping[self._next_agent]]
            else:
                hidden_reward = cumulative_hidden_reward - self._last_hidden_reward[self.possible_agents[0]]
                self._last_hidden_reward[self.possible_agents[0]] = cumulative_hidden_reward
        else:
            hidden_reward = None

        info.update({
            INFO_HIDDEN_REWARD: hidden_reward,
            INFO_OBSERVED_REWARD: rewards[self._next_agent],
            INFO_DISCOUNT: timestep.discount,            
        })

        # self.rewards[self._next_agent] = reward
        self.rewards.update(rewards)


        board = copy.deepcopy(obs["board"])   # TODO: option to return observation as character array

        if self._use_transitions:
            state = np.stack([self._last_board, board], axis=0)
            self._last_board = board
        else:
            state = board[np.newaxis, :]

        #if self._flatten_observations:   # flatten only when returning state, not yet here
        #    state = state.flatten()


        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
            # done = { self._next_agent: timestep.step_type[self.agent_name_mapping[self._next_agent]].last() }
            done = timestep.step_type[self.agent_name_mapping[self._next_agent]].last()
        else:
            # done = { agent: timestep.step_type.last() for agent in self.agent_name_mapping.values() }
            done = timestep.step_type.last()


        if gym_v26:
            # https://gymnasium.farama.org/content/migration-guide/
            # For users wishing to update, in most cases, replacing done with terminated and truncated=False in step() should address most issues. 
            # TODO: However, environments that have reasons for episode truncation rather than termination should read through the associated PR https://github.com/openai/gym/pull/2752
            terminated = done
            truncated = False    # TODO      
            self._given_agents_last_step_result[self._next_agent] = (state, rewards[self._next_agent], terminated, truncated, info)            
            self.terminations[self._next_agent] = terminated
            self.truncations[self._next_agent] = truncated
        else:
            self._given_agents_last_step_result[self._next_agent] = (state, rewards[self._next_agent], done, info)
            self.dones[self._next_agent] = done

        self._move_to_next_agent()    # https://pettingzoo.farama.org/content/basic_usage/#interacting-with-environments  

        # return self.last_step

    def reset(self, *args, **kwargs):                     # CHANGED: added *args, **kwargs
        timestep = self._env.reset(*args, **kwargs)       # CHANGED: added *args, **kwargs      

        if self._viewer is not None:
            self._viewer.reset_time()

        self._last_observed_agent_board = {}

        board = copy.deepcopy(timestep.observation["board"])   # TODO: option to return observation as character array

        obs = timestep.observation
        self._process_observation(obs)
        self.infos = { agent: self._compute_info(obs, agent) for agent in self.possible_agents }


        if self._use_transitions:
            state = np.stack([np.zeros_like(board), board], axis=0)
            self._last_board = board
        else:
            state = board[np.newaxis, :]

        #if self._flatten_observations:   # flatten only when returning state, not yet here
        #    state = state.flatten()
                    
        self._next_agent = self.possible_agents[0]
        self._next_agent_index = 0
        self._all_agents_done = False

        reward = 0.0    # Zoo api_test requires reward to be 0 upon reset()
        self.rewards = { agent: reward for agent in self.possible_agents }


        if gym_v26:
            self._given_agents_last_step_result = { agent: (state, reward, False, False, self.infos[agent]) for agent in self.possible_agents }
            self.terminations = { agent: False for agent in self.possible_agents }
            self.truncations = { agent: False for agent in self.possible_agents }  
        else:
            self._given_agents_last_step_result = { agent: (state, reward, False, self.infos[agent]) for agent in self.possible_agents }
            self.dones = { agent: False for agent in self.possible_agents }

        # return state

    def get_reward_unit_space(self):                    # ADDED
        return self._env.get_reward_unit_space()

    def get_trial_no(self):                             # ADDED
        return self._env.get_trial_no()

    def get_episode_no(self):                           # ADDED
        return self._env.get_episode_no()

    # gym does not support additional arguments to .step() method so we need to use a separate method. See also https://github.com/openai/gym/issues/2399
    def set_current_q_value_per_action(self, q_value_per_action_per_agent: dict):                           # ADDED
        q_value_per_action = next(iter(q_value_per_action_per_agent.items()))[1]
        return self._env.set_current_q_value_per_action(q_value_per_action)

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

        if isinstance(env, safety_game_moma.SafetyEnvironmentMoMa):
            assert action_spec[0].name == "discrete"
            assert action_spec[0].dtype == "int32"
            assert action_spec[1].name == "continuous"
            assert action_spec[1].dtype == "float32"
            # self.min_action = action_spec[0].minimum.astype(int)
            # self.max_action = action_spec[0].maximum.astype(int)
            self.min_action = action_spec[0].minimum.astype(int)[0]   # spec for step modality
            self.max_action = action_spec[0].maximum.astype(int)[0]   # spec for step modality
            # TODO: multimodal action spec
            action_spec = action_spec[0]
            shape = (1,)
        else:
            assert action_spec.name == "discrete"
            assert action_spec.dtype == "int32"
            assert len(action_spec.shape) == 1 and action_spec.shape[0] == 1
            self.min_action = int(action_spec.minimum)
            self.max_action = int(action_spec.maximum)
            shape = action_spec.shape

        self.n = (self.max_action - self.min_action) + 1
        super(GridworldsActionSpace, self).__init__(
            shape=shape, dtype=action_spec.dtype
        )

    def sample(self):
        if self.shape[0] == 1:
            return random.randint(self.min_action, self.max_action)
        else:
            return np.random.randint(self.min_action, self.max_action)

    def contains(self, x):
        """
        Return True is x is a valid action. Note, that this does not use the
        pycolab validate function, because that expects a numpy array and not
        an individual action.
        """
        if self.shape[0] == 1:
            return self.min_action <= x <= self.max_action
        else:
            return all(self.min_action <= x <= self.max_action)


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
