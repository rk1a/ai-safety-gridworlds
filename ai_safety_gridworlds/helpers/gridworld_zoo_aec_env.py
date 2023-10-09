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
from ai_safety_gridworlds.environments.shared import safety_game_ma
from ai_safety_gridworlds.environments.shared import safety_game_moma
from ai_safety_gridworlds.helpers import factory
from ai_safety_gridworlds.helpers.agent_viewer import AgentViewer

#from safe_grid_gym_orig.envs.common.interface import (
#    INFO_HIDDEN_REWARD,
#    INFO_OBSERVED_REWARD,
#    INFO_DISCOUNT,
#)
#INFO_HIDDEN_REWARD = "hidden_reward"
INFO_OBSERVED_REWARD = "observed_reward"
INFO_DISCOUNT = "discount"


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
                        - 'friend_foe'
                        - 'island_navigation'
                        - 'island_navigation_ex'
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

    def __init__(self, env_name, use_transitions=False, render_animation_delay=0.1, flatten_observations=False, *args, **kwargs):

        self._env_name = env_name
        self._render_animation_delay = render_animation_delay
        self._viewer = None
        self._env = factory.get_environment_obj(env_name, *args, **kwargs)
        self._rgb = None
        self._use_transitions = use_transitions
        self._flatten_observations = flatten_observations
        self._last_board = None

        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
            agents = safety_game_ma.get_players(self._env.environment_data)
            # num_agents = len(agents)
            self.possible_agents = [f"agent_{agent.character}" for agent in agents]
            self.agent_name_mapping = dict(
                zip(self.possible_agents, [agent.character for agent in agents])
            )
        else:
            num_agents = 1
            self.possible_agents = [f"agent_{r}" for r in range(1, num_agents + 1)]
            self.agent_name_mapping = dict(
                zip(self.possible_agents, [str(r) for r in range(1, num_agents + 1)])
            )

        self._last_hidden_reward = { agent: 0 for agent in self.possible_agents }

        self._next_agent = None

        self.rewards = { agent: None for agent in self.possible_agents }
        self.infos: { agent: None for agent in self.possible_agents }

        if gym_v26:
            self.terminations = { agent: False for agent in self.possible_agents }
            self.truncations = { agent: False for agent in self.possible_agents }
        else:
            self.dones = { agent: False for agent in self.possible_agents }

        self.action_spaces = {
            agent: GridworldsActionSpace(self._env) for agent in self.possible_agents
        }  
        self.observation_spaces = {
            agent: GridworldsObservationSpace(self._env, use_transitions, flatten_observations) for agent in self.possible_agents
        }

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def action_space(self, agent):
        return self.action_spaces[agent]


    def agent_iter(self):     # TODO
        return ZooAECAgentIter(self)

    def _set_next_agent(self, agent):
        self._next_agent = agent

    class ZooAECAgentIter:

        def __init__(self, env):
            self.env = env

        def __iter__(self):
            self.agent_index = 0
            return self

        def __next__(self):
            if self.agent_index < self.max:
                agent = self.env.possible_agents[self.agent_index]
                self.env._set_next_agent(agent)
                self.agent_index += 1
                return agent
            else:
                self.env._set_next_agent(None)
                raise StopIteration


    def last(self):     # TODO
        return self.last_step_result

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
            timestep = self._env.step({ self._next_agent: action }, *args, **kwargs)      # CHANGED: added *args, **kwargs 
        else:
            timestep = self._env.step(action, *args, **kwargs)
            
        obs = timestep.observation
        self._rgb = obs["RGB"]

        reward = 0.0 if timestep.reward is None else timestep.reward
        done = timestep.step_type.last()

        cumulative_hidden_reward = self._env._get_hidden_reward(default_reward=None)
        if cumulative_hidden_reward is not None:
            if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
                hidden_reward = cumulative_hidden_reward - self._last_hidden_reward[self._next_agent]
                self._last_hidden_reward[self._next_agent] = cumulative_hidden_reward
            else:
                hidden_reward = cumulative_hidden_reward - self._last_hidden_reward
                self._last_hidden_reward[self.possible_agents[0]] = cumulative_hidden_reward
        else:
            hidden_reward = None

        info = {
            INFO_HIDDEN_REWARD: hidden_reward,
            INFO_OBSERVED_REWARD: reward,
            INFO_DISCOUNT: timestep.discount
        }

        for k, v in obs.items():
            if k not in ("board", "RGB"):
                info[k] = v

        board = copy.deepcopy(obs["board"])

        if self._use_transitions:
            state = np.stack([self._last_board, board], axis=0)
            self._last_board = board
        else:
            state = board[np.newaxis, :]

        if self._flatten_observations:
            state = state.flatten()


        if isinstance(self._env, safety_game_moma.SafetyEnvironmentMoMa):
            agent_name = self._next_agent
        else:
            agent_name = self.possible_agents[0]

        self.rewards[agent_name] = reward
        self.infos[agent_name] = info


        if gym_v26:
            # https://gymnasium.farama.org/content/migration-guide/
            # For users wishing to update, in most cases, replacing done with terminated and truncated=False in step() should address most issues. 
            # TODO: However, environments that have reasons for episode truncation rather than termination should read through the associated PR https://github.com/openai/gym/pull/2752
            terminated = done
            truncated = False    # TODO      
            self.last_step_result = (state, reward, terminated, truncated, info)            
            self.terminations[agent_name] = terminated
            self.truncations[agent_name] = truncated
        else:
            self.last_step_result = (state, reward, done, info)
            self.dones[agent_name] = done

        # return self.last_step

    def reset(self, *args, **kwargs):                     # CHANGED: added *args, **kwargs
        timestep = self._env.reset(*args, **kwargs)       # CHANGED: added *args, **kwargs      

        self._rgb = timestep.observation["RGB"]
        if self._viewer is not None:
            self._viewer.reset_time()

        board = copy.deepcopy(timestep.observation["board"])

        if self._use_transitions:
            state = np.stack([np.zeros_like(board), board], axis=0)
            self._last_board = board
        else:
            state = board[np.newaxis, :]

        if self._flatten_observations:
            state = state.flatten()

        # TODO: reward, info 
        reward = None
        info = None

        self.rewards[agent_name] = reward
        self.infos[agent_name] = info

        if gym_v26:
            self.last_step_result = (state, reward, False, False, info)              
            self.terminations[agent_name] = False
            self.truncations[agent_name] = False      
        else:
            self.last_step_result = (state, reward, False, info)   
            self.dones[agent_name] = False     

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
        assert action_spec.name == "discrete"
        assert action_spec.dtype == "int32"
        assert len(action_spec.shape) == 1 and action_spec.shape[0] == 1
        self.min_action = action_spec.minimum
        self.max_action = action_spec.maximum
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
        for key, spec in self.observation_spec_dict.items():
            if spec == {}:
                observation[key] = {}
            else:
                observation[key] = spec.generate_value()
        result = observation["board"][np.newaxis, :]
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
