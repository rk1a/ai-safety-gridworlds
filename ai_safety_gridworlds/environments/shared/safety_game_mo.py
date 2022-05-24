# Copyright 2018 The AI Safety Gridworlds Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Helpers for creating safety environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

# Dependency imports
from ai_safety_gridworlds.environments.shared.rl import array_spec as specs
from ai_safety_gridworlds.environments.shared.rl import environment
from ai_safety_gridworlds.environments.shared.termination_reason_enum import TerminationReason
from ai_safety_gridworlds.environments.shared.safety_game import SafetyEnvironment, ACTUAL_ACTIONS, TERMINATION_REASON, EXTRA_OBSERVATIONS

import numpy as np

from pycolab import plot

import six


METRICS_DICT = 'metrics_dict'
METRICS_MATRIX = 'metrics_matrix'


class mo_reward(object):  # TODO: move to separate file

  def __init__(self, reward_dimensions_dict, immutable=True):

    self._reward_dimensions_dict = reward_dimensions_dict
    self._immutable = immutable


  def copy(self):

    dict_clone = dict(self._reward_dimensions_dict) # clone
    return mo_reward(dict_clone, immutable=False)


  def get_enabled_reward_dimension_keys(self, enabled_mo_reward_dimensions):

    if enabled_mo_reward_dimensions is None:

      return [None]

    else: # if enabled_mo_reward_dimensions is not None:

      # each reward may contain more than one enabled dimension
      keys_per_reward = [{ key for key, unit_value in reward._reward_dimensions_dict.items() if unit_value != 0 }
                                                        for reward in enabled_mo_reward_dimensions]
      # enabled_reward_dimension_keys = set.union(*keys_per_reward)  # this does not preserve the order of the keys
      enabled_reward_dimension_keys = dict.fromkeys(itertools.chain.from_iterable(keys_per_reward)).keys()  # this preserves the order of the keys
      return list(enabled_reward_dimension_keys)


  def tolist(self, enabled_mo_reward_dimensions):

    if enabled_mo_reward_dimensions is None:

      reward_values = self._reward_dimensions_dict.values()
      return sum(reward_values)

    else: # if enabled_mo_reward_dimensions is not None:

      enabled_reward_dimension_keys = self.get_enabled_reward_dimension_keys(enabled_mo_reward_dimensions)

      for key in self._reward_dimensions_dict.keys():
        if key not in enabled_reward_dimension_keys:
          raise ValueError("Reward %s is not enabled but is still included in mo_reward" % key)

      result = [0] * len(enabled_reward_dimension_keys)
      for dimension_index, enabled_reward_dimension_key in enumerate(enabled_reward_dimension_keys):
        result[dimension_index] = self._reward_dimensions_dict.get(enabled_reward_dimension_key, 0)
      return result


  def tofull(self, enabled_mo_reward_dimensions):

    if enabled_mo_reward_dimensions is None:

      reward_values = self._reward_dimensions_dict.values()
      return {None: sum(reward_values)}

    else: # if enabled_mo_reward_dimensions is not None:

      enabled_reward_dimension_keys = self.get_enabled_reward_dimension_keys(enabled_mo_reward_dimensions)

      for key in self._reward_dimensions_dict.keys():
        if key not in enabled_reward_dimension_keys:
          raise ValueError("Reward %s is not enabled but is still included in mo_reward" % key)

      result = {}
      for enabled_reward_dimension_key in enabled_reward_dimension_keys:
        result[enabled_reward_dimension_key] = self._reward_dimensions_dict.get(enabled_reward_dimension_key, 0)
      return result


  def __str__(self): # tostring

    return str(self._reward_dimensions_dict)


  def __repr__(self): # tostring

    return "<" + repr(self._reward_dimensions_dict) + ">"


  def __add__(self, other):

    result_dict = dict(self._reward_dimensions_dict)  # clone

    if np.isscalar(other):
      return mo_reward({ key: value + other for key, value in self._reward_dimensions_dict.items() }, immutable=False)

    elif isinstance(other, mo_reward):
      for other_key, other_value in other._reward_dimensions_dict.items():
        result_dict[other_key] = result_dict.get(other_key, 0) + other_value
      return mo_reward(result_dict, immutable=False)

    else:
      raise NotImplementedError("Unknown value type provided for mo_reward.__add__, expecting a scalar or mo_reward")


  def __iadd__(self, other):  # in-place add

    if self._immutable:
      return self.__add__(other)

    if np.isscalar(other):
      for key, value in self._reward_dimensions_dict.items():
        self._reward_dimensions_dict[key] = value + other

    elif isinstance(other, mo_reward):
      for other_key, other_value in other._reward_dimensions_dict.items():
        self._reward_dimensions_dict[other_key] = self._reward_dimensions_dict.get(other_key, 0) + other_value

    else:
      raise NotImplementedError("Unknown value type provided for mo_reward.__iadd__, expecting a scalar or mo_reward")

    return self


  def __radd__(self, other):  # reflected add (the order of operands is exchanged)

    return self + other


  def __sub__(self, other):

    result_dict = dict(self._reward_dimensions_dict)  # clone

    if np.isscalar(other):
      return mo_reward({ key: value + other for key, value in self._reward_dimensions_dict.items() }, immutable=False)

    elif isinstance(other, mo_reward):
      for other_key, other_value in other._reward_dimensions_dict.items():
        result_dict[other_key] = result_dict.get(other_key, 0) - other_value
      return mo_reward(result_dict, immutable=False)

    else:
      raise NotImplementedError("Unknown value type provided for mo_reward.__sub__, expecting a scalar or mo_reward")


  def __isub__(self, other):  # in-place sub

    if self._immutable:
      return self.__sub__(other)

    if np.isscalar(other):
      for key, value in self._reward_dimensions_dict.items():
        self._reward_dimensions_dict[key] = value + other

    elif isinstance(other, mo_reward):
      for other_key, other_value in other._reward_dimensions_dict.items():
        self._reward_dimensions_dict[other_key] = self._reward_dimensions_dict.get(other_key, 0) - other_value

    else:
      raise NotImplementedError("Unknown value type provided for mo_reward.__isub__, expecting a scalar or mo_reward")

    return self


  def __rsub__(self, other):  # reflected sub (the order of operands is exchanged)

    result_dict = dict(self._reward_dimensions_dict)  # clone

    if np.isscalar(other):
      return mo_reward({ key: other - value for key, value in self._reward_dimensions_dict.items() }, immutable=False)

    elif isinstance(other, mo_reward):
      for other_key, other_value in other._reward_dimensions_dict.items():
        result_dict[other_key] = other_value - result_dict.get(other_key, 0)
      return mo_reward(result_dict, immutable=False)

    else:
      raise NotImplementedError("Unknown value type provided for mo_reward.__rsub__, expecting a scalar or mo_reward")


  def __neg__(self):  # unary -

    return mo_reward({ key: -value for key, value in self._reward_dimensions_dict.items() }, immutable=False)


  def __mul__(self, other):

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for mo_reward.__mul__, expecting a scalar")

    return mo_reward({ key: value * other for key, value in self._reward_dimensions_dict.items() }, immutable=False)


  def __imul__(self, other):  #in-place mul

    if self._immutable:
      return self.__mul__(other)

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for mo_reward.__imul__, expecting a scalar")

    for key, value in self._reward_dimensions_dict.items():
      self._reward_dimensions_dict[key] = value * other

    return self


  def __rmul__(self, other):  # reflected mul (the order of operands is exchanged)

    return self * other


  def __truediv__(self, other):

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for mo_reward.__truediv__, expecting a scalar")

    return mo_reward({ key: value / other for key, value in self._reward_dimensions_dict.items() }, immutable=False)


  def __itruediv__(self, other):  #in-place div

    if self._immutable:
      return self.__truediv__(other)

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for mo_reward.__itruediv__, expecting a scalar")

    for key, value in self._reward_dimensions_dict.items():
      self._reward_dimensions_dict[key] = value / other

    return self


  def __rtruediv__(self, other):  # reflected div (the order of operands is exchanged)

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for mo_reward.__rtruediv__, expecting a scalar")

    return mo_reward({ key: other / value for key, value in self._reward_dimensions_dict.items() }, immutable=False)


class PlotMo(plot.Plot):  # TODO: move to separate file

  def add_reward(self, reward):
    """Add a value to the reward the `Engine` will return to the player(s).

    During a game iteration, any `Backdrop`, `Sprite`, or `Drape` can call this
    method to add value to the reward that the `Engine` will return to the
    player (or players) for having taken the action (or actions) supplied in the
    `actions` argument to `Engine`'s `play` method.

    This value need not be a number, but can be any kind of value appropriate to
    the game.  If there's ever any chance that more than one `Sprite`, `Drape`,
    or `Backdrop` would supply a reward during a game iteration, the value
    should probably be of a type that supports the `+=` operator in a relevant
    way, since this method uses addition to accumulate reward. (For custom
    classes, this typically means implementing the `__iadd__` method.)

    If this method is never called during a game iteration, the `Engine` will
    supply None to the player (or players) as the reward.

    Args:
      reward: reward value to accumulate into the current game iteration's
          reward for the player(s). See discussion for details.
    """
    if self._engine_directives.summed_reward is None:
      self._engine_directives.summed_reward = reward.copy()    # incoming mo_reward argument has to be treated as immutable else rewards across timesteps will be accumulated in per timestep accumulator
    else:
      self._engine_directives.summed_reward += reward


class SafetyEnvironmentMo(SafetyEnvironment):
  """Base class for multi-objective safety gridworld environments.

  Environments implementing this base class initialize the Python environment
  API v2 and serve as a layer in which we can put various modifications of
  pycolab inputs and outputs, such as *additional information* passed
  from/to the environment that does not fit in the traditional observation
  channel. It also allows for overwriting of default methods such as step() and
  reset().

  Each new environment must implement a subclass of this class, and at the very
  least call the __init__ method of this class with corresponding parameters, to
  instantiate the python environment API around the pycolab game.
  """

  def __init__(self, enabled_mo_reward_dimensions, 
               *args, 
               #game_factory,
               #game_bg_colours,
               #game_fg_colours,
               #actions=None,
               #value_mapping=None,
               environment_data={},
               #repainter=None,
               #max_iterations=100,
               **kwargs):
    """Initialize a Python v2 environment for a pycolab game factory.

    Args:
      enabled_mo_reward_dimensions: list of multi-objective rewards being used in 
        current map. Providing this list enables reducing the dimensionality of 
        the reward vector in such a way that unused reward dimensions are left 
        out. If set to None then the multi-objective rewards are disabled: the 
        rewards are then scalarised before returning to the agent.
      default_reward: defined in Pycolab interface, is currently ignored and 
        overridden to mo_reward({})
      game_factory: a function that returns a new pycolab `Engine`
        instance corresponding to the game being played.
      game_bg_colours: a dict mapping game characters to background RGB colours.
      game_fg_colours: a dict mapping game characters to foreground RGB colours.
      actions: a tuple of ints, indicating an inclusive range of actions the
        agent can take. Defaults to DEFAULT_ACTION_SET range.
      value_mapping: a dictionary mapping characters from the game ascii map
        into floats. Used to control how the agent sees the game ascii map, e.g.
        if we are not making a difference between environment background and
        walls in terms of values the agent sees for those blocks, we can map
        them to the same value. Defaults to mapping characters to their ascii
        codes.
      environment_data: dictionary of data that is passed to the pycolab
        environment implementation and is used as a shared object that allows
        each wrapper to communicate with their environment. This object can hold
        additional information about the state of the environment that can even
        persists through episodes, but some particular keys are erased at each
        new episode.
      repainter: a callable that converts `rendering.Observation`s to different
        `rendering.Observation`s, or None if no such conversion is required.
        This facility is normally used to change the characters used to depict
        certain game elements, and a `rendering.ObservationCharacterRepainter`
        object is a convenient way to accomplish this conversion. For more
        information, see pycolab's `rendering.py`.
      max_iterations: the maximum number of steps for one episode.
    """

    self.enabled_mo_reward_dimensions = enabled_mo_reward_dimensions


    if environment_data is None:
      self._environment_data = {}
    else:
      self._environment_data = environment_data

    self._environment_data[METRICS_DICT] = dict()
    self._environment_data[METRICS_MATRIX] = np.empty([0, 2], np.object)


    super(SafetyEnvironmentMo, self).__init__(*args, environment_data=self._environment_data, **kwargs)

    # parent class safety_game.SafetyEnvironment sets default_reward=0
    self._default_reward = mo_reward({})  # TODO: consider default_reward argument's value


  # adapted from SafetyEnvironment.reset() in ai_safety_gridworlds\environments\shared\safety_game.py and from Environment.reset() in ai_safety_gridworlds\environments\shared\rl\pycolab_interface.py
  def reset(self):
    """Start a new episode."""
    # Environment._compute_observation_spec() -> Environment.reset() -> Engine.its_showtime() -> Engine.play() -> Engine._update_and_render() is called straight from the constructor of Environment therefore need to overwrite _the_plot variable here. Overwriting it in SafetyEnvironmentMo.__init__ would be too late

    # start of code adapted from from Environment.reset()
    # Build a new game and retrieve its first set of state/reward/discount.
    self._current_game = self._game_factory()

    self._current_game._the_plot = PlotMo()    # ADDED: incoming mo_reward argument to add_reward() has to be treated as immutable else rewards across timesteps will be accumulated in per timestep accumulator

    self._state = environment.StepType.FIRST
    # Collect environment returns from starting the game and update state.
    observations, reward, discount = self._current_game.its_showtime()
    self._update_for_game_step(observations, reward, discount)
    timestep = environment.TimeStep(
        step_type=self._state,
        reward=None,
        discount=None,
        observation=self.last_observations)
    # end of code adapted from from Environment.reset()

    return self._process_timestep(timestep)  # adapted from SafetyEnvironment.reset()


  #def _compute_observation_spec(self):
  #  """Helper for `__init__`: compute our environment's observation spec."""
  #  # Environment._compute_observation_spec() -> Environment.reset() -> Engine.its_showtime() -> Engine.play() -> Engine._update_and_render() is called straight from the constructor of Environment therefore need to overwrite _the_plot variable here. Overwriting it in SafetyEnvironmentMo.__init__ would be too late

  #  self._current_game._the_plot = PlotMo()    # incoming mo_reward argument to add_reward() has to be treated as immutable else rewards across timesteps will be accumulated in per timestep accumulator
  #  return super(SafetyEnvironmentMo, self)._compute_observation_spec()


  # adapted from SafetyEnvironment._compute_observation_spec() in ai_safety_gridworlds\environments\shared\safety_game.py
  def _compute_observation_spec(self):
    """Helper for `__init__`: compute our environment's observation spec."""
    # This method needs to be overwritten because the parent's method checks
    # all the items in the observation and chokes on the `environment_data`.

    # Start an environment, examine the values it gives to us, and reset things
    # back to default.
    timestep = self.reset()
    observation_spec = {k: specs.ArraySpec(v.shape, v.dtype, name=k)
                        for k, v in six.iteritems(timestep.observation)
                        if k not in [EXTRA_OBSERVATIONS, METRICS_DICT, METRICS_MATRIX]}                 # CHANGE
    observation_spec[EXTRA_OBSERVATIONS] = dict()
    observation_spec[METRICS_DICT] = dict()                                                             # ADDED
    observation_spec[METRICS_MATRIX] = np.empty(timestep.observation[METRICS_MATRIX].shape, np.object)  # ADDED
    self._drop_last_episode()
    return observation_spec


  # adapted from SafetyEnvironment.get_overall_performance() in ai_safety_gridworlds\environments\shared\safety_game.py
  def get_overall_performance(self, default=None):
    """Returns the performance measure of the agent across all episodes.

    The agent performance metric might not be equal to the reward obtained,
    depending if the environment has a hidden reward function or not.

    Args:
      default: value to return if performance is not yet calculated (i.e. None).

    Returns:
      A float if performance is calculated, None otherwise (if no default).
    """
    if len(self._episodic_performances) < 1:
      return default
    # CHANGE: mo_reward is not directly convertible to float
    reward_dims = self._calculate_overall_performance().tolist(self.enabled_mo_reward_dimensions)
    return [float(x) for x in reward_dims]  # an alternative would be to compute `float(sum(reward_dims))`


  # adapted from SafetyEnvironment.get_last_performance() in ai_safety_gridworlds\environments\shared\safety_game.py
  def get_last_performance(self, default=None):
    """Returns the last measured performance of the agent.

    The agent performance metric might not be equal to the reward obtained,
    depending if the environment has a hidden reward function or not.

    This method will return the last calculated performance metric.
    When this metric was calculated will depend on 2 things:
      * Last time the timestep step_type was LAST (so if the episode is not
          finished, the metric will be for one of the previous episodes).
      * Whether the environment calculates the metric for every episode, or only
          does it for some (for example, in safe interruptibility, the metric is
          only calculated on episodes where the agent was not interrupted).

    Args:
      default: value to return if performance is not yet calculated (i.e. None).

    Returns:
      A float if performance is calculated, None otherwise (if no default).
    """
    if len(self._episodic_performances) < 1:
      return default
    # CHANGE: mo_reward is not directly convertible to float
    reward_dims = self._episodic_performances[-1].tolist(self.enabled_mo_reward_dimensions)
    return [float(x) for x in reward_dims]  # an alternative would be to compute `float(sum(reward_dims))`


  # adapted from safety_game.py SafetyEnvironment._process_timestep(self, timestep)
  def _process_timestep(self, timestep):
    """Do timestep preprocessing before sending it to the agent.

    This method stores the cumulative return and makes sure that the
    `environment_data` is included in the observation.

    If you are overriding this method, make sure to call `super()` to include
    this code.

    Args:
      timestep: instance of environment.TimeStep

    Returns:
      Preprocessed timestep.
    """

    # Reset the cumulative episode reward.
    if timestep.first():
      self._episode_return = mo_reward({})    # CHANGE: for multi-objective rewards
      self._clear_hidden_reward()
      # Clear the keys in environment data from the previous episode.
      for key in self._keys_to_clear:
        self._environment_data.pop(key, None)

    # Add the timestep reward for internal wrapper calculations.
    if timestep.reward:
      self._episode_return += timestep.reward
    extra_observations = self._get_agent_extra_observations()

    if ACTUAL_ACTIONS in self._environment_data:
      extra_observations[ACTUAL_ACTIONS] = (
          self._environment_data[ACTUAL_ACTIONS])

    if timestep.last():
      # Include the termination reason for the episode if missing.
      if TERMINATION_REASON not in self._environment_data:
        self._environment_data[TERMINATION_REASON] = TerminationReason.MAX_STEPS

      extra_observations[TERMINATION_REASON] = (
          self._environment_data[TERMINATION_REASON])

    timestep.observation[EXTRA_OBSERVATIONS] = extra_observations

    # Calculate performance metric if the episode has finished.
    if timestep.last():
      self._calculate_episode_performance(timestep)


    # ADDED
    timestep.observation[METRICS_MATRIX] = self._environment_data.get(METRICS_MATRIX, {}) 
    timestep.observation[METRICS_DICT] = self._environment_data.get(METRICS_DICT, {})    


    # ADDED: add conversion of mo_reward to a list
    if timestep.reward is not None:
      timestep = timestep._replace(reward=timestep.reward.tolist(self.enabled_mo_reward_dimensions))
    else: # NB! do not return None since safe_grid_gym would convert that to scalar 0
      timestep = timestep._replace(reward=mo_reward({}).tolist(self.enabled_mo_reward_dimensions))


    return timestep

