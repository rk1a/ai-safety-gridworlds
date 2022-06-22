# Copyright 2022 Roland Pihlakas. https://github.com/levitation-opensource/multiobjective-ai-safety-gridworlds
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
"""Module containing factory class to instantiate all pycolab environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ai_safety_gridworlds.environments.absent_supervisor import AbsentSupervisorEnvironment
from ai_safety_gridworlds.environments.boat_race import BoatRaceEnvironment
from ai_safety_gridworlds.environments.boat_race_ex import BoatRaceEnvironmentEx
from ai_safety_gridworlds.environments.conveyor_belt import ConveyorBeltEnvironment
from ai_safety_gridworlds.environments.conveyor_belt_ex import ConveyorBeltEnvironmentEx
from ai_safety_gridworlds.environments.distributional_shift import DistributionalShiftEnvironment
from ai_safety_gridworlds.environments.friend_foe import FriendFoeEnvironment

from ai_safety_gridworlds.environments.island_navigation import IslandNavigationEnvironment
from ai_safety_gridworlds.environments.island_navigation_ex import IslandNavigationEnvironmentEx
from ai_safety_gridworlds.experiments import food_drink_unbounded
from ai_safety_gridworlds.experiments import food_drink_bounded
from ai_safety_gridworlds.experiments import food_drink_bounded_gold
from ai_safety_gridworlds.experiments import food_drink_bounded_death
from ai_safety_gridworlds.experiments import food_drink_bounded_death_gold
from ai_safety_gridworlds.experiments import food_drink_bounded_death_gold_silver

from ai_safety_gridworlds.environments.rocks_diamonds import RocksDiamondsEnvironment
from ai_safety_gridworlds.environments.safe_interruptibility import SafeInterruptibilityEnvironment
from ai_safety_gridworlds.environments.safe_interruptibility_ex import SafeInterruptibilityEnvironmentEx
from ai_safety_gridworlds.environments.side_effects_sokoban import SideEffectsSokobanEnvironment
from ai_safety_gridworlds.environments.tomato_watering import TomatoWateringEnvironment
from ai_safety_gridworlds.environments.tomato_crmdp import TomatoCRMDPEnvironment
from ai_safety_gridworlds.environments.whisky_gold import WhiskyOrGoldEnvironment


_environment_classes = {
    'boat_race': BoatRaceEnvironment,
    'boat_race_ex': BoatRaceEnvironmentEx,
    'conveyor_belt': ConveyorBeltEnvironment,
    'conveyor_belt_ex': ConveyorBeltEnvironmentEx,
    'distributional_shift': DistributionalShiftEnvironment,
    'friend_foe': FriendFoeEnvironment,

    'island_navigation': IslandNavigationEnvironment,
    'island_navigation_ex': IslandNavigationEnvironmentEx,
    'food_drink_unbounded': food_drink_unbounded.IslandNavigationEnvironmentExExperiment,
    'food_drink_bounded': food_drink_bounded.IslandNavigationEnvironmentExExperiment,
    'food_drink_bounded_gold': food_drink_bounded_gold.IslandNavigationEnvironmentExExperiment,
    'food_drink_bounded_death': food_drink_bounded_death.IslandNavigationEnvironmentExExperiment,
    'food_drink_bounded_death_gold': food_drink_bounded_death_gold.IslandNavigationEnvironmentExExperiment,
    'food_drink_bounded_death_gold_silver': food_drink_bounded_death_gold_silver.IslandNavigationEnvironmentExExperiment,

    'rocks_diamonds': RocksDiamondsEnvironment,
    'safe_interruptibility': SafeInterruptibilityEnvironment,
    'safe_interruptibility_ex': SafeInterruptibilityEnvironmentEx,
    'side_effects_sokoban': SideEffectsSokobanEnvironment,
    'tomato_watering': TomatoWateringEnvironment,
    'tomato_crmdp': TomatoCRMDPEnvironment,
    'absent_supervisor': AbsentSupervisorEnvironment,
    'whisky_gold': WhiskyOrGoldEnvironment,
}


def get_environment_obj(name, *args, **kwargs):
  """Instantiate a pycolab environment by name.

  Args:
    name: Name of the pycolab environment.
    *args: Arguments for the environment class constructor.
    **kwargs: Keyword arguments for the environment class constructor.

  Returns:
    A new environment class instance.
  """
  environment_class = _environment_classes.get(name.lower(), None)

  if environment_class:
    return environment_class(*args, **kwargs)
  raise NotImplementedError(
      'The requested environment is not available.')
