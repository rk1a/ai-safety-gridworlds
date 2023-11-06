# Copyright 2023 Roland Pihlakas. https://github.com/levitation-opensource/multiobjective-ai-safety-gridworlds
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
"""Pycolab rendering wrapper for enabling video recording.

This module contains wrappers that allow for simultaneous transformation of
environment observations into agent view (a numpy 2-D array) and human RGB view
(a numpy 3-D array).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from ai_safety_gridworlds.environments.shared import observation_distiller

import numpy as np

from pycolab import rendering


class ObservationToArrayWithRGBEx(observation_distiller.ObservationToArrayWithRGB):
  """Convert an `Observation` to a 2-D `board` and 3-D `RGB` numpy array.

  This class is a general utility for converting `Observation`s into 2-D
  `board` representation and 3-D `RGB` numpy arrays. They are returned as a
  dictionary containing the aforementioned keys.

  Additonally, `layers` key is returned exactly with same value as in the 
  `Observation.layers`, `ascii_codes` key, containing the `Observation.board` 
  value, and `ascii` key, containing the char array derived from 
  `Observation.board` value.
  """

  def __init__(self, value_mapping, colour_mapping):
    """Construct an `ObservationToArrayWithRGBEx`.

    Builds a callable that will take `Observation`s and emit a dictionary
    containing a 2-D and 3-D numpy array. The rows and columns of the 2-D array
    contain the values obtained after mapping the characters of the original
    `Observation` through `value_mapping`. The rows and columns of the 3-D array
    contain RGB values of the previous 2-D mapping in the [0,1] range.

    Args:
      value_mapping: a dict mapping any characters that might appear in the
          original `Observation`s to a scalar or 1-D vector value. All values
          in this dict must be the same type and dimension. Note that strings
          are considered 1-D vectors, not scalar values.
      colour_mapping: a dict mapping any characters that might appear in the
          original `Observation`s to a 3-tuple of RGB values in the range
          [0,999].

    """
    super(ObservationToArrayWithRGBEx, self).__init__(value_mapping, colour_mapping)

    self._board_to_ascii_vectorize = np.vectorize(chr)
    
    self._renderers.update({
        'layers': lambda observation: observation.layers, # "Rendering" function for the `layers` value.
        'ascii_codes': lambda observation: observation.board,   # "Rendering" function for the `ascii_ord` value.
        'ascii': lambda observation: self._board_to_ascii_vectorize(observation.board),   # Rendering function for the `ascii` value. converting ordinals to chars.
    })
