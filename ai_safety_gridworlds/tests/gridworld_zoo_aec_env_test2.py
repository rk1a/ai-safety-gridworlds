import os
import sys
import pytest
import numpy as np
import numpy.testing as npt

from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo.test import api_test
from pettingzoo.test.seed_test import seed_test
# from pettingzoo.utils import parallel_to_aec

from ai_safety_gridworlds.helpers import factory
from ai_safety_gridworlds.demonstrations import demonstrations
from ai_safety_gridworlds.environments.shared.safety_game import Actions
from ai_safety_gridworlds.helpers.gridworld_zoo_aec_env import GridworldZooAecEnv


@pytest.fixture
def demos():
    _demos = {}
    for env_name in factory._environment_classes.keys():
        try:
            demos = demonstrations.get_demonstrations(env_name)
        except ValueError:
            # no demos available
            demos = []
        _demos[env_name] = demos

    # add demo that fails, to test hidden reward
    _demos["absent_supervisor"].append(   # TODO: Functionality to provide ready env for Zoo wrapper constructor
        demonstrations.Demonstration(0, [Actions.DOWN] * 3, 47, 17, True)
    )
    return _demos


def test_gridworlds_api_sequential(demos):
    for env_name in demos.keys():
        # seed = int(time.time()) & 0xFFFFFFFF
        # np.random.seed(seed)
        # print(seed)
        env_params = {
            "env_name": env_name,
            "ascii_observation_format": False,    # Zoo API tester cannot handle ascii observations
            "scalarise": True,   # Zoo API tester cannot handle multi-objective rewards
            # "seed": seed,    # TODO
        }
        sequential_env = GridworldZooAecEnv(**env_params)
        # TODO: Nathan was able to get the sequential-turn env to work, using this conversion, but not the parallel env. why??
        # sequential_env = parallel_to_aec(parallel_env)
        api_test(sequential_env, num_cycles=10, verbose_progress=True)


def test_gridworlds_seed(demos):
    for env_name in demos.keys():
        env_params = {
            "env_name": env_name,
            "ascii_observation_format": False,    # Zoo API tester cannot handle ascii observations
            "scalarise": True,   # Zoo API tester cannot handle multi-objective rewards
            "override_infos": True  # Zoo seed_test is unable to compare infos unless they have simple structure.
        }
        sequential_env = lambda: GridworldZooAecEnv(**env_params)  # seed test requires lambda
        try:
            seed_test(sequential_env, num_cycles=10)
        except TypeError:
            # for some reason the test env in Git does not recognise the num_cycles neither as named or positional argument
            seed_test(sequential_env)


def test_gridworlds_step_result(demos):
    for env_name in demos.keys():
        env_params = {
            "env_name": env_name,
            "num_iters": 2,            
            "scalarise": True,   # Zoo API tester cannot handle multi-objective rewards
        }
        env = GridworldZooAecEnv(**env_params)
        num_agents = len(env.possible_agents)
        assert num_agents, f"expected 1 agent, got: {num_agents}"
        env.reset()

        agent = env.agent_selection
        action = env.action_space(agent).sample()

        env.step(action)
        # NB! env.last() provides observation from NEXT agent in case of multi-agent environment
        observation, reward, terminated, truncated, info = env.last()    # TODO: multi-agent iteration
        done = terminated or truncated

        assert isinstance(
            observation, np.ndarray
        ), "observation of agent is not an array"
        # assert isinstance(reward, np.float64), "reward of agent is not a float64"


def test_gridworlds_done_step(demos):
    for env_name in demos.keys():
        env_params = {
            "env_name": env_name,   
            "num_iters": 2,                 
            "scalarise": True,   # Zoo API tester cannot handle multi-objective rewards
        }
        env = GridworldZooAecEnv(**env_params)
        assert len(env.possible_agents) == 1
        env.reset()

        for _ in range(env_params["num_iters"]):
            agent = env.agent_selection
            action = env.action_space(agent).sample()
            env.step(action)
            # env.last() provides observation from NEXT agent in case of multi-agent environment
            terminated = env.terminations[agent]
            truncated = env.truncations[agent]
            done = terminated or truncated

        assert done
        with pytest.raises(ValueError):
            action = env.action_space(agent).sample()
            env.step(action)


def test_gridworlds_agents(demos):
    for env_name in demos.keys():
        env_params = {
            "env_name": env_name,  
            "amount_agents": 1,
            "scalarise": True,   # Zoo API tester cannot handle multi-objective rewards
        }
        env = GridworldZooAecEnv(**env_params)

        assert len(env.possible_agents) == env_params["amount_agents"]
        assert isinstance(env.possible_agents, list)
        assert isinstance(env.unwrapped.agent_name_mapping, dict)
        assert all(
            agent_name in env.unwrapped.agent_name_mapping
            for agent_name in env.possible_agents
        )


def test_gridworlds_action_spaces(demos):
    for env_name in demos.keys():
        env_params = {
            "env_name": env_name,           
            "scalarise": True,   # Zoo API tester cannot handle multi-objective rewards
        }
        env = GridworldZooAecEnv(**env_params)

        for agent in env.possible_agents:
            assert isinstance(env.action_space(agent), MultiDiscrete)
            # assert env.action_space(agent).n == 5   # includes no-op


if __name__ == "__main__" and os.name == 'nt':  # detect debugging
    pytest.main([__file__])  # run tests only in this file
    # test_gridworlds_api_sequential()
