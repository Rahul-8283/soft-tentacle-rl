# ==============================================================================
# ENVIRONMENT MODULE - Soft Robotic Tentacle RL
# ==============================================================================
"""
This module provides the Gym-compatible environment for training RL agents
to control a soft robotic tentacle in MuJoCo simulation.

USAGE:
------
    from envs import TentacleEnv
    
    # Create environment
    env = TentacleEnv()
    
    # Or with custom config
    env = TentacleEnv(config_path="configs/env.yaml")
    
    # Standard Gym interface
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

CURRENT MODEL: spiral_5link.xml (5 joints, planar motion)
FUTURE MODEL:  tentacle.xml (10 joints, 3D motion)

To switch models, change 'xml_path' in configs/env.yaml
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

# Main environment class
from envs.tentacle_env import TentacleEnv

# Reward functions (to be implemented)
# FUTURE: Uncomment when reward.py is complete
# from envs.reward import (
#     compute_reward,
#     distance_reward,
#     success_reward,
#     smoothness_penalty,
#     energy_penalty,
# )

# Observation builders (to be implemented)
# FUTURE: Uncomment when observation.py is complete
# from envs.observation import (
#     build_observation,
#     get_joint_positions,
#     get_joint_velocities,
#     get_tip_position,
# )

# Termination conditions (to be implemented)
# FUTURE: Uncomment when termination.py is complete
# from envs.termination import (
#     check_termination,
#     is_success,
#     is_timeout,
#     is_failure,
# )

# Utility functions (to be implemented)
# FUTURE: Uncomment when utils.py is complete
# from envs.utils import (
#     normalize,
#     denormalize,
#     compute_distance,
#     load_config,
# )

# ==============================================================================
# GYMNASIUM REGISTRATION (Optional)
# ==============================================================================
"""
To register this environment with Gymnasium, add this to your training script:

    import gymnasium as gym
    from gymnasium.envs.registration import register
    
    register(
        id='SoftTentacle-v0',
        entry_point='envs:TentacleEnv',
        max_episode_steps=500,
    )
    
    # Then create with:
    env = gym.make('SoftTentacle-v0')
"""

# ==============================================================================
# MODULE EXPORTS
# ==============================================================================
__all__ = [
    'TentacleEnv',
    # FUTURE: Add these when files are complete
    # 'compute_reward',
    # 'build_observation', 
    # 'check_termination',
]

# Version
__version__ = '0.1.0'
