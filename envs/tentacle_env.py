# ==============================================================================
# TENTACLE ENVIRONMENT - Main Gym Environment Class
# ==============================================================================
"""
Gym-compatible environment for soft robotic tentacle control using MuJoCo.

CURRENT CONFIGURATION (spiral_5link.xml):
-----------------------------------------
- 5 segments, 5 joints (planar motion)
- Action space: Box(-1, 1, shape=(5,))
- Single axis rotation (Y-axis)

FUTURE CONFIGURATION (tentacle.xml):
------------------------------------
- 5 segments, 10 joints (3D motion)
- Action space: Box(-1, 1, shape=(10,))
- Dual axis rotation (pitch + yaw per segment)

TO SWITCH MODELS:
-----------------
1. Change 'xml_path' in configs/env.yaml to "models/tentacle.xml"
2. Change 'num_actuators' in configs/env.yaml to 10
3. Update observation dimensions accordingly

SEARCH FOR: "# TODO: TENTACLE.XML" to find all places that need changes for 3D
"""

import os
import numpy as np
import yaml
import mujoco
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple


class TentacleEnv(gym.Env):
    """
    Soft Robotic Tentacle Environment for Reinforcement Learning.
    
    This environment simulates a multi-segment soft robotic tentacle
    and provides a standard Gymnasium interface for RL training.
    
    Attributes:
        model: MuJoCo model object
        data: MuJoCo data object
        action_space: Gymnasium action space
        observation_space: Gymnasium observation space
    """
    
    # Gymnasium metadata
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 50,
    }
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the Tentacle Environment.
        
        Args:
            config_path: Path to YAML config file. If None, uses default.
            render_mode: 'human' for window, 'rgb_array' for pixel observations.
        """
        super().__init__()
        
        # ======================================================================
        # LOAD CONFIGURATION
        # ======================================================================
        self.config = self._load_config(config_path)
        self.render_mode = render_mode
        
        # ======================================================================
        # SETUP PATHS
        # ======================================================================
        # Get project root directory
        self.project_root = self._get_project_root()
        
        # Build full path to MuJoCo XML
        # CURRENT: models/spiral_5link.xml
        # TODO: TENTACLE.XML - Change to models/tentacle.xml for 3D motion
        xml_path = os.path.join(
            self.project_root, 
            self.config['model']['xml_path']
        )
        
        # ======================================================================
        # LOAD MUJOCO MODEL
        # ======================================================================
        print(f"[TentacleEnv] Loading model: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # ======================================================================
        # ENVIRONMENT PARAMETERS
        # ======================================================================
        self.frame_skip = self.config['model']['frame_skip']
        self.dt = self.model.opt.timestep * self.frame_skip
        
        # Episode settings
        self.max_steps = self.config['episode']['max_steps']
        self.current_step = 0
        
        # Target settings
        self.target_pos = np.array(self.config['target']['fixed_position'])
        self.success_threshold = self.config['target']['success_threshold']
        
        # Store previous action for smoothness reward
        self.prev_action = None
        
        # ======================================================================
        # ACTION SPACE
        # ======================================================================
        # CURRENT (spiral_5link.xml): 5 actuators
        # TODO: TENTACLE.XML - Will be 10 actuators for 3D motion
        num_actuators = self.config['action']['num_actuators']
        self.action_space = spaces.Box(
            low=self.config['action']['low'],
            high=self.config['action']['high'],
            shape=(num_actuators,),
            dtype=np.float32
        )
        print(f"[TentacleEnv] Action space: {self.action_space}")
        
        # ======================================================================
        # OBSERVATION SPACE
        # ======================================================================
        # Build observation space based on config
        obs_dim = self._calculate_observation_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        print(f"[TentacleEnv] Observation space: {self.observation_space}")
        
        # ======================================================================
        # RENDERING SETUP
        # ======================================================================
        self.renderer = None
        self.viewer = None
        if render_mode == 'human':
            self._setup_renderer()
        
        # ======================================================================
        # CACHE BODY/SITE IDs FOR FAST ACCESS
        # ======================================================================
        # Tip body ID (end effector)
        # CURRENT: link5 is the tip
        # For both models, link5 is the tip
        self.tip_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'link5'
        )
        print(f"[TentacleEnv] Tip body ID: {self.tip_body_id}")
        
        print("[TentacleEnv] Environment initialized successfully!")
    
    # ==========================================================================
    # CONFIGURATION LOADING
    # ==========================================================================
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        if config_path is None:
            # Default config path
            config_path = os.path.join(
                self._get_project_root(),
                'configs',
                'env.yaml'
            )
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _get_project_root(self) -> str:
        """Get the project root directory."""
        # This file is in envs/, so go up one level
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        return project_root
    
    def _calculate_observation_dim(self) -> int:
        """
        Calculate observation dimension based on config.
        
        CURRENT (spiral_5link.xml):
        - joint_pos: 5
        - joint_vel: 5
        - tip_pos: 3
        - target_pos: 3
        - distance: 1
        - prev_action: 5
        Total: 22
        
        TODO: TENTACLE.XML - Update for 10 joints:
        - joint_pos: 10
        - joint_vel: 10
        - tip_pos: 3
        - target_pos: 3
        - distance: 1
        - prev_action: 10
        Total: 37
        """
        obs_config = self.config['observation']
        num_joints = self.model.nq  # Number of joint positions
        num_actuators = self.config['action']['num_actuators']
        
        dim = 0
        
        if obs_config['include_joint_pos']:
            dim += num_joints
        
        if obs_config['include_joint_vel']:
            dim += self.model.nv  # Number of joint velocities
        
        if obs_config['include_tip_pos']:
            dim += 3  # x, y, z
        
        if obs_config['include_target_pos']:
            dim += 3  # x, y, z
        
        if obs_config['include_distance']:
            dim += 1  # scalar distance
        
        if obs_config['include_prev_action']:
            dim += num_actuators
        
        return dim
    
    # ==========================================================================
    # GYMNASIUM INTERFACE - CORE METHODS
    # ==========================================================================
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        # Reset step counter
        self.current_step = 0
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # ======================================================================
        # RANDOMIZE INITIAL STATE (if enabled)
        # ======================================================================
        if self.config['randomization']['randomize_init_pos']:
            init_range = self.config['randomization']['init_pos_range']
            self.data.qpos[:] += self.np_random.uniform(
                -init_range, init_range, size=self.model.nq
            )
        
        if self.config['randomization']['randomize_init_vel']:
            vel_range = self.config['randomization']['init_vel_range']
            self.data.qvel[:] = self.np_random.uniform(
                -vel_range, vel_range, size=self.model.nv
            )
        
        # ======================================================================
        # SET TARGET POSITION
        # ======================================================================
        self._set_target()
        
        # ======================================================================
        # INITIALIZE PREVIOUS ACTION
        # ======================================================================
        self.prev_action = np.zeros(self.config['action']['num_actuators'])
        
        # Forward simulation to update positions
        mujoco.mj_forward(self.model, self.data)
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Control signal for actuators
                    CURRENT: shape=(5,) for spiral_5link.xml
                    TODO: TENTACLE.XML - shape=(10,) for tentacle.xml
        
        Returns:
            observation: Current observation after action
            reward: Reward for this step
            terminated: Whether episode ended (success/failure)
            truncated: Whether episode was cut short (timeout)
            info: Additional information
        """
        # ======================================================================
        # VALIDATE AND CLIP ACTION
        # ======================================================================
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = action.astype(np.float32)
        
        # ======================================================================
        # APPLY ACTION TO ACTUATORS
        # ======================================================================
        self.data.ctrl[:] = action
        
        # ======================================================================
        # SIMULATE PHYSICS
        # ======================================================================
        # Step simulation multiple times (frame_skip)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        # Increment step counter
        self.current_step += 1
        
        # ======================================================================
        # GET OBSERVATION
        # ======================================================================
        observation = self._get_observation()
        
        # ======================================================================
        # COMPUTE REWARD
        # ======================================================================
        # FUTURE: Move this to reward.py for modularity
        # from envs.reward import compute_reward
        # reward = compute_reward(self, action)
        reward = self._compute_reward(action)
        
        # ======================================================================
        # CHECK TERMINATION CONDITIONS
        # ======================================================================
        # FUTURE: Move this to termination.py for modularity
        # from envs.termination import check_termination
        # terminated, truncated = check_termination(self)
        terminated, truncated = self._check_termination()
        
        # ======================================================================
        # GET INFO
        # ======================================================================
        info = self._get_info()
        
        # ======================================================================
        # UPDATE PREVIOUS ACTION
        # ======================================================================
        self.prev_action = action.copy()
        
        # ======================================================================
        # RENDER (if needed)
        # ======================================================================
        if self.render_mode == 'human':
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            if self.viewer is None:
                self._setup_viewer()
            self._render_human()
        elif self.render_mode == 'rgb_array':
            return self._render_rgb_array()
    
    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
    
    # ==========================================================================
    # OBSERVATION BUILDING
    # ==========================================================================
    # FUTURE: Move these methods to observation.py
    # from envs.observation import build_observation, get_tip_position, etc.
    
    def _get_observation(self) -> np.ndarray:
        """
        Build observation vector from current state.
        
        CURRENT (spiral_5link.xml):
        - Joint positions (5)
        - Joint velocities (5)
        - Tip position (3)
        - Target position (3)
        - Distance to target (1)
        - Previous action (5)
        
        TODO: TENTACLE.XML - Update dimensions for 10 joints
        
        FUTURE: Refactor to use observation.py
        >>> from envs.observation import build_observation
        >>> return build_observation(self)
        """
        obs_config = self.config['observation']
        obs_parts = []
        
        # ----------------------------------------------------------------------
        # JOINT POSITIONS
        # ----------------------------------------------------------------------
        # CURRENT: 5 values (spiral_5link.xml)
        # TODO: TENTACLE.XML - Will be 10 values
        if obs_config['include_joint_pos']:
            joint_pos = self.data.qpos.copy()
            obs_parts.append(joint_pos)
        
        # ----------------------------------------------------------------------
        # JOINT VELOCITIES
        # ----------------------------------------------------------------------
        # CURRENT: 5 values (spiral_5link.xml)
        # TODO: TENTACLE.XML - Will be 10 values
        if obs_config['include_joint_vel']:
            joint_vel = self.data.qvel.copy()
            obs_parts.append(joint_vel)
        
        # ----------------------------------------------------------------------
        # TIP (END EFFECTOR) POSITION
        # ----------------------------------------------------------------------
        if obs_config['include_tip_pos']:
            tip_pos = self._get_tip_position()
            obs_parts.append(tip_pos)
        
        # ----------------------------------------------------------------------
        # TARGET POSITION
        # ----------------------------------------------------------------------
        if obs_config['include_target_pos']:
            obs_parts.append(self.target_pos.copy())
        
        # ----------------------------------------------------------------------
        # DISTANCE TO TARGET
        # ----------------------------------------------------------------------
        if obs_config['include_distance']:
            distance = self._compute_distance_to_target()
            obs_parts.append(np.array([distance]))
        
        # ----------------------------------------------------------------------
        # PREVIOUS ACTION
        # ----------------------------------------------------------------------
        # CURRENT: 5 values (spiral_5link.xml)
        # TODO: TENTACLE.XML - Will be 10 values
        if obs_config['include_prev_action']:
            obs_parts.append(self.prev_action.copy())
        
        # Concatenate all parts
        observation = np.concatenate(obs_parts).astype(np.float32)
        
        return observation
    
    def _get_tip_position(self) -> np.ndarray:
        """
        Get the position of the tentacle tip (end effector).
        
        For spiral_5link.xml and tentacle.xml, the tip is at link5.
        
        Returns:
            tip_pos: 3D position [x, y, z]
        """
        # Get body position from xpos array
        # xpos contains world positions of all bodies
        tip_pos = self.data.xpos[self.tip_body_id].copy()
        
        # Add offset for the tip (link5 is 0.25m long)
        # The tip is at the end of link5, so we need to add the link length
        # along the body's z-axis in world frame
        link_length = 0.25  # From XML: fromto="0 0 0  0 0 0.25"
        
        # Get the rotation matrix of the tip body
        tip_rot = self.data.xmat[self.tip_body_id].reshape(3, 3)
        
        # The link extends along local z-axis, so add offset
        tip_offset = tip_rot @ np.array([0, 0, link_length])
        tip_pos = tip_pos + tip_offset
        
        return tip_pos
    
    # ==========================================================================
    # REWARD COMPUTATION
    # ==========================================================================
    # FUTURE: Move these methods to reward.py
    # from envs.reward import compute_reward
    
    def _compute_reward(self, action: np.ndarray) -> float:
        """
        Compute reward for the current step.
        
        Reward = w_dist * r_dist + w_success * r_success + w_smooth * r_smooth + w_energy * r_energy
        
        FUTURE: Refactor to use reward.py
        >>> from envs.reward import compute_reward
        >>> return compute_reward(self, action)
        """
        reward_config = self.config['reward']
        total_reward = 0.0
        
        # ----------------------------------------------------------------------
        # DISTANCE REWARD (negative distance to encourage getting closer)
        # ----------------------------------------------------------------------
        distance = self._compute_distance_to_target()
        distance_reward = -distance * reward_config['weight_distance']
        total_reward += distance_reward
        
        # ----------------------------------------------------------------------
        # SUCCESS BONUS
        # ----------------------------------------------------------------------
        if distance < self.success_threshold:
            success_bonus = reward_config['bonus_success'] * reward_config['weight_success']
            total_reward += success_bonus
        
        # ----------------------------------------------------------------------
        # SMOOTHNESS PENALTY (penalize jerky actions)
        # ----------------------------------------------------------------------
        if self.prev_action is not None:
            action_diff = np.sum(np.square(action - self.prev_action))
            smoothness_penalty = -action_diff * reward_config['weight_smoothness']
            total_reward += smoothness_penalty
        
        # ----------------------------------------------------------------------
        # ENERGY PENALTY (penalize large control signals)
        # ----------------------------------------------------------------------
        energy = np.sum(np.square(action))
        energy_penalty = -energy * reward_config['weight_energy']
        total_reward += energy_penalty
        
        return total_reward
    
    def _compute_distance_to_target(self) -> float:
        """Compute Euclidean distance from tip to target."""
        tip_pos = self._get_tip_position()
        distance = np.linalg.norm(tip_pos - self.target_pos)
        return distance
    
    # ==========================================================================
    # TERMINATION CONDITIONS
    # ==========================================================================
    # FUTURE: Move these methods to termination.py
    # from envs.termination import check_termination
    
    def _check_termination(self) -> Tuple[bool, bool]:
        """
        Check if episode should end.
        
        Returns:
            terminated: True if success or failure (episode ends naturally)
            truncated: True if timeout (episode cut short)
        
        FUTURE: Refactor to use termination.py
        >>> from envs.termination import check_termination
        >>> return check_termination(self)
        """
        terminated = False
        truncated = False
        
        # ----------------------------------------------------------------------
        # SUCCESS: Target reached
        # ----------------------------------------------------------------------
        if self.config['episode']['terminate_on_success']:
            distance = self._compute_distance_to_target()
            if distance < self.success_threshold:
                terminated = True
        
        # ----------------------------------------------------------------------
        # TRUNCATION: Timeout (max steps exceeded)
        # ----------------------------------------------------------------------
        if self.current_step >= self.max_steps:
            truncated = True
        
        # ----------------------------------------------------------------------
        # FAILURE: Instability detection (optional)
        # ----------------------------------------------------------------------
        # FUTURE: Add collision detection, self-intersection, etc.
        # if self.config['episode']['terminate_on_failure']:
        #     if self._check_failure():
        #         terminated = True
        
        return terminated, truncated
    
    # ==========================================================================
    # TARGET MANAGEMENT
    # ==========================================================================
    
    def _set_target(self):
        """Set target position based on config mode."""
        mode = self.config['target']['mode']
        
        if mode == 'fixed':
            self.target_pos = np.array(self.config['target']['fixed_position'])
        
        elif mode == 'random':
            bounds = self.config['target']['random_bounds']
            self.target_pos = np.array([
                self.np_random.uniform(bounds['x_min'], bounds['x_max']),
                self.np_random.uniform(bounds['y_min'], bounds['y_max']),
                self.np_random.uniform(bounds['z_min'], bounds['z_max']),
            ])
        
        # TODO: Add curriculum mode for progressive difficulty
        # elif mode == 'curriculum':
        #     self.target_pos = self._get_curriculum_target()
    
    # ==========================================================================
    # INFO DICTIONARY
    # ==========================================================================
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state."""
        tip_pos = self._get_tip_position()
        distance = self._compute_distance_to_target()
        
        return {
            'tip_position': tip_pos.copy(),
            'target_position': self.target_pos.copy(),
            'distance_to_target': distance,
            'is_success': distance < self.success_threshold,
            'current_step': self.current_step,
            'joint_positions': self.data.qpos.copy(),
            'joint_velocities': self.data.qvel.copy(),
        }
    
    # ==========================================================================
    # RENDERING
    # ==========================================================================
    
    def _setup_renderer(self):
        """Setup MuJoCo renderer."""
        self.renderer = mujoco.Renderer(
            self.model,
            self.config['render']['width'],
            self.config['render']['height']
        )
    
    def _setup_viewer(self):
        """Setup interactive viewer for human rendering."""
        try:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        except Exception as e:
            print(f"[TentacleEnv] Warning: Could not create viewer: {e}")
            self.viewer = None
    
    def _render_human(self):
        """Render to screen."""
        if self.viewer is not None:
            self.viewer.sync()
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render to RGB array."""
        if self.renderer is None:
            self._setup_renderer()
        
        self.renderer.update_scene(self.data)
        return self.renderer.render()
    
    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================
    # FUTURE: Move to utils.py
    
    def get_body_position(self, body_name: str) -> np.ndarray:
        """Get position of a named body."""
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        return self.data.xpos[body_id].copy()
    
    def get_joint_positions(self) -> np.ndarray:
        """Get all joint positions."""
        return self.data.qpos.copy()
    
    def get_joint_velocities(self) -> np.ndarray:
        """Get all joint velocities."""
        return self.data.qvel.copy()
    
    def set_joint_positions(self, qpos: np.ndarray):
        """Set joint positions and update simulation."""
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
    
    @property
    def num_joints(self) -> int:
        """Number of joints in the model."""
        return self.model.nq
    
    @property
    def num_actuators(self) -> int:
        """Number of actuators in the model."""
        return self.model.nu


# ==============================================================================
# QUICK TEST
# ==============================================================================
if __name__ == "__main__":
    """
    Quick test to verify environment works.
    Run: python envs/tentacle_env.py
    """
    print("=" * 60)
    print("Testing TentacleEnv")
    print("=" * 60)
    
    # Create environment
    env = TentacleEnv()
    
    print(f"\nAction space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Number of joints: {env.num_joints}")
    print(f"Number of actuators: {env.num_actuators}")
    
    # Test reset
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial distance to target: {info['distance_to_target']:.4f}")
    
    # Test step with random actions
    print("\nRunning 10 random steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.4f}, distance={info['distance_to_target']:.4f}")
        
        if terminated or truncated:
            print(f"  Episode ended! Success: {info['is_success']}")
            break
    
    env.close()
    print("\n[TentacleEnv] Test complete!")
