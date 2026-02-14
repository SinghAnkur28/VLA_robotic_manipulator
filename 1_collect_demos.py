"""
DATA COLLECTION SCRIPT: Language-Guided Manipulation Dataset (FINAL FIX)
Compatible with robosuite 1.4.0+ and h5py string encoding

Author: Assignment Submission
Task: Collect 50 episodes of successful Lift demonstrations
"""

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
import numpy as np
import h5py
import os
from datetime import datetime

# Language instruction templates for the Lift task
LANGUAGE_INSTRUCTIONS = [
    "Pick up the cube",
    "Lift the red cube",
    "Grasp the cube and lift it up",
    "Move the cube upward",
    "Grab the object and raise it",
    "Pick the cube from the table",
    "Lift the object above the table",
    "Grasp and elevate the cube",
]


class ScriptedLiftPolicy:
    """
    A simple scripted policy for the Lift task.
    Uses a state machine approach to complete the task.
    """
    
    def __init__(self, env):
        self.env = env
        self.phase = "approach"
        self.target_height = 0.15  # Target lift height
        
    def reset(self):
        """Reset the policy state"""
        self.phase = "approach"
        
    def get_action(self, obs):
        """
        Returns an action based on current observation.
        Action space: [dx, dy, dz, dax, day, daz, gripper]
        """
        # Get cube position from observation
        cube_pos = obs["cube_pos"]
        eef_pos = obs["robot0_eef_pos"]
        
        # Compute relative position
        delta = cube_pos - eef_pos
        
        # Default action
        action = np.zeros(7)
        
        if self.phase == "approach":
            # Move towards cube (slightly above it)
            target = cube_pos.copy()
            target[2] += 0.05  # Hover above cube
            
            delta = target - eef_pos
            action[:3] = delta * 5.0  # Position control with gain
            action[6] = 1.0  # Open gripper
            
            # Check if close enough to cube
            if np.linalg.norm(delta) < 0.02:
                self.phase = "grasp"
                
        elif self.phase == "grasp":
            # Move down to cube level
            delta = cube_pos - eef_pos
            action[:3] = delta * 5.0
            action[6] = -1.0  # Close gripper
            
            # Wait for grasp
            if np.linalg.norm(delta) < 0.015:
                self.phase = "lift"
                
        elif self.phase == "lift":
            # Lift the cube
            target = cube_pos.copy()
            target[2] = self.target_height
            
            delta = target - eef_pos
            action[:3] = delta * 3.0
            action[6] = -1.0  # Keep gripper closed
            
            # Check if lifted
            if cube_pos[2] > 0.10:
                self.phase = "hold"
                
        elif self.phase == "hold":
            # Hold position
            action[:3] = 0.0
            action[6] = -1.0  # Keep gripper closed
            
        # Clip actions
        action = np.clip(action, -1.0, 1.0)
        return action


def collect_demos(n_episodes=50, data_dir="demo_data"):
    """
    Collect demonstration episodes using a scripted policy.
    
    Args:
        n_episodes: Number of episodes to collect
        data_dir: Directory to save the dataset
    """
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Setup environment
    try:
        controller_config = load_composite_controller_config(
            controller="OSC_POSE",
            robot="Panda"
        )
    except:
        controller_config = None
    
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=224,
        camera_widths=224,
        reward_shaping=True,
    )
    
    # Create HDF5 file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file_path = os.path.join(data_dir, f"lift_demos_{timestamp}.h5")
    data_file = h5py.File(data_file_path, "w")
    
    # Create main data group
    grp = data_file.create_group("data")
    
    # Metadata
    metadata = grp.create_group("metadata")
    metadata.attrs["env_name"] = "Lift"
    metadata.attrs["robot"] = "Panda"
    metadata.attrs["n_episodes"] = n_episodes
    metadata.attrs["timestamp"] = timestamp
    
    print(f"Collecting {n_episodes} episodes...")
    print(f"Output file: {data_file_path}")
    print("=" * 80)
    
    successful_episodes = 0
    
    for i in range(n_episodes):
        obs = env.reset()
        policy = ScriptedLiftPolicy(env)
        
        # Select random language instruction
        language_instruction = np.random.choice(LANGUAGE_INSTRUCTIONS)
        
        # Storage for episode data
        images = []
        actions = []
        rewards = []
        states = []
        
        done = False
        t = 0
        max_steps = 200
        episode_reward = 0
        
        while not done and t < max_steps:
            # Get action from policy
            action = policy.get_action(obs)
            
            # Store observation (image)
            image = obs["agentview_image"]
            images.append(image)
            
            # Store state info (for debugging, not used during inference)
            state = {
                'eef_pos': obs['robot0_eef_pos'],
                'cube_pos': obs['cube_pos']
            }
            states.append(state)
            
            # Execute action
            obs, reward, done, info = env.step(action)
            
            # Store action and reward
            actions.append(action)
            rewards.append(reward)
            episode_reward += reward
            
            t += 1
        
        # Check success
        cube_height = obs['cube_pos'][2]
        success = cube_height > 0.08  # Cube is lifted
        
        if success:
            successful_episodes += 1
            
        # Save episode data
        ep_grp = grp.create_group(f"demo_{i}")
        
        # Store arrays
        ep_grp.create_dataset("images", data=np.array(images), compression="gzip")
        ep_grp.create_dataset("actions", data=np.array(actions))
        ep_grp.create_dataset("rewards", data=np.array(rewards))
        
        # FIX: Store language instruction as bytes to avoid encoding issues
        ep_grp.attrs["language_instruction"] = language_instruction.encode('utf-8')
        ep_grp.attrs["success"] = success
        ep_grp.attrs["episode_length"] = len(actions)
        ep_grp.attrs["total_reward"] = episode_reward
        
        print(f"Episode {i+1}/{n_episodes}: "
              f"Success={success}, "
              f"Length={len(actions)}, "
              f"Reward={episode_reward:.2f}, "
              f"Instruction='{language_instruction}'")
    
    # Save final statistics
    data_file.attrs["successful_episodes"] = successful_episodes
    data_file.attrs["success_rate"] = successful_episodes / n_episodes
    
    print("=" * 80)
    print(f"Data collection complete!")
    print(f"Successful episodes: {successful_episodes}/{n_episodes} "
          f"({successful_episodes/n_episodes*100:.1f}%)")
    print(f"Dataset saved to: {data_file_path}")
    
    data_file.close()
    env.close()
    
    return data_file_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect demonstration data')
    parser.add_argument('--n_episodes', type=int, default=50,
                        help='Number of episodes to collect')
    parser.add_argument('--data_dir', type=str, default='demo_data',
                        help='Directory to save dataset')
    
    args = parser.parse_args()
    
    collect_demos(n_episodes=args.n_episodes, data_dir=args.data_dir)
