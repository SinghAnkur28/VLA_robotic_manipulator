"""
EVALUATION SCRIPT: Test trained VLA model in simulation (FIXED)
Compatible with robosuite 1.4.0+

Author: Assignment Submission
"""

import torch
import numpy as np
import argparse
import os
from tqdm import tqdm
import json

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

from model_architecture import LightweightVLA
from dataset_loader import LanguageGuidedDataset


class VLAEvaluator:
    """
    Evaluator for VLA model in simulation.
    """
    
    def __init__(
        self,
        model,
        device='cuda',
        image_size=(224, 224)
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.image_size = image_size
        
        # Use dataset's image transform
        from torchvision import transforms
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image):
        """Preprocess raw image from simulation"""
        image_tensor = self.image_transform(image)
        return image_tensor
    
    def predict_action(self, image_sequence, language_instruction):
        """
        Predict action given image sequence and language instruction.
        
        Args:
            image_sequence: List of images
            language_instruction: String
        Returns:
            action: numpy array (action_dim,)
        """
        # Preprocess images
        image_tensors = []
        for img in image_sequence:
            img_tensor = self.preprocess_image(img)
            image_tensors.append(img_tensor)
        
        # Stack and add batch dimension
        images = torch.stack(image_tensors).unsqueeze(0)  # (1, T, C, H, W)
        images = images.to(self.device)
        
        # Predict
        with torch.no_grad():
            actions = self.model(images, [language_instruction])
        
        # Return first action (action chunking - we could use more)
        action = actions[0, 0].cpu().numpy()
        
        return action
    
    def run_episode(
        self,
        env,
        language_instruction="Pick up the cube",
        max_steps=200,
        sequence_length=10,
        render=False
    ):
        """
        Run one evaluation episode.
        
        Returns:
            success: bool
            episode_length: int
            total_reward: float
        """
        obs = env.reset()
        
        # Initialize image buffer
        image_buffer = []
        
        done = False
        t = 0
        total_reward = 0
        
        while not done and t < max_steps:
            # Get current image
            current_image = obs['agentview_image']
            image_buffer.append(current_image)
            
            # Keep only last sequence_length images
            if len(image_buffer) > sequence_length:
                image_buffer.pop(0)
            
            # Predict action when we have enough images
            if len(image_buffer) == sequence_length:
                action = self.predict_action(image_buffer, language_instruction)
            else:
                # Use zero action until buffer is full
                action = np.zeros(7)
            
            # Execute action
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if render:
                env.render()
            
            t += 1
        
        # Check success
        cube_height = obs['cube_pos'][2]
        success = cube_height > 0.08  # Lifted threshold
        
        return success, t, total_reward
    
    def evaluate(
        self,
        n_episodes=10,
        language_instruction="Pick up the cube",
        render=False
    ):
        """
        Run evaluation for multiple episodes.
        
        Returns:
            results: dict with statistics
        """
        # Setup environment - Fixed for newer robosuite
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
            has_renderer=render,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names="agentview",
            camera_heights=self.image_size[0],
            camera_widths=self.image_size[1],
            reward_shaping=True,
        )
        
        print("=" * 80)
        print(f"Running evaluation for {n_episodes} episodes...")
        print(f"Language instruction: '{language_instruction}'")
        print("=" * 80)
        
        results = {
            'successes': [],
            'episode_lengths': [],
            'rewards': []
        }
        
        for i in tqdm(range(n_episodes), desc="Evaluating"):
            success, length, reward = self.run_episode(
                env=env,
                language_instruction=language_instruction,
                render=render
            )
            
            results['successes'].append(success)
            results['episode_lengths'].append(length)
            results['rewards'].append(reward)
            
            print(f"\nEpisode {i+1}/{n_episodes}:")
            print(f"  Success: {success}")
            print(f"  Length: {length}")
            print(f"  Reward: {reward:.2f}")
        
        env.close()
        
        # Compute statistics
        success_rate = np.mean(results['successes'])
        avg_length = np.mean(results['episode_lengths'])
        avg_reward = np.mean(results['rewards'])
        
        results['summary'] = {
            'n_episodes': n_episodes,
            'success_rate': float(success_rate),
            'successful_episodes': int(np.sum(results['successes'])),
            'avg_episode_length': float(avg_length),
            'avg_reward': float(avg_reward),
            'language_instruction': language_instruction
        }
        
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"Success Rate: {success_rate*100:.1f}% ({int(np.sum(results['successes']))}/{n_episodes})")
        print(f"Average Episode Length: {avg_length:.1f}")
        print(f"Average Reward: {avg_reward:.2f}")
        print("=" * 80)
        
        return results


def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
    # Create model
    model = LightweightVLA(
        action_dim=7,
        feature_dim=512,
        chunk_size=10
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate VLA model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--n_episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--language', type=str, default="Pick up the cube",
                        help='Language instruction')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--render', action='store_true',
                        help='Render episodes')
    parser.add_argument('--output', type=str, default='eval_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Create evaluator
    evaluator = VLAEvaluator(model, device=device)
    
    # Run evaluation
    results = evaluator.evaluate(
        n_episodes=args.n_episodes,
        language_instruction=args.language,
        render=args.render
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
