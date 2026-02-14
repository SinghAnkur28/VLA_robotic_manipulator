"""
TRAINING SCRIPT: Vision-Language-Action Model (FIXED)
Trains the VLA model on collected demonstration data.

Author: Assignment Submission
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from tqdm import tqdm
import json
from datetime import datetime

# Fixed imports - use proper module names
from model_architecture import LightweightVLA
from dataset_loader import create_dataloader


class VLATrainer:
    """
    Trainer class for the VLA model.
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        learning_rate=1e-4,
        weight_decay=1e-5,
        device='cuda',
        log_dir='logs',
        checkpoint_dir='checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 100,  # 100 epochs
            eta_min=1e-6
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Logging
        self.writer = SummaryWriter(log_dir)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['images'].to(self.device)
            actions = batch['actions'].to(self.device)
            language_list = batch['language']
            
            # Forward pass
            predicted_actions = self.model(images, language_list)
            
            # Compute loss
            loss = self.criterion(predicted_actions, actions)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Logging
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # TensorBoard logging
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/learning_rate',
                                     self.scheduler.get_last_lr()[0],
                                     self.global_step)
        
        avg_loss = total_loss / num_batches
        self.history['train_loss'].append(avg_loss)
        self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
        
        return avg_loss
    
    def validate(self, epoch):
        """Validate the model"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['images'].to(self.device)
                actions = batch['actions'].to(self.device)
                language_list = batch['language']
                
                # Forward pass
                predicted_actions = self.model(images, language_list)
                
                # Compute loss
                loss = self.criterion(predicted_actions, actions)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.history['val_loss'].append(avg_loss)
        
        # TensorBoard logging
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss={self.best_val_loss:.4f}")
    
    def train(self, num_epochs, save_freq=10):
        """Main training loop"""
        print("=" * 80)
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Training batches per epoch: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Validation batches per epoch: {len(self.val_loader)}")
        print("=" * 80)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch) if self.val_loader else None
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"  Val Loss:   {val_loss:.4f}")
            
            # Save checkpoint
            is_best = False
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
            
            if epoch % save_freq == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
        
        # Save final history
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train VLA model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to HDF5 dataset')
    parser.add_argument('--sequence_length', type=int, default=10,
                        help='Sequence length')
    
    # Model arguments
    parser.add_argument('--feature_dim', type=int, default=512,
                        help='Feature dimension')
    parser.add_argument('--num_decoder_layers', type=int, default=4,
                        help='Number of decoder layers')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader = create_dataloader(
        hdf5_path=args.data_path,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_workers=args.num_workers,
        shuffle=True
    )
    
    # For validation, we could split the data, but for simplicity using same data
    # In production, should have separate val set
    val_loader = None
    
    # Create model
    print("\nCreating model...")
    model = LightweightVLA(
        action_dim=7,
        feature_dim=args.feature_dim,
        chunk_size=args.sequence_length,
        num_decoder_layers=args.num_decoder_layers
    )
    
    num_params = model.get_num_parameters()
    print(f"Model parameters: {num_params:,} (~{num_params * 4 / 1e6:.1f} MB)")
    
    # Create trainer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    checkpoint_dir = os.path.join(args.checkpoint_dir, timestamp)
    
    trainer = VLATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir
    )
    
    # Train
    trainer.train(num_epochs=args.num_epochs, save_freq=args.save_freq)


if __name__ == "__main__":
    main()
