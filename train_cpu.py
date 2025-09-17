# train_cpu.py
"""
Simplified CPU training script for quick testing
"""

import torch
import torchvision.models as models
from clic.builder import CLIC
from clic.loader import CLICDataset, TwoCropsTransform
import os

# Parameters optimized for CPU
BATCH_SIZE = 4  # Small batch size for CPU
QUEUE_SIZE = 1024  # Smaller queue for CPU
EPOCHS = 5  # Fewer epochs for testing
WORKERS = 0  # No multiprocessing for simplicity


def train_cpu():
    print("Starting CPU training with optimized parameters...")

    # Device
    device = torch.device('cpu')

    # Model
    print("Creating model...")
    base_encoder = models.resnet50
    model = CLIC(
        base_encoder,
        dim=128,
        K=QUEUE_SIZE,
        m=0.999,
        T=0.07,
        device=device
    )

    # Dataset
    print("Loading dataset...")
    dataset = CLICDataset(
        root_dir='./data/clic_dataset/images',
        transform=TwoCropsTransform()
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        drop_last=True
    )

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)

    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Queue size: {QUEUE_SIZE}")
    print(f"Starting training for {EPOCHS} epochs...")
    print("-" * 60)

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        batch_count = 0

        for batch_idx, (images, _) in enumerate(dataloader):
            # Get two augmented views
            im_q, im_k = images[0], images[1]

            # Forward pass
            output, target = model(im_q=im_q, im_k=im_k)

            # Compute loss
            loss = criterion(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss
            epoch_loss += loss.item()
            batch_count += 1

            # Print progress
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch + 1}/{EPOCHS}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}')

        # Epoch summary
        avg_loss = epoch_loss / batch_count
        print(f'Epoch [{epoch + 1}/{EPOCHS}] Complete - Average Loss: {avg_loss:.4f}')
        print("-" * 60)

        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            os.makedirs('./checkpoints', exist_ok=True)
            checkpoint_path = f'./checkpoints/cpu_checkpoint_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')

    print("\nTraining completed!")


if __name__ == "__main__":
    train_cpu()