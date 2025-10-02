# run_complete_pipeline.py
"""
Run the complete CLIC pipeline from start to finish
"""

import subprocess
import sys
import os
import time


def run_step(step_name, command):
    """Run a pipeline step"""
    print("\n" + "=" * 60)
    print(f"STEP: {step_name}")
    print("=" * 60)

    start = time.time()
    result = subprocess.run(command, shell=True)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"✓ {step_name} completed in {elapsed:.2f} seconds")
    else:
        print(f"✗ {step_name} failed")
        return False

    return True


def main():
    print("\n" + "=" * 70)
    print("CLIC: Complete Pipeline Execution")
    print("=" * 70)

    steps = [
        ("Data Preparation", "python quick_setup.py"),
        ("Unsupervised Pre-training", "python train_cpu.py"),
        ("Fine-tuning on IC9600", "python fine_tuning.py"),
        ("Model Evaluation", "python evaluate_model.py"),
        ("Feature Visualization", "python visualize_features.py")
    ]

    for step_name, command in steps:
        if not run_step(step_name, command):
            print(f"\nPipeline stopped at: {step_name}")
            break

    print("\n" + "=" * 70)
    print("Pipeline Complete! Check the following outputs:")
    print("=" * 70)
    print("1. Model checkpoints: ./checkpoints/")
    print("2. Complexity ranking: ./complexity_ranking.png")
    print("3. t-SNE visualization: ./tsne_visualization.png")
    print("4. Attention maps: ./attention_maps.png")
    print("=" * 70)


if __name__ == "__main__":
    main()