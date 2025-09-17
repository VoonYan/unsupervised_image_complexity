# test_import.py
"""
Test if all imports work correctly
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing CLIC module imports...")

try:
    from clic import builder, loader, CAL

    print("✓ Successfully imported CLIC modules")

    # Test creating model
    import torchvision.models as models

    base_encoder = models.resnet50
    model = builder.CLIC(base_encoder)
    print("✓ Successfully created CLIC model")

    # Test dataset
    import tempfile
    import numpy as np
    from PIL import Image

    # Create temporary directory with dummy images
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy images
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(os.path.join(tmpdir, f'test_{i}.jpg'))

        # Test dataset loading
        dataset = loader.CLICDataset(tmpdir)
        print(f"✓ Successfully created dataset with {len(dataset)} images")

    print("\n✅ All tests passed! You can now run training.")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\nPlease make sure the clic/ directory exists with all required files.")