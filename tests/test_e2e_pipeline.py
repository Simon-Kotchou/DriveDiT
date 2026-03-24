"""
End-to-end pipeline verification test.
Tests the complete flow: data loading -> model -> training -> inference.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


def test_configuration():
    """Test configuration system."""
    print("\n=== Testing Configuration System ===")
    from config.config import (
        DriveDiTConfig, get_minimal_config, ComponentType
    )

    config = get_minimal_config()
    print(f"  Model dim: {config.model_dim}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Image size: {config.image_size}")
    print(f"  Enabled components: {[c.value for c in config.enabled_components]}")

    # Test component management
    config.enable_component(ComponentType.MEMORY)
    assert config.is_component_enabled(ComponentType.MEMORY)
    print("  ✓ Configuration system works")
    return config


def test_data_loading(config):
    """Test data loading with synthetic data."""
    print("\n=== Testing Data Loading ===")
    from data.enfusion_loader import EnfusionDataset, EnfusionDatasetConfig

    # Create synthetic test data directory (Enfusion session structure)
    test_dir = "/tmp/drivedit_test_data"
    session_dir = os.path.join(test_dir, "session_0001")
    frames_dir = os.path.join(session_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Create a minimal CSV file (telemetry.csv)
    csv_path = os.path.join(session_dir, "telemetry.csv")
    with open(csv_path, 'w') as f:
        f.write("frame_id,timestamp,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,speed,steering,throttle,brake,gear,rpm\n")
        for i in range(20):
            f.write(f"{i},{i*0.2:.3f},{i*0.5},{0},{0},{0},{i*0.01},{0},{10+i*0.1},{0.1},{0.5},{0},{3},{2000}\n")

    # Create dummy frame images
    import numpy as np
    for i in range(20):
        img = np.random.randint(0, 255, (config.image_size, config.image_size, 3), dtype=np.uint8)
        img_path = os.path.join(frames_dir, f"frame_{i:06d}.png")
        try:
            import cv2
            cv2.imwrite(img_path, img)
        except ImportError:
            # Fallback: create empty file
            with open(img_path, 'wb') as f:
                f.write(b'\x00' * 1000)

    # Test dataset creation with correct API
    try:
        dataset_config = EnfusionDatasetConfig(
            sequence_length=4,
            image_size=(config.image_size, config.image_size),
            load_depth=False,
            load_scene=False
        )
        dataset = EnfusionDataset(
            data_root=test_dir,
            config=dataset_config,
            split="train"
        )
        print(f"  Dataset created with {len(dataset)} samples")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  Sample keys: {list(sample.keys())}")
            if 'frames' in sample:
                print(f"  Frames shape: {sample['frames'].shape}")
        print("  ✓ Data loading works")
        return True
    except Exception as e:
        print(f"  ⚠ Data loading issue: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation(config):
    """Test model instantiation."""
    print("\n=== Testing Model Creation ===")

    from models.world_model import WorldModel

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")

    model = WorldModel(config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print("  ✓ Model creation works")
    return model, device


def test_forward_pass(model, config, device):
    """Test model forward pass."""
    print("\n=== Testing Forward Pass ===")

    B = 2  # batch size
    T = config.sequence_length
    C = config.in_channels
    H = W = config.image_size  # image_size is an int

    # Create dummy inputs
    frames = torch.randn(B, T, C, H, W, device=device)
    controls = torch.randn(B, config.control_input_dim, device=device)

    print(f"  Input frames shape: {frames.shape}")
    print(f"  Control shape: {controls.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(
                frames=frames,
                controls=controls,
                mode='train'
            )

            if isinstance(outputs, dict):
                print(f"  Output keys: {list(outputs.keys())}")
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: {v.shape}")
            else:
                print(f"  Output shape: {outputs.shape}")

            print("  ✓ Forward pass works")
            return True
        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_training_step(model, config, device):
    """Test training step."""
    print("\n=== Testing Training Step ===")

    from training.unified_trainer import UnifiedTrainer
    from training.losses import UnifiedLoss

    # Create trainer
    trainer = UnifiedTrainer(model, config)
    loss_fn = UnifiedLoss(config)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Create dummy batch
    B = 2
    T = config.sequence_length
    C = config.in_channels
    H = W = config.image_size

    batch = {
        'frames': torch.randn(B, T, C, H, W, device=device),
        'controls': torch.randn(B, config.control_input_dim, device=device),
    }

    print(f"  Batch frames shape: {batch['frames'].shape}")

    # Training step
    model.train()
    try:
        losses = trainer.train_step(batch, optimizer)
        print(f"  Loss values: {losses}")
        print("  ✓ Training step works")
        return True
    except Exception as e:
        print(f"  ✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference(model, config, device):
    """Test inference/rollout components."""
    print("\n=== Testing Inference ===")

    from inference.rollout import MemoryBank, InferenceConfig

    # Test InferenceConfig
    inf_config = InferenceConfig(
        max_sequence_length=16,
        context_window=4,
        chunk_size=4
    )
    print(f"  InferenceConfig created: chunk_size={inf_config.chunk_size}")

    # Test MemoryBank
    d_model = config.model_dim
    memory_bank = MemoryBank(d_model=d_model, max_spatial_memory=64)
    print(f"  MemoryBank created: d_model={d_model}")

    # Test model inference mode
    B = 1
    T = 4
    C = config.in_channels
    H = W = config.image_size

    context_frames = torch.randn(B, T, C, H, W, device=device)
    controls = torch.randn(B, config.control_input_dim, device=device)

    print(f"  Context frames shape: {context_frames.shape}")

    try:
        model.eval()
        with torch.no_grad():
            # Test inference forward pass
            outputs = model(
                frames=context_frames,
                controls=controls,
                mode='inference'
            )
            if isinstance(outputs, dict):
                print(f"  Inference output keys: {list(outputs.keys())}")
            print("  ✓ Inference works")
        return True
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vae():
    """Test VAE encoding/decoding."""
    print("\n=== Testing VAE ===")

    try:
        from models.vae3d import VAE3D, VAE3DConfig

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create VAE with minimal config
        vae_config = VAE3DConfig(
            in_channels=3,
            latent_dim=8,
            hidden_dims=[32, 64, 128],
            use_attention=False,
            use_perceptual_loss=False
        )
        vae = VAE3D(config=vae_config).to(device)

        # Test forward pass - note VAE expects [B, C, T, H, W] not [B, T, C, H, W]
        B, C, T, H, W = 1, 3, 4, 64, 64
        x = torch.randn(B, C, T, H, W, device=device)

        with torch.no_grad():
            # Full forward pass returns dict with outputs and losses
            outputs = vae(x)
            print(f"  Output keys: {list(outputs.keys())}")
            if 'recon' in outputs:
                print(f"  Recon shape: {outputs['recon'].shape}")
            if 'z' in outputs:
                print(f"  Latent shape: {outputs['z'].shape}")

        print("  ✓ VAE works")
        return True
    except Exception as e:
        print(f"  ⚠ VAE test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all E2E tests."""
    print("=" * 60)
    print("DriveDiT End-to-End Pipeline Verification")
    print("=" * 60)

    results = {}

    # Test configuration
    try:
        config = test_configuration()
        results['configuration'] = True
    except Exception as e:
        print(f"  ✗ Configuration failed: {e}")
        results['configuration'] = False
        return results

    # Test data loading
    results['data_loading'] = test_data_loading(config)

    # Test VAE
    results['vae'] = test_vae()

    # Test model creation
    try:
        model, device = test_model_creation(config)
        results['model_creation'] = True
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        results['model_creation'] = False
        return results

    # Test forward pass
    results['forward_pass'] = test_forward_pass(model, config, device)

    # Test training step
    results['training_step'] = test_training_step(model, config, device)

    # Test inference
    results['inference'] = test_inference(model, config, device)

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 All tests passed! Pipeline is functional.")
    else:
        print("⚠ Some tests failed. See details above.")

    return results


if __name__ == "__main__":
    run_all_tests()
