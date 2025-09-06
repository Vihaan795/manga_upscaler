#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to validate safetensors support in simple_manga_upscaler.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add parent directory to sys.path to import upscale module
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from upscale import Upscale


def create_test_image(size=(32, 32)):
    """Create a simple test image for upscaling."""
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return img


def test_safetensors_loading():
    """Test that safetensors models can be loaded and used."""
    print("Testing safetensors support...")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Create test image
        test_img = create_test_image()
        test_img_path = input_dir / "test.png"
        cv2.imwrite(str(test_img_path), test_img)
        
        # Test FDAT M model
        print("  Testing FDAT M model...")
        try:
            upscaler = Upscale(
                model="4x_IllustrationJaNai_V2standard_FDAT_M_52k.safetensors",
                input=input_dir,
                output=output_dir,
                cpu=True
            )
            upscaler.run()
            
            # Check if output was created
            output_file = output_dir / "test.png"
            if output_file.exists():
                print("    ✓ FDAT M model works")
            else:
                print("    ✗ FDAT M model failed - no output created")
                return False
        except Exception as e:
            print(f"    ✗ FDAT M model failed: {e}")
            return False
        
        # Clear output for next test
        shutil.rmtree(str(output_dir))
        output_dir.mkdir()
        
        # Test FDAT XL model
        print("  Testing FDAT XL model...")
        try:
            upscaler = Upscale(
                model="4x_IllustrationJaNai_V2standard_FDAT_XL_18k.safetensors",
                input=input_dir,
                output=output_dir,
                cpu=True
            )
            upscaler.run()
            
            # Check if output was created
            output_file = output_dir / "test.png"
            if output_file.exists():
                print("    ✓ FDAT XL model works")
            else:
                print("    ✗ FDAT XL model failed - no output created")
                return False
        except Exception as e:
            print(f"    ✗ FDAT XL model failed: {e}")
            return False
            
    return True


def test_pth_compatibility():
    """Test that existing .pth models still work."""
    print("Testing .pth compatibility...")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Create test image
        test_img = create_test_image()
        test_img_path = input_dir / "test.png"
        cv2.imwrite(str(test_img_path), test_img)
        
        # Test .pth model
        print("  Testing 4x-AnimeSharp.pth...")
        try:
            upscaler = Upscale(
                model="4x-AnimeSharp.pth",
                input=input_dir,
                output=output_dir,
                cpu=True
            )
            upscaler.run()
            
            # Check if output was created
            output_file = output_dir / "test.png"
            if output_file.exists():
                print("    ✓ .pth model works")
            else:
                print("    ✗ .pth model failed - no output created")
                return False
        except Exception as e:
            print(f"    ✗ .pth model failed: {e}")
            return False
            
    return True


def test_model_interpolation():
    """Test model interpolation functionality."""
    print("Testing model interpolation...")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Create test image
        test_img = create_test_image()
        test_img_path = input_dir / "test.png"
        cv2.imwrite(str(test_img_path), test_img)
        
        # Test .pth interpolation
        print("  Testing .pth interpolation...")
        try:
            upscaler = Upscale(
                model="4x-AnimeSharp.pth@50&4x_eula_digimanga_bw_v2_nc1_307k.pth@50",
                input=input_dir,
                output=output_dir,
                cpu=True
            )
            upscaler.run()
            
            # Check if output was created
            output_file = output_dir / "test.png"
            if output_file.exists():
                print("    ✓ .pth interpolation works")
            else:
                print("    ✗ .pth interpolation failed - no output created")
                return False
        except Exception as e:
            print(f"    ✗ .pth interpolation failed: {e}")
            return False
            
    return True


def test_file_extension_detection():
    """Test that file extension detection works correctly."""
    print("Testing file extension detection...")
    
    from upscale import Upscale
    
    # Create a dummy upscaler instance
    dummy_upscaler = Upscale(
        model="dummy",
        input=Path("dummy"),
        output=Path("dummy"),
        cpu=True
    )
    
    # Test safetensors detection
    try:
        # This should use safetensors.torch.load_file
        result = dummy_upscaler._load_state_dict("models/4x_IllustrationJaNai_V2standard_FDAT_M_52k.safetensors")
        if isinstance(result, dict) and len(result) > 0:
            print("    ✓ Safetensors file extension detection works")
        else:
            print("    ✗ Safetensors file extension detection failed")
            return False
    except Exception as e:
        print(f"    ✗ Safetensors file extension detection failed: {e}")
        return False
    
    # Test .pth detection
    try:
        # This should use torch.load
        result = dummy_upscaler._load_state_dict("models/4x-AnimeSharp.pth")
        if isinstance(result, dict) and len(result) > 0:
            print("    ✓ .pth file extension detection works")
        else:
            print("    ✗ .pth file extension detection failed")
            return False
    except Exception as e:
        print(f"    ✗ .pth file extension detection failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("Running safetensors support tests...\n")
    
    # Check if we're in the right directory
    if not Path("models").exists():
        print("Error: tests must be run from the repository root directory")
        sys.exit(1)
    
    tests = [
        test_file_extension_detection,
        test_pth_compatibility,
        test_safetensors_loading,
        test_model_interpolation,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())