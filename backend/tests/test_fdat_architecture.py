#!/usr/bin/env python3
"""
Test FDAT architecture detection and loading.
This test ensures that FDAT models load correctly without warnings about unexpected keys.
"""

import unittest
import warnings
from pathlib import Path
from io import StringIO
import sys
import contextlib
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import safetensors.torch

from upscale import Upscale
from utils.architecture.FDAT import FDATNet


class TestFDATArchitecture(unittest.TestCase):
    """Test FDAT architecture detection and loading."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.models_dir = Path("models")
        self.fdat_models = [
            "4x_IllustrationJaNai_V2standard_FDAT_XL_18k.safetensors",
            "4x_IllustrationJaNai_V2standard_FDAT_M_52k.safetensors"
        ]
    
    def test_fdat_models_exist(self):
        """Test that FDAT models exist in the models directory."""
        for model_name in self.fdat_models:
            model_path = self.models_dir / model_name
            self.assertTrue(model_path.exists(), f"FDAT model {model_name} not found")
    
    def test_fdat_parameter_detection(self):
        """Test that FDAT models have correct parameter detection."""
        expected_params = {
            "4x_IllustrationJaNai_V2standard_FDAT_XL_18k.safetensors": {
                "scale": 4,
                "in_nc": 3,
                "out_nc": 3,
                "num_feat": 180,
                "num_groups": 6
            },
            "4x_IllustrationJaNai_V2standard_FDAT_M_52k.safetensors": {
                "scale": 4,
                "in_nc": 3,
                "out_nc": 3,
                "num_feat": 120,
                "num_groups": 4
            }
        }
        
        for model_name, expected in expected_params.items():
            model_path = self.models_dir / model_name
            if not model_path.exists():
                self.skipTest(f"Model {model_name} not found")
            
            with self.subTest(model=model_name):
                state_dict = safetensors.torch.load_file(model_path)
                model = FDATNet(state_dict)
                
                self.assertEqual(model.scale, expected["scale"])
                self.assertEqual(model.in_nc, expected["in_nc"])
                self.assertEqual(model.out_nc, expected["out_nc"])
                self.assertEqual(model.num_feat, expected["num_feat"])
                self.assertEqual(model.num_blocks, expected["num_groups"])
    
    def test_fdat_loading_no_warnings(self):
        """Test that FDAT models load without unexpected key warnings."""
        for model_name in self.fdat_models:
            model_path = self.models_dir / model_name
            if not model_path.exists():
                self.skipTest(f"Model {model_name} not found")
            
            with self.subTest(model=model_name):
                # Capture stdout to check for warnings
                captured_output = StringIO()
                
                with contextlib.redirect_stdout(captured_output):
                    state_dict = safetensors.torch.load_file(model_path)
                    model = FDATNet(state_dict)
                
                output = captured_output.getvalue()
                
                # Should not have unexpected key warnings
                self.assertNotIn("Warning: Unexpected keys", output, 
                                f"FDAT model {model_name} has unexpected key warnings")
                
                # Should not have missing key warnings (for properly detected models)
                self.assertNotIn("Warning: Missing keys", output,
                                f"FDAT model {model_name} has missing key warnings")
    
    def test_fdat_upscale_integration(self):
        """Test that FDAT models integrate correctly with the upscale system."""
        for model_name in self.fdat_models:
            model_path = self.models_dir / model_name
            if not model_path.exists():
                self.skipTest(f"Model {model_name} not found")
            
            with self.subTest(model=model_name):
                # Create upscaler instance
                upscaler = Upscale(
                    model=str(model_path),
                    input=Path("input"),
                    output=Path("output"),
                    cpu=True
                )
                
                # Test model loading
                upscaler.load_model(str(model_path))
                
                # Verify model properties
                self.assertEqual(upscaler.last_kind, "FDAT")
                self.assertEqual(upscaler.last_scale, 4)
                self.assertIn(upscaler.last_in_nc, [3, 4])  # Common input channels
                self.assertIn(upscaler.last_out_nc, [3, 4])  # Common output channels
                self.assertIsNotNone(upscaler.model)
    
    def test_group_structure_detection(self):
        """Test that group structure detection works correctly."""
        for model_name in self.fdat_models:
            model_path = self.models_dir / model_name
            if not model_path.exists():
                self.skipTest(f"Model {model_name} not found")
            
            with self.subTest(model=model_name):
                state_dict = safetensors.torch.load_file(model_path)
                
                # Create a temporary FDATNet to test detection
                temp_model = FDATNet.__new__(FDATNet)
                num_groups, depth_per_group = temp_model._detect_group_structure(state_dict)
                
                # Verify reasonable values
                self.assertGreater(num_groups, 0)
                self.assertGreater(depth_per_group, 0)
                
                # Check that the pattern length assumption is correct
                # FDAT uses ["spatial", "channel"] pattern (length 2)
                # So total blocks per group should be depth_per_group * 2
                
                # Count actual blocks in first group
                actual_blocks = 0
                for key in state_dict.keys():
                    if key.startswith('groups.0.blocks.'):
                        parts = key.split('.')
                        if len(parts) >= 4 and parts[3].isdigit():
                            block_idx = int(parts[3])
                            actual_blocks = max(actual_blocks, block_idx + 1)
                
                expected_total_blocks = depth_per_group * 2
                self.assertEqual(actual_blocks, expected_total_blocks,
                               f"Expected {expected_total_blocks} blocks, found {actual_blocks}")


if __name__ == "__main__":
    unittest.main()