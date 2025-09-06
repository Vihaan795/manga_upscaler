#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DAT architecture variants (DAT, DAT-2, DAT-S, DAT-light).
Adapted from https://github.com/lovindata/pillow-dat/
"""

from .DAT import DAT


def create_dat_base(**kwargs):
    """Create DAT Base model configuration"""
    defaults = {
        'img_size': 64,
        'in_chans': 3,
        'embed_dim': 180,
        'split_size': [8, 32],
        'depth': [6, 6, 6, 6, 6, 6],
        'num_heads': [6, 6, 6, 6, 6, 6],
        'expansion_factor': 4.0,
        'qkv_bias': True,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'use_chk': False,
        'upscale': 2,
        'img_range': 1.0,
        'resi_connection': '1conv',
        'upsampler': 'pixelshuffle',
    }
    defaults.update(kwargs)
    return DAT(**defaults)


def create_dat_2(**kwargs):
    """Create DAT-2 model configuration"""
    defaults = {
        'img_size': 64,
        'in_chans': 3,
        'embed_dim': 180,
        'split_size': [8, 32],
        'depth': [6, 6, 6, 6, 6, 6],
        'num_heads': [6, 6, 6, 6, 6, 6],
        'expansion_factor': 2.0,  # Reduced from 4.0
        'qkv_bias': True,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'use_chk': False,
        'upscale': 2,
        'img_range': 1.0,
        'resi_connection': '1conv',
        'upsampler': 'pixelshuffle',
    }
    defaults.update(kwargs)
    return DAT(**defaults)


def create_dat_s(**kwargs):
    """Create DAT-S model configuration"""
    defaults = {
        'img_size': 64,
        'in_chans': 3,
        'embed_dim': 180,
        'split_size': [8, 16],  # Smaller split size
        'depth': [6, 6, 6, 6, 6, 6],
        'num_heads': [6, 6, 6, 6, 6, 6],
        'expansion_factor': 2.0,  # Reduced from 4.0
        'qkv_bias': True,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'use_chk': False,
        'upscale': 2,
        'img_range': 1.0,
        'resi_connection': '1conv',
        'upsampler': 'pixelshuffle',
    }
    defaults.update(kwargs)
    return DAT(**defaults)


def create_dat_light(**kwargs):
    """Create DAT-light model configuration"""
    defaults = {
        'img_size': 64,
        'in_chans': 3,
        'embed_dim': 60,  # Much smaller
        'split_size': [8, 32],
        'depth': [18],  # Single deeper layer
        'num_heads': [6],
        'expansion_factor': 2.0,
        'qkv_bias': True,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'use_chk': False,
        'upscale': 2,
        'img_range': 1.0,
        'resi_connection': '3conv',  # Different connection type
        'upsampler': 'pixelshuffledirect',  # Direct upsampling
    }
    defaults.update(kwargs)
    return DAT(**defaults)


# Model configuration mapping
DAT_CONFIGS = {
    'dat': create_dat_base,
    'dat_base': create_dat_base,
    'dat-2': create_dat_2,
    'dat_2': create_dat_2,
    'dat2': create_dat_2,
    'dat-s': create_dat_s,
    'dat_s': create_dat_s,
    'dats': create_dat_s,
    'dat-light': create_dat_light,
    'dat_light': create_dat_light,
    'datlight': create_dat_light,
}


def detect_dat_variant(state_dict):
    """
    Detect DAT variant from state dictionary keys.
    
    Args:
        state_dict (dict): Model state dictionary
        
    Returns:
        str: DAT variant name
    """
    # Check for specific DAT patterns
    has_layers = any('layers.' in key for key in state_dict.keys())
    has_blocks = any('blocks.' in key for key in state_dict.keys())
    has_attn = any('attn.' in key for key in state_dict.keys())
    has_conv_first = 'conv_first.weight' in state_dict
    has_before_rg = any('before_RG' in key for key in state_dict.keys())
    
    # Check embedding dimension to determine variant
    if has_conv_first:
        embed_dim = state_dict['conv_first.weight'].shape[0]
        
        # DAT-light has smaller embedding dimension
        if embed_dim <= 64:
            return 'dat_light'
        elif embed_dim <= 120:
            return 'dat_s'
        elif embed_dim <= 180:
            # Check for other distinguishing features
            # DAT-2 and DAT-S have expansion factor 2, DAT has 4
            # This is harder to detect from state_dict, so default to dat_2
            return 'dat_2'
        else:
            return 'dat'
    
    # Default fallback
    return 'dat'