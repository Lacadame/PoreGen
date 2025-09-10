#!/usr/bin/env python3
"""
Test script to demonstrate 2D and 3D permeability calculations
"""

import numpy as np
from poregen.features.permeability_from_lbm import permeability_from_lbm, pad_with_zeros

def create_test_2d_sample(size=50):
    """Create a simple 2D test sample with some random pores"""
    np.random.seed(42)
    sample = np.random.choice([0, 1], size=(size, size), p=[0.7, 0.3])
    return sample.astype(bool)

def create_test_3d_sample(size=30):
    """Create a simple 3D test sample with some random pores"""
    np.random.seed(42)
    sample = np.random.choice([0, 1], size=(size, size, size), p=[0.7, 0.3])
    return sample.astype(bool)

def test_2d_permeability():
    """Test 2D permeability calculation"""
    print("Testing 2D permeability calculation...")
    
    # Create a 2D test sample
    sample_2d = create_test_2d_sample(50)
    print(f"Created 2D sample with shape: {sample_2d.shape}")
    print(f"Porosity: {(~sample_2d).mean():.3f}")
    
    try:
        # Calculate permeability with default parameters (shorter simulation for testing)
        result = permeability_from_lbm(
            sample_2d,
            it_max=1000,  # Reduced for testing
            it_check=50,
            direction=[1, 0]  # Flow in x-direction
        )
        print(f"2D Permeability - Bulk: {result['bulk']:.6e}, Surface: {result['surface']:.6e}")
        return True
    except Exception as e:
        print(f"Error in 2D calculation: {e}")
        return False

def test_3d_permeability():
    """Test 3D permeability calculation"""
    print("\nTesting 3D permeability calculation...")
    
    # Create a 3D test sample
    sample_3d = create_test_3d_sample(30)
    print(f"Created 3D sample with shape: {sample_3d.shape}")
    print(f"Porosity: {(~sample_3d).mean():.3f}")
    
    try:
        # Calculate permeability with default parameters (shorter simulation for testing)
        result = permeability_from_lbm(
            sample_3d,
            it_max=1000,  # Reduced for testing
            it_check=50,
            direction=[1, 0, 0]  # Flow in x-direction
        )
        print(f"3D Permeability - Bulk: {result['bulk']:.6e}, Surface: {result['surface']:.6e}")
        return True
    except Exception as e:
        print(f"Error in 3D calculation: {e}")
        return False

def test_padding():
    """Test the padding function for both 2D and 3D"""
    print("\nTesting padding function...")
    
    # Test 2D padding
    sample_2d = np.ones((10, 15), dtype=bool)
    padded_2d = pad_with_zeros(sample_2d, buffer_size=5)
    expected_shape_2d = (20, 25)
    print(f"2D: Original shape {sample_2d.shape} -> Padded shape {padded_2d.shape}")
    assert padded_2d.shape == expected_shape_2d, f"Expected {expected_shape_2d}, got {padded_2d.shape}"
    
    # Test 3D padding
    sample_3d = np.ones((8, 12, 10), dtype=bool)
    padded_3d = pad_with_zeros(sample_3d, buffer_size=3)
    expected_shape_3d = (14, 18, 16)
    print(f"3D: Original shape {sample_3d.shape} -> Padded shape {padded_3d.shape}")
    assert padded_3d.shape == expected_shape_3d, f"Expected {expected_shape_3d}, got {padded_3d.shape}"
    
    print("Padding tests passed!")

if __name__ == "__main__":
    print("Testing 2D/3D permeability calculation adaptation")
    print("=" * 50)
    
    # Test padding function
    test_padding()
    
    # Test 2D permeability (comment out if lettuce not available)
    # success_2d = test_2d_permeability()
    
    # Test 3D permeability (comment out if lettuce not available)
    # success_3d = test_3d_permeability()
    
    print("\n" + "=" * 50)
    print("Test completed! Uncomment the permeability tests if you have lettuce installed.")
    # if success_2d and success_3d:
    #     print("All tests passed successfully!")
    # else:
    #     print("Some tests failed.")