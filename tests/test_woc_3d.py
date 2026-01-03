import os
import sys
import importlib.util
import numpy as np

# disable numba JIT to avoid requiring numba for test execution
os.environ['NUMBA_DISABLE_JIT'] = '1'

# Load local modules explicitly to avoid pulling the site package version
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

spec_radial = importlib.util.spec_from_file_location(
    'pywoc.radial_profile', os.path.join(base_path, 'pywoc', 'radial_profile.py')
)
radial_mod = importlib.util.module_from_spec(spec_radial)
sys.modules[spec_radial.name] = radial_mod
spec_radial.loader.exec_module(radial_mod)

spec_woc = importlib.util.spec_from_file_location(
    'pywoc.woc', os.path.join(base_path, 'pywoc', 'woc.py')
)
woc_mod = importlib.util.module_from_spec(spec_woc)
sys.modules[spec_woc.name] = woc_mod
spec_woc.loader.exec_module(woc_mod)

woc = woc_mod.woc


def generate_3d_volumes():
    """Generate simple 3D Gaussian-like volumes for testing."""
    np.random.seed(123456)
    size = 50
    vol1 = np.zeros((size, size, size))
    vol2 = np.zeros((size, size, size))

    # Create a Gaussian-like peak in the center for vol1
    center = size // 2
    for z in range(size):
        for y in range(size):
            for x in range(size):
                r = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
                vol1[z, y, x] = np.exp(-r**2 / (2 * 10**2))
                # Second volume is similar but slightly offset
                r2 = np.sqrt((x - center - 3)**2 + (y - center)**2 + (z - center)**2)
                vol2[z, y, x] = np.exp(-r2**2 / (2 * 10**2))

    return vol1, vol2


def test_woc_3d():
    """Test WOC calculation on 3D volumes."""
    vol1, vol2 = generate_3d_volumes()
    radii = [5, 10, 15]
    result = woc(vol1, vol2, radii, pixelsize=1, rbins=20)

    # Check that result is a valid number and in reasonable range
    assert isinstance(result, (float, np.floating))
    assert 0.0 <= result <= 1.0
    # Check approximate expected value (based on test run)
    assert np.isclose(result, 0.662761, rtol=1e-3)


def test_woc_2d_backward_compatibility():
    """Verify 2D functionality still works after 3D extension."""
    vol1, vol2 = generate_3d_volumes()

    # Take 2D slices
    size = vol1.shape[0]
    map1 = vol1[size//2, :, :]
    map2 = vol2[size//2, :, :]

    radii = [5, 10, 15]
    result = woc(map1, map2, radii, pixelsize=1, rbins=20)

    # Check that result is valid
    assert isinstance(result, (float, np.floating))
    assert 0.0 <= result <= 1.0
    # Check approximate expected value
    assert np.isclose(result, 0.703920, rtol=1e-3)


def test_3d_shape_validation():
    """Test that 3D volumes must have matching shapes."""
    vol1 = np.random.rand(10, 10, 10)
    vol2 = np.random.rand(10, 10, 15)  # Different shape

    radii = [2, 4, 6]
    try:
        woc(vol1, vol2, radii)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must have the same shape" in str(e)


def test_dimension_validation():
    """Test that only 2D and 3D arrays are accepted."""
    vol1 = np.random.rand(10, 10, 10, 10)  # 4D
    vol2 = np.random.rand(10, 10, 10, 10)

    radii = [2, 4, 6]
    try:
        woc(vol1, vol2, radii)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be 2D or 3D" in str(e)


if __name__ == '__main__':
    print("Running 3D WOC tests...")

    print("Test 1: 3D WOC calculation...", end=" ")
    test_woc_3d()
    print("PASSED")

    print("Test 2: 2D backward compatibility...", end=" ")
    test_woc_2d_backward_compatibility()
    print("PASSED")

    print("Test 3: 3D shape validation...", end=" ")
    test_3d_shape_validation()
    print("PASSED")

    print("Test 4: Dimension validation...", end=" ")
    test_dimension_validation()
    print("PASSED")

    print("\nAll 3D tests passed!")
