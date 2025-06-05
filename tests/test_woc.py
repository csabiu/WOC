import os
import sys
import importlib.util
import numpy as np
from astropy.convolution import convolve, Gaussian2DKernel

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


def generate_maps():
    """Generate data similar to nb/tutorial.py."""
    np.random.seed(123456)
    x, y = np.random.multivariate_normal((500, 500), ((8600, -10200), (4000, 6600)), 200000, check_valid='ignore').T
    a, _, _ = np.histogram2d(x, y, 100, range=((200, 800), (200, 800)))
    kernel = Gaussian2DKernel(10, mode='linear_interp')
    dm_model = convolve(a, kernel, boundary='extend', nan_treatment='interpolate', preserve_nan=False)

    x, y = np.random.multivariate_normal((450, 550), ((8600, -1200), (4000, 6600)), 200000, check_valid='ignore').T
    a, _, _ = np.histogram2d(x, y, 100, range=((200, 800), (200, 800)))
    kernel = Gaussian2DKernel(10, mode='linear_interp')
    icl_model = convolve(a, kernel, boundary='extend', nan_treatment='interpolate', preserve_nan=False)
    return dm_model, icl_model


def test_woc_value():
    dm, icl = generate_maps()
    value = woc(dm, icl, [10, 20, 30], plot=False, rbins=20)
    assert np.isclose(value, 0.4768612628717713, rtol=1e-5)
