import sys
import types
import unittest

import numpy as np


def install_dependency_stubs():
    pil_module = types.ModuleType("PIL")
    pil_image_module = types.ModuleType("PIL.Image")
    pil_module.Image = pil_image_module
    sys.modules.setdefault("PIL", pil_module)
    sys.modules.setdefault("PIL.Image", pil_image_module)

    scipy_module = types.ModuleType("scipy")
    scipy_ndimage_module = types.ModuleType("scipy.ndimage")
    scipy_stats_module = types.ModuleType("scipy.stats")
    scipy_stats_module.gaussian_kde = lambda *args, **kwargs: None
    scipy_module.ndimage = scipy_ndimage_module
    sys.modules.setdefault("scipy", scipy_module)
    sys.modules.setdefault("scipy.ndimage", scipy_ndimage_module)
    sys.modules.setdefault("scipy.stats", scipy_stats_module)

    matplotlib_module = types.ModuleType("matplotlib")
    matplotlib_pyplot_module = types.ModuleType("matplotlib.pyplot")
    matplotlib_colors_module = types.ModuleType("matplotlib.colors")

    class FakeLinearSegmentedColormap:
        @staticmethod
        def from_list(*args, **kwargs):
            return object()

    class FakeNormalize:
        def __init__(self, vmin=None, vmax=None):
            self.vmin = vmin
            self.vmax = vmax

    matplotlib_colors_module.LinearSegmentedColormap = FakeLinearSegmentedColormap
    matplotlib_colors_module.Normalize = FakeNormalize
    sys.modules.setdefault("matplotlib", matplotlib_module)
    sys.modules.setdefault("matplotlib.pyplot", matplotlib_pyplot_module)
    sys.modules.setdefault("matplotlib.colors", matplotlib_colors_module)

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, **kwargs: iterable
    sys.modules.setdefault("tqdm", tqdm_module)


install_dependency_stubs()

from data_statistics import mask_distribution as md


class SharedPlotParameterTests(unittest.TestCase):
    def test_mask_distribution_imports_normalize_for_shared_position_scale(self):
        self.assertTrue(hasattr(md, "Normalize"))

    def test_compute_shared_size_xlim_uses_global_relative_size_maximum(self):
        dataset_stats = [
            {"rel_sizes": np.array([0.10, 0.20])},
            {"rel_sizes": np.array([0.30, 0.40])},
        ]

        shared_xlim = md.compute_shared_size_xlim(dataset_stats)

        self.assertAlmostEqual(shared_xlim, 0.48)

    def test_compute_shared_position_scale_uses_global_density_maximum(self):
        density_maps = [
            np.array([[0.10, 0.20], [0.30, 0.40]]),
            np.array([[0.25, 0.50], [0.60, 0.90]]),
        ]

        levels, vmax = md.compute_shared_position_scale(density_maps, n_levels=5)

        self.assertEqual(len(levels), 5)
        self.assertAlmostEqual(levels[0], 0.0)
        self.assertAlmostEqual(levels[-1], 0.90)
        self.assertAlmostEqual(vmax, 0.90)


if __name__ == "__main__":
    unittest.main()
