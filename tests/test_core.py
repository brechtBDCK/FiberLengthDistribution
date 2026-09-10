import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from fiber_analysis.measure import Fiber, _orientation
from fiber_analysis.pipeline import _gray
from fiber_analysis.report import write_result


class CoreTests(unittest.TestCase):
    def test_orientation_zero_is_horizontal(self):
        self.assertAlmostEqual(_orientation([(5, 1), (5, 9)])[0], 0)
        self.assertAlmostEqual(_orientation([(1, 5), (9, 5)])[0], 90)

    def test_one_percent_crop(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.png"
            Image.new("L", (1000, 500)).save(path)
            image, scale, origin = _gray(path, None, .01)
            self.assertEqual(image.shape, (490, 980))
            self.assertEqual(scale, 1)
            self.assertEqual(origin, (5, 10))

    def test_graphs_include_each_accepted_fiber(self):
        fiber = Fiber("fiber_00001", 42, 40, None, 0, 0, 0, 1.05, 2, 3, [5, 5], [[5, 1], [5, 9]], .9, False)
        with tempfile.TemporaryDirectory() as folder:
            write_result(Path(folder), "image", "method", [fiber], {}, False, "um")
            output = Path(folder) / "image" / "method"
            self.assertTrue((output / "length_distribution.png").is_file())
            self.assertTrue((output / "orientation_distribution.png").is_file())


if __name__ == "__main__":
    unittest.main()
