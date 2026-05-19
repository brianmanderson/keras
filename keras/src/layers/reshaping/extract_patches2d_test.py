import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src import testing


class ExtractPatches2DTest(testing.TestCase):
    @parameterized.parameters(
        # (H, W, C, size, padding, strides)
        (16, 16, 3, (4, 4), "valid", None),
        (16, 16, 3, (4, 4), "valid", (2, 2)),
        (17, 19, 3, (4, 4), "same", None),
        (32, 32, 1, 8, "valid", None),
    )
    def test_layer_matches_op(self, H, W, C, size, padding, strides):
        x = np.random.RandomState(0).rand(2, H, W, C).astype("float32")
        x_t = ops.convert_to_tensor(x)
        op_out = ops.convert_to_numpy(ops.image.extract_patches(
            x_t, size=size, strides=strides, padding=padding,
        ))
        layer_out = ops.convert_to_numpy(layers.ExtractPatches2D(
            size=size, strides=strides, padding=padding,
        )(x_t))
        np.testing.assert_array_equal(op_out, layer_out)

    def test_compute_output_shape_valid(self):
        layer = layers.ExtractPatches2D(size=(4, 4), padding="valid")
        self.assertEqual(
            layer.compute_output_shape((None, 16, 16, 3)),
            (None, 4, 4, 48),
        )

    def test_compute_output_shape_same(self):
        layer = layers.ExtractPatches2D(size=(4, 4), padding="same")
        self.assertEqual(
            layer.compute_output_shape((None, 17, 19, 3)),
            (None, 5, 5, 48),
        )

    def test_get_config_roundtrip(self):
        layer = layers.ExtractPatches2D(
            size=(3, 5), strides=(2, 3), padding="same", name="my_extract",
        )
        config = layer.get_config()
        restored = layers.ExtractPatches2D.from_config(config)
        self.assertEqual(restored.size, (3, 5))
        self.assertEqual(restored.strides, (2, 3))
        self.assertEqual(restored.padding, "same")
        self.assertEqual(restored.name, "my_extract")

    def test_invalid_size(self):
        with self.assertRaisesRegex(ValueError, "length 2"):
            layers.ExtractPatches2D(size=(2, 3, 4))

    def test_invalid_padding(self):
        with self.assertRaisesRegex(ValueError, "'same' or 'valid'"):
            layers.ExtractPatches2D(size=(4, 4), padding="reflect")

    def test_symmetric_layer_pair_roundtrip(self):
        """ExtractPatches2D + ReconstructPatches2D should be identity."""
        H, W, C = 16, 16, 3
        x = np.random.RandomState(0).rand(2, H, W, C).astype("float32")
        x_t = ops.convert_to_tensor(x)
        patches = layers.ExtractPatches2D(size=(4, 4), padding="valid")(x_t)
        recon = layers.ReconstructPatches2D(
            size=(4, 4), output_size=(H, W), padding="valid",
        )(patches)
        np.testing.assert_allclose(
            ops.convert_to_numpy(recon), x, atol=1e-6,
        )
