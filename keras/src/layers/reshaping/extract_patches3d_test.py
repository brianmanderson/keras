import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src import testing


class ExtractPatches3DTest(testing.TestCase):
    @parameterized.parameters(
        # (D, H, W, C, size, padding)
        (8, 16, 16, 2, (4, 4, 4), "valid"),
        (9, 17, 19, 2, (4, 4, 4), "same"),
        (8, 8, 8, 3, 4, "valid"),
    )
    def test_layer_matches_op(self, D, H, W, C, size, padding):
        x = np.random.RandomState(1).rand(1, D, H, W, C).astype("float32")
        x_t = ops.convert_to_tensor(x)
        op_out = ops.convert_to_numpy(ops.image.extract_patches_3d(
            x_t, size=size, padding=padding,
        ))
        layer_out = ops.convert_to_numpy(layers.ExtractPatches3D(
            size=size, padding=padding,
        )(x_t))
        np.testing.assert_array_equal(op_out, layer_out)

    def test_compute_output_shape_valid(self):
        layer = layers.ExtractPatches3D(size=(4, 4, 4), padding="valid")
        self.assertEqual(
            layer.compute_output_shape((None, 8, 16, 16, 2)),
            (None, 2, 4, 4, 128),
        )

    def test_get_config_roundtrip(self):
        layer = layers.ExtractPatches3D(
            size=(2, 4, 4), strides=(2, 2, 2), padding="valid",
        )
        config = layer.get_config()
        restored = layers.ExtractPatches3D.from_config(config)
        self.assertEqual(restored.size, (2, 4, 4))
        self.assertEqual(restored.padding, "valid")

    def test_invalid_size(self):
        with self.assertRaisesRegex(ValueError, "length 3"):
            layers.ExtractPatches3D(size=(2, 3))

    def test_symmetric_layer_pair_roundtrip(self):
        D, H, W, C = 8, 16, 16, 2
        x = np.random.RandomState(2).rand(1, D, H, W, C).astype("float32")
        x_t = ops.convert_to_tensor(x)
        patches = layers.ExtractPatches3D(size=(4, 4, 4), padding="valid")(x_t)
        recon = layers.ReconstructPatches3D(
            size=(4, 4, 4), output_size=(D, H, W), padding="valid",
        )(patches)
        np.testing.assert_allclose(
            ops.convert_to_numpy(recon), x, atol=1e-6,
        )
