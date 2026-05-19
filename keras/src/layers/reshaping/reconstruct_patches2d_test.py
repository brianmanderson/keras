import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src import testing


def _gradient_image_2d(H, W, C=1, batch=1):
    h = np.linspace(0, 1, H)
    w = np.linspace(0, 1, W)
    c = np.linspace(0, 1, C) if C > 1 else np.array([0.5])
    Hg, Wg, Cg = np.meshgrid(h, w, c, indexing="ij")
    img = ((2 * Hg + 3 * Wg + 4 * Cg) / 9.0).astype("float32")
    return np.broadcast_to(img[None, ...], (batch, H, W, C)).copy()


def _gradient_image(H, W, C=1, batch=1):
    h = np.linspace(0, 1, H)
    w = np.linspace(0, 1, W)
    c = np.linspace(0, 1, C) if C > 1 else np.array([0.5])
    Hg, Wg, Cg = np.meshgrid(h, w, c, indexing="ij")
    img = ((2 * Hg + 3 * Wg + 4 * Cg) / 9.0).astype("float32")
    return np.broadcast_to(img[None, ...], (batch, H, W, C)).copy()


class ReconstructPatches2DTest(testing.TestCase):
    @parameterized.parameters(
        # (H, W, C, size, padding)
        (64, 64, 1, (8, 8), "valid"),
        (64, 64, 3, (8, 8), "valid"),
        (32, 48, 1, (4, 8), "valid"),
        (59, 55, 3, (8, 8), "same"),
        (33, 41, 2, (5, 7), "same"),
        (7, 11, 1, (3, 5), "same"),
    )
    def test_extract_then_reconstruct_roundtrip(self, H, W, C, size, padding):
        x = _gradient_image(H, W, C, batch=2)
        x_t = ops.convert_to_tensor(x)
        patches = ops.image.extract_patches(x_t, size=size, padding=padding)
        layer = layers.ReconstructPatches2D(
            size=size, output_size=(H, W), padding=padding,
        )
        recon = layer(patches)
        self.assertEqual(tuple(recon.shape), x.shape)
        self.assertAllClose(recon, x, atol=1e-6)

    def test_dynamic_spatial_dim(self):
        size = (4, 4)
        flat = size[0] * size[1] * 3
        input_layer = layers.Input(batch_shape=(1, None, None, flat))
        recon = layers.ReconstructPatches2D(
            size=size, output_size=(16, 16), padding="valid",
        )(input_layer)
        self.assertEqual(recon.shape, (1, 16, 16, 3))

    def test_get_config(self):
        layer = layers.ReconstructPatches2D(
            size=(3, 4), output_size=(12, 16), padding="valid",
        )
        config = layer.get_config()
        restored = layers.ReconstructPatches2D.from_config(config)
        self.assertEqual(restored.size, (3, 4))
        self.assertEqual(restored.output_size, (12, 16))
        self.assertEqual(restored.padding, "valid")

    def test_invalid_size(self):
        with self.assertRaisesRegex(ValueError, "length 2"):
            layers.ReconstructPatches2D(size=(2, 3, 4), output_size=(10, 15))

    def test_invalid_padding(self):
        with self.assertRaisesRegex(ValueError, "'same' or 'valid'"):
            layers.ReconstructPatches2D(
                size=(2, 2), output_size=(8, 8), padding="reflect",
            )

    def test_gapped_strides_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "gapped"):
            layers.ReconstructPatches2D(
                size=(4, 4), output_size=(32, 32),
                strides=(8, 8), padding="valid",
            )

    def test_overlapping_strides_supported(self):
        x = _gradient_image_2d(16, 16, C=3, batch=2)
        x_t = ops.convert_to_tensor(x)
        patches = ops.image.extract_patches(
            x_t, size=(4, 4), strides=2, padding="valid",
        )
        recon = layers.ReconstructPatches2D(
            size=(4, 4), output_size=(16, 16),
            strides=(2, 2), padding="valid",
        )(patches)
        self.assertAllClose(recon, x, atol=1e-5)

    @pytest.mark.skipif(
        backend.backend() == "tensorflow",
        reason="extract_patches with channels_first requires NCHW conv which "
               "is unavailable on tensorflow-cpu.",
    )
    def test_channels_first_roundtrip(self):
        x = np.random.RandomState(0).rand(2, 3, 16, 16).astype("float32")
        x_t = ops.convert_to_tensor(x)
        patches = ops.image.extract_patches(
            x_t, size=(4, 4), padding="valid", data_format="channels_first",
        )
        recon = layers.ReconstructPatches2D(
            size=(4, 4), output_size=(16, 16), padding="valid",
            data_format="channels_first",
        )(patches)
        self.assertEqual(tuple(recon.shape), x.shape)
        self.assertAllClose(recon, x, atol=1e-6)

    def test_dual_input_dynamic_output_size(self):
        x = np.random.RandomState(1).rand(2, 17, 19, 3).astype("float32")
        x_t = ops.convert_to_tensor(x)
        patches = ops.image.extract_patches(x_t, size=(4, 4), padding="same")
        layer = layers.ReconstructPatches2D(size=(4, 4), padding="same")
        recon = layer([patches, x_t])
        self.assertAllClose(recon, x, atol=1e-6)

    def test_reduction_sum_overshoots_at_overlap(self):
        x = np.ones((1, 16, 16, 3), dtype="float32")
        x_t = ops.convert_to_tensor(x)
        patches = ops.image.extract_patches(
            x_t, size=(4, 4), strides=2, padding="valid",
        )
        recon_sum = layers.ReconstructPatches2D(
            size=(4, 4), output_size=(16, 16),
            strides=(2, 2), padding="valid", reduction="sum",
        )(patches)
        self.assertGreater(float(ops.max(recon_sum)), 1.5)

    def test_auto_infer_output_size_valid(self):
        x = np.random.RandomState(2).rand(1, 16, 16, 3).astype("float32")
        x_t = ops.convert_to_tensor(x)
        patches = ops.image.extract_patches(x_t, size=(4, 4), padding="valid")
        layer = layers.ReconstructPatches2D(size=(4, 4), padding="valid")
        recon = layer(patches)
        self.assertAllClose(recon, x, atol=1e-6)
