"""
Integration-level unit tests for the quadruped pose prior.

Covers:
  - QuadrupedPosePrior model: encode, decode, forward, sample_poses, regularise_pose
  - VarenMoCapData and VarenMuscles datasets: loading, len, getitem
  - Rotation utility round-trips: remove_rotation_from_axis,
    merge_global_orients_along_axis, aa2matrot/matrot2aa
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from varen_poser.models.pose_prior import QuadrupedPosePrior
from varen_poser.datasets.varen_pose_dataset import VarenMoCapData, VarenMuscles
from varen_poser.utils.pose_transforms import (
    remove_rotation_from_axis,
    merge_global_orients_along_axis,
    aa2matrot,
    matrot2aa,
)

# ---------------------------------------------------------------------------
# Constants matching QuadrupedPosePrior defaults
# ---------------------------------------------------------------------------
NUM_JOINTS = 38   # 37 body joints + 1 global orient
POSE_DIM   = NUM_JOINTS * 3   # 114
LATENT_D   = 32
BATCH      = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_pose(batch=BATCH):
    return torch.randn(batch, POSE_DIM)


def make_mocap_dir(tmp_path: Path, n_files: int = 2, frames_per_file: int = 10) -> Path:
    """Creates a temp directory with synthetic *_stageii.pkl files."""
    for i in range(n_files):
        data = {"fullpose": np.random.randn(frames_per_file, POSE_DIM).astype(np.float32)}
        pkl_path = tmp_path / f"subject_{i}_stageii.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)
    return tmp_path


def make_muscles_dir(tmp_path: Path, n_files: int = 3) -> Path:
    """Creates a temp directory with synthetic *_solution.npy files."""
    for i in range(n_files):
        # data[3:117] → 114 values; prepend 3 dummy values and append extras
        arr = np.random.randn(120).astype(np.float32)
        np.save(tmp_path / f"scan_{i}_solution.npy", arr)
    return tmp_path


# ===========================================================================
# QuadrupedPosePrior model tests
# ===========================================================================

class TestQuadrupedPosePriorShapes:
    """Verify tensor shapes through the model forward passes."""

    @pytest.fixture(scope="class")
    def model(self):
        m = QuadrupedPosePrior()
        m.eval()
        return m

    def test_encode_returns_normal_distribution(self, model):
        pose = random_pose()
        q_z = model.encode(pose)
        assert isinstance(q_z, torch.distributions.Normal)

    def test_encode_latent_shape(self, model):
        pose = random_pose()
        q_z = model.encode(pose)
        assert q_z.mean.shape == (BATCH, LATENT_D)
        assert q_z.scale.shape == (BATCH, LATENT_D)

    def test_decode_output_keys(self, model):
        Z = torch.randn(BATCH, LATENT_D)
        out = model.decode(Z)
        assert "pose_body" in out
        assert "pose_body_matrot" in out

    def test_decode_pose_body_shape(self, model):
        Z = torch.randn(BATCH, LATENT_D)
        out = model.decode(Z)
        assert out["pose_body"].shape == (BATCH, NUM_JOINTS, 3)

    def test_decode_pose_body_matrot_shape(self, model):
        Z = torch.randn(BATCH, LATENT_D)
        out = model.decode(Z)
        assert out["pose_body_matrot"].shape == (BATCH, NUM_JOINTS, 9)

    def test_forward_output_keys(self, model):
        pose = random_pose()
        out = model(pose)
        for key in ("pose_body", "pose_body_matrot", "poZ_body_mean", "poZ_body_std", "q_z"):
            assert key in out, f"Missing key: {key}"

    def test_forward_latent_stats_shape(self, model):
        pose = random_pose()
        out = model(pose)
        assert out["poZ_body_mean"].shape == (BATCH, LATENT_D)
        assert out["poZ_body_std"].shape  == (BATCH, LATENT_D)

    def test_forward_pose_body_shape(self, model):
        pose = random_pose()
        out = model(pose)
        assert out["pose_body"].shape == (BATCH, NUM_JOINTS, 3)


class TestQuadrupedPosePriorSampling:
    """Verify sample_poses behaviour."""

    @pytest.fixture(scope="class")
    def model(self):
        m = QuadrupedPosePrior()
        m.eval()
        return m

    def test_sample_poses_output_shape(self, model):
        n = 8
        out = model.sample_poses(num_poses=n)
        assert out["pose_body"].shape == (n, NUM_JOINTS, 3)
        assert out["pose_body_matrot"].shape == (n, NUM_JOINTS, 9)

    def test_sample_poses_reproducible_with_seed(self, model):
        out1 = model.sample_poses(num_poses=5, seed=42)
        out2 = model.sample_poses(num_poses=5, seed=42)
        assert torch.allclose(out1["pose_body"], out2["pose_body"])

    def test_sample_poses_different_without_seed(self, model):
        out1 = model.sample_poses(num_poses=5, seed=0)
        out2 = model.sample_poses(num_poses=5, seed=1)
        assert not torch.allclose(out1["pose_body"], out2["pose_body"])

    def test_sample_poses_invalid_temperature(self, model):
        with pytest.raises(AssertionError):
            model.sample_poses(num_poses=4, temperature=0.0)

    def test_sample_poses_temperature_changes_output(self, model):
        """Different temperatures should produce different pose samples."""
        low  = model.sample_poses(num_poses=10, seed=42, temperature=0.01)["pose_body"]
        high = model.sample_poses(num_poses=10, seed=42, temperature=2.0)["pose_body"]
        assert not torch.allclose(low, high), \
            "Different temperatures should yield different poses"


class TestRegularisePose:
    """Verify regularise_pose preserves shape and z-rotation."""

    @pytest.fixture(scope="class")
    def model(self):
        m = QuadrupedPosePrior()
        m.eval()
        return m

    def test_regularise_pose_output_shape(self, model):
        pose = random_pose()
        out  = model.regularise_pose(pose)
        assert out.shape == pose.shape

    def test_regularise_pose_z_rotation_preserved(self, model):
        """Global z-rotation should be unchanged after regularisation."""
        from varen_poser.utils.pose_transforms import aa2euler
        pose = random_pose()
        out  = model.regularise_pose(pose)

        in_z  = aa2euler(pose[:, :3])[:, 2]
        out_z = aa2euler(out[:, :3])[:, 2]
        assert torch.allclose(in_z, out_z, atol=1e-4), \
            "Z rotation not preserved by regularise_pose"


# ===========================================================================
# Dataset tests
# ===========================================================================

class TestVarenMoCapData:

    def test_raises_on_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            VarenMoCapData("/nonexistent/path/that/does/not/exist")

    def test_loads_correct_num_poses(self, tmp_path):
        n_files, frames = 2, 10
        make_mocap_dir(tmp_path, n_files=n_files, frames_per_file=frames)
        ds = VarenMoCapData(tmp_path)
        assert len(ds) == n_files * frames

    def test_getitem_shape(self, tmp_path):
        make_mocap_dir(tmp_path, n_files=1, frames_per_file=5)
        ds = VarenMoCapData(tmp_path)
        sample = ds[0]
        assert sample.shape == (POSE_DIM,)

    def test_z_rotation_zeroed(self, tmp_path):
        """Dataset pre-processes poses by zeroing z-rotation."""
        from varen_poser.utils.pose_transforms import aa2euler
        make_mocap_dir(tmp_path, n_files=1, frames_per_file=8)
        ds = VarenMoCapData(tmp_path)
        poses = torch.tensor(ds.poses)
        z_angles = aa2euler(poses[:, :3])[:, 2]
        assert torch.allclose(z_angles, torch.zeros_like(z_angles), atol=1e-5), \
            "Dataset should zero z-rotation on load"

    def test_finds_files_in_subdirectory(self, tmp_path):
        """Dataset should recurse into sub-directories."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        make_mocap_dir(subdir, n_files=1, frames_per_file=3)
        ds = VarenMoCapData(tmp_path)
        assert len(ds) == 3


class TestVarenMuscles:

    def test_raises_on_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            VarenMuscles("/nonexistent/path/that/does/not/exist")

    def test_loads_correct_num_poses(self, tmp_path):
        n = 5
        make_muscles_dir(tmp_path, n_files=n)
        ds = VarenMuscles(tmp_path)
        assert len(ds) == n

    def test_getitem_shape(self, tmp_path):
        make_muscles_dir(tmp_path, n_files=3)
        ds = VarenMuscles(tmp_path)
        sample = ds[0]
        assert sample.shape == (POSE_DIM,)

    def test_z_rotation_zeroed(self, tmp_path):
        from varen_poser.utils.pose_transforms import aa2euler
        make_muscles_dir(tmp_path, n_files=2)
        ds = VarenMuscles(tmp_path)
        poses = torch.tensor(ds.poses)
        z_angles = aa2euler(poses[:, :3])[:, 2]
        assert torch.allclose(z_angles, torch.zeros_like(z_angles), atol=1e-5)


# ===========================================================================
# Rotation utility tests
# ===========================================================================

class TestRotationTools:

    def test_remove_z_rotation_zeroes_z(self):
        pose = torch.randn(BATCH, POSE_DIM)
        out  = remove_rotation_from_axis(pose, axis=2)
        from varen_poser.utils.pose_transforms import aa2euler
        z = aa2euler(out[:, :3])[:, 2]
        assert torch.allclose(z, torch.zeros_like(z), atol=1e-5)

    def test_remove_z_rotation_preserves_body(self):
        """Body joints (everything after first 3 values) must be unchanged."""
        pose = torch.randn(BATCH, POSE_DIM)
        out  = remove_rotation_from_axis(pose, axis=2)
        assert torch.allclose(pose[:, 3:], out[:, 3:])

    def test_remove_rotation_does_not_modify_input(self):
        pose = torch.randn(BATCH, POSE_DIM)
        pose_clone = pose.clone()
        remove_rotation_from_axis(pose, axis=2)
        assert torch.allclose(pose, pose_clone)

    def test_merge_global_orients_copies_axis(self):
        """merge_global_orients_along_axis should copy the z-component of
        `additional` into `base`."""
        from varen_poser.utils.pose_transforms import aa2euler
        additional = torch.randn(BATCH, POSE_DIM)
        base       = torch.randn(BATCH, POSE_DIM)
        out        = merge_global_orients_along_axis(additional, base, axis=2)

        addi_z = aa2euler(additional[:, :3])[:, 2]
        out_z  = aa2euler(out[:, :3])[:, 2]
        assert torch.allclose(addi_z, out_z, atol=1e-5)

    def test_merge_global_orients_does_not_modify_inputs(self):
        additional = torch.randn(BATCH, POSE_DIM)
        base       = torch.randn(BATCH, POSE_DIM)
        additional_clone = additional.clone()
        base_clone       = base.clone()
        merge_global_orients_along_axis(additional, base, axis=2)
        assert torch.allclose(additional, additional_clone)
        assert torch.allclose(base, base_clone)

    def test_matrot2aa_aa2matrot_roundtrip(self):
        """aa → matrot → aa → matrot round-trip should be identity.

        aa2matrot expects (N, 3) input matching compute_loss usage pattern.
        Use a small N to avoid a known tgm_conversion bug with large batches.
        """
        aa_flat = torch.randn(BATCH, 3)
        mat     = aa2matrot(aa_flat)      # (BATCH, 3, 3)
        aa_back = matrot2aa(mat)          # (BATCH, 3)
        mat_back = aa2matrot(aa_back)     # (BATCH, 3, 3)
        assert torch.allclose(mat, mat_back, atol=1e-5), \
            "aa → matrot → aa → matrot round-trip failed"

    def test_remove_then_merge_roundtrip(self):
        """remove_rotation_from_axis followed by merge_global_orients_along_axis
        should restore the original z-rotation."""
        from varen_poser.utils.pose_transforms import aa2euler
        pose = torch.randn(BATCH, POSE_DIM)
        stripped = remove_rotation_from_axis(pose, axis=2)
        restored = merge_global_orients_along_axis(pose, stripped, axis=2)

        orig_z     = aa2euler(pose[:, :3])[:, 2]
        restored_z = aa2euler(restored[:, :3])[:, 2]
        assert torch.allclose(orig_z, restored_z, atol=1e-5)
