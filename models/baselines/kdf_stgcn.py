"""Kinematic + Hankel-DMD / Koopman fusion into ST-GCN (proposed ISL model).

Pipeline (matches the comparison-deck novel row):
  Kalman-smoothed landmarks → delay-embedded Hankel-DMD → Koopman eigendecomposition
  → fuse pos/vel/acc with spatial graph modes → ST-GCN, plus an eigenvalue head
    (growth/decay + frequency) that FFT/CWT cannot represent.

Input tensor is (N, C, T, V) with C = 9 + 3*n_modes (kinematics + reconstructed
modes). Eigenvalues are a separate (N, 2*n_modes) vector: log|λ| and arg(λ)/π.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .stgcn import STGCN

N_MODES = 4
N_DELAYS = 4
KIN_CHANNELS = 9  # xyz + vel + acc
MODE_CHANNELS = 3 * N_MODES
KDF_IN_CHANNELS = KIN_CHANNELS + MODE_CHANNELS  # 21
EIG_DIM = 2 * N_MODES  # log magnitude + wrapped angle


def _kalman_nd(z: np.ndarray, q: float = 3e-4, r: float = 2e-2) -> np.ndarray:
    """Independent constant-velocity Kalman on each column. z: (T, N) → (T, N)."""
    t_len, n = z.shape
    z64 = np.asarray(z, dtype=np.float64).reshape(t_len, n)
    pos = z64[0].copy()
    vel = np.zeros(n, dtype=np.float64)
    p11 = np.ones(n, dtype=np.float64)
    p12 = np.zeros(n, dtype=np.float64)
    p22 = np.ones(n, dtype=np.float64)
    q11, q12, q22 = 0.25 * q, 0.5 * q, q
    out = np.empty_like(z64)
    r = float(r)
    for t in range(t_len):
        pos = pos + vel
        p11 = p11 + 2.0 * p12 + p22 + q11
        p12 = p12 + p22 + q12
        p22 = p22 + q22
        innov = z64[t] - pos
        s = np.maximum(p11 + r, 1e-12)
        k0 = p11 / s
        k1 = p12 / s
        pos = pos + k0 * innov
        vel = vel + k1 * innov
        n11 = (1.0 - k0) * p11
        n12 = (1.0 - k0) * p12
        n21 = -k1 * p11 + p12
        n22 = -k1 * p12 + p22
        p11 = n11
        p12 = 0.5 * (n12 + n21)
        p22 = n22
        out[t] = pos
    return out.astype(np.float32)


def _interp_missing_matrix(joints: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Fill occluded joints along time. joints: (T, V, C), missing: (T, V)."""
    out = joints.astype(np.float32, copy=True)
    t_idx = np.arange(joints.shape[0], dtype=np.float64)
    for v in range(joints.shape[1]):
        miss = missing[:, v]
        if not bool(miss.any()) or bool(miss.all()):
            continue
        good = ~miss
        for c in range(joints.shape[2]):
            out[:, v, c] = np.interp(t_idx, t_idx[good], joints[good, v, c].astype(np.float64))
    return out


def kalman_smooth_joints(joints: np.ndarray, q: float = 3e-4, r: float = 2e-2) -> np.ndarray:
    """Occlusion fill + forward–backward Kalman on (T, V, C) trajectories."""
    t_len, n_j, n_c = joints.shape
    missing = np.linalg.norm(joints, axis=-1) < 1e-6
    filled = _interp_missing_matrix(joints, missing).reshape(t_len, n_j * n_c)
    fwd = _kalman_nd(filled, q=q, r=r)
    bwd = _kalman_nd(fwd[::-1], q=q, r=r)[::-1]
    return (0.5 * (fwd + bwd)).reshape(t_len, n_j, n_c).astype(np.float32)


def _hankel(x: np.ndarray, delays: int) -> np.ndarray:
    """x: (D, T) → Hankel (D*delays, T-delays+1)."""
    dim, t_len = x.shape
    cols = t_len - delays + 1
    h = np.empty((dim * delays, cols), dtype=np.float64)
    for i in range(delays):
        h[i * dim : (i + 1) * dim] = x[:, i : i + cols]
    return h


def hankel_dmd(x: np.ndarray, delays: int = N_DELAYS, rank: int = N_MODES):
    """Exact DMD on a delay-embedded Hankel matrix.

    x: (D, T)  →  spatial modes (D, r), time coeffs (r, T), eigs (r,)
    """
    x64 = np.asarray(x, dtype=np.float64)
    dim, t_len = x64.shape
    delays = int(max(2, min(int(delays), max(2, t_len // 3))))
    if t_len < delays + 2 or dim < 1:
        return (
            np.zeros((dim, rank), dtype=np.float32),
            np.zeros((rank, t_len), dtype=np.float32),
            np.zeros(rank, dtype=np.complex64),
        )
    h0 = _hankel(x64[:, :-1], delays)
    h1 = _hankel(x64[:, 1:], delays)
    cols = min(h0.shape[1], h1.shape[1])
    if cols < 2:
        return (
            np.zeros((dim, rank), dtype=np.float32),
            np.zeros((rank, t_len), dtype=np.float32),
            np.zeros(rank, dtype=np.complex64),
        )
    h0, h1 = h0[:, :cols], h1[:, :cols]
    u, s, vh = np.linalg.svd(h0, full_matrices=False)
    r = int(min(rank, u.shape[1], max(1, int((s > 1e-6).sum()))))
    u, s, vh = u[:, :r], s[:r], vh[:r]
    k = u.T @ h1 @ vh.T @ np.diag(1.0 / np.clip(s, 1e-8, None))
    eigvals, eigvecs = np.linalg.eig(k)
    order = np.argsort(-np.abs(eigvals))
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    phi = np.real(u @ eigvecs)  # (D*delays, r) delay-embedded spatial modes
    spatial = phi[:dim, :r]
    amp = np.linalg.pinv(spatial) @ x64  # (r, T)
    if spatial.shape[1] < rank:
        pad = rank - spatial.shape[1]
        spatial = np.pad(spatial, ((0, 0), (0, pad)))
        amp = np.pad(amp, ((0, pad), (0, 0)))
        eigvals = np.concatenate([eigvals, np.zeros(pad, dtype=np.complex128)])
    return (
        spatial[:, :rank].astype(np.float32),
        amp[:rank].astype(np.float32),
        np.asarray(eigvals[:rank], dtype=np.complex64),
    )


def eig_features(eigs: np.ndarray, n_modes: int = N_MODES) -> np.ndarray:
    """log|λ| (growth/decay) and arg(λ)/π (frequency). Shape (2 * n_modes,)."""
    z = np.zeros(n_modes, dtype=np.complex64)
    n = min(n_modes, int(np.asarray(eigs).size))
    z[:n] = np.asarray(eigs, dtype=np.complex64).ravel()[:n]
    mag = np.log1p(np.abs(z)).astype(np.float32)
    ang = (np.angle(z) / np.pi).astype(np.float32)
    return np.concatenate([mag, ang], axis=0)


def kdf_joint_features(
    joints: np.ndarray,
    n_modes: int = N_MODES,
    delays: int = N_DELAYS,
) -> tuple[np.ndarray, np.ndarray]:
    """Kalman + kinematics + Hankel-DMD spatial modes.

    joints: (T, V, 3)
    returns:
      x:   (C, T, V) with C = 9 + 3*n_modes
      eig: (2*n_modes,)
    """
    joints = np.asarray(joints, dtype=np.float32)
    t_len, n_j, n_c = joints.shape
    smooth = kalman_smooth_joints(joints)
    vel = np.diff(smooth, axis=0, prepend=smooth[:1])
    acc = np.diff(vel, axis=0, prepend=vel[:1])
    kin = np.concatenate([smooth, vel, acc], axis=-1)  # (T, V, 9)

    flat = smooth.reshape(t_len, n_j * n_c).T  # (D, T)
    spatial, amp, eigs = hankel_dmd(flat, delays=delays, rank=n_modes)
    # spatial: (D, r) → (V, 3, r); amp: (r, T)
    modes = np.zeros((t_len, n_j, 3 * n_modes), dtype=np.float32)
    for i in range(n_modes):
        pattern = spatial[:, i].reshape(n_j, n_c)  # (V, 3)
        coeff = amp[i]  # (T,)
        modes[:, :, i * 3 : (i + 1) * 3] = pattern[None, :, :] * coeff[:, None, None]

    fused = np.concatenate([kin, modes], axis=-1)  # (T, V, C)
    x = np.transpose(fused, (2, 0, 1)).astype(np.float32)  # (C, T, V)
    return x, eig_features(eigs, n_modes=n_modes)


class KDFSTGCN(nn.Module):
    """ST-GCN on fused kinematic/DMD joint channels + Koopman eigenvalue head."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = KDF_IN_CHANNELS,
        n_modes: int = N_MODES,
        eig_hidden: int = 32,
        dropout: float = 0.2,
        **_ignored,
    ):
        super().__init__()
        self.n_modes = int(n_modes)
        self.backbone = STGCN(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
        self.eig_mlp = nn.Sequential(
            nn.LayerNorm(2 * self.n_modes),
            nn.Linear(2 * self.n_modes, eig_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fuse = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 + eig_hidden, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.shape
        x = self.backbone.data_bn(x.reshape(n, c * v, t)).reshape(n, c, t, v)
        x = self.backbone.blocks(x)
        return F.adaptive_avg_pool2d(x, 1).reshape(n, -1)

    def forward(self, x: torch.Tensor, eig: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.encode(x)
        if eig is None:
            return self.backbone.head(feat)
        return self.fuse(torch.cat([feat, self.eig_mlp(eig)], dim=-1))


def kdf_forward(model, batch, criterion, device, train: bool = True):
    """Unpack (joint_features, koopman_eigs) from the KDF collate."""
    x, y = batch
    if isinstance(x, (tuple, list)):
        joints, eig = x
        joints = joints.to(device, non_blocking=True)
        eig = eig.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(joints, eig)
    else:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
    loss = criterion(logits, y)
    return logits, y, loss
