"""Kinematic + part-wise Hankel-DMD / Koopman fusion (proposed ISL model).

Koopman / Hankel-DMD features are unchanged (kdfv2 cache). The classifier is not
a 2M ST-GCN+GAP+FiLM stack — that overfit 7-shot (val−test gap ~0.33). v3 is a
compact joint+bone graph encoder that keeps time, a temporal transformer (the
family that actually wins this protocol), and concat fusion of Koopman tokens.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import PositionalEncoding
from .skeleton import N_JOINTS, PARTS, bone_pairs, spatial_partition_adjacency
from .stgcn import STGCNBlock

CACHE_TAG = "kdfv2"
N_MODES = 3  # per body part
N_DELAYS = 8
PART_ORDER = ("upper", "left_hand", "right_hand")
N_PARTS = len(PART_ORDER)
KIN_CHANNELS = 3
KDF_IN_CHANNELS = KIN_CHANNELS
FEAT_PER_MODE = 4  # log1p|λ|, sin θ, cos θ, RMS amplitude
EIG_DIM = N_PARTS * N_MODES * FEAT_PER_MODE + N_PARTS
MODE_MAP_DIM = N_PARTS * N_MODES  # 9 spatial-mode channels over joints


def _kalman_nd(z: np.ndarray, q: float = 8e-4, r: float = 1.5e-2) -> np.ndarray:
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


def kalman_smooth_joints(joints: np.ndarray, q: float = 8e-4, r: float = 1.5e-2) -> np.ndarray:
    """Occlusion fill + forward–backward Kalman on (T, V, C) trajectories."""
    t_len, n_j, n_c = joints.shape
    missing = np.linalg.norm(joints, axis=-1) < 1e-6
    filled = _interp_missing_matrix(joints, missing).reshape(t_len, n_j * n_c)
    fwd = _kalman_nd(filled, q=q, r=r)
    bwd = _kalman_nd(fwd[::-1], q=q, r=r)[::-1]
    return (0.5 * (fwd + bwd)).reshape(t_len, n_j, n_c).astype(np.float32)


def normalize_joints(joints: np.ndarray) -> np.ndarray:
    """Root-center at the nose and scale by mean shoulder width."""
    x = joints.astype(np.float32, copy=True)
    x = x - x[:, :1, :]
    shoulder = np.linalg.norm(x[:, 1] - x[:, 2], axis=-1)
    scale = float(np.nanmean(shoulder))
    if not np.isfinite(scale) or scale < 1e-4:
        scale = float(np.std(x) * 2.0 + 1e-3)
    return (x / scale).astype(np.float32)


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

    x: (D, T)  →  |spatial| (D, r), |amp| (r, T), eigs (r,)
    Modes are ordered oscillatory-first so clips align better than raw |λ|.
    """
    x64 = np.asarray(x, dtype=np.float64)
    dim, t_len = x64.shape
    delays = int(max(2, min(int(delays), max(2, t_len // 3))))
    empty = (
        np.zeros((dim, rank), dtype=np.float32),
        np.zeros((rank, t_len), dtype=np.float32),
        np.zeros(rank, dtype=np.complex64),
    )
    if t_len < delays + 2 or dim < 1:
        return empty
    h0 = _hankel(x64[:, :-1], delays)
    h1 = _hankel(x64[:, 1:], delays)
    cols = min(h0.shape[1], h1.shape[1])
    if cols < 2:
        return empty
    h0, h1 = h0[:, :cols], h1[:, :cols]
    u, s, vh = np.linalg.svd(h0, full_matrices=False)
    r = int(min(rank, u.shape[1], max(1, int((s > 1e-6).sum()))))
    u, s, vh = u[:, :r], s[:r], vh[:r]
    k = u.T @ h1 @ vh.T @ np.diag(1.0 / np.clip(s, 1e-8, None))
    eigvals, eigvecs = np.linalg.eig(k)
    order = np.lexsort((-np.abs(eigvals), -np.abs(np.angle(eigvals))))
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    phi = u @ eigvecs
    spatial_c = phi[:dim, :r]
    mag = np.clip(np.abs(eigvals), 0.5, 1.5)
    lam = mag * np.exp(1j * np.angle(eigvals))
    try:
        b0 = np.linalg.pinv(spatial_c, rcond=1e-6) @ x64[:, 0]
    except np.linalg.LinAlgError:
        b0 = np.zeros(r, dtype=np.complex128)
    t_idx = np.arange(t_len, dtype=np.float64)
    amp = np.zeros((r, t_len), dtype=np.float64)
    for i in range(r):
        amp[i] = np.abs(b0[i] * np.power(lam[i], t_idx))
    spatial = np.abs(spatial_c).astype(np.float32)
    if spatial.shape[1] < rank:
        pad = rank - spatial.shape[1]
        spatial = np.pad(spatial, ((0, 0), (0, pad)))
        amp = np.pad(amp, ((0, pad), (0, 0)))
        eigvals = np.concatenate([eigvals, np.zeros(pad, dtype=np.complex128)])
    return (
        spatial[:, :rank],
        amp[:rank].astype(np.float32),
        np.asarray(eigvals[:rank], dtype=np.complex64),
    )


def _part_spectrum(eigs: np.ndarray, amp: np.ndarray, n_modes: int) -> np.ndarray:
    """log|λ|, sin θ, cos θ, RMS amp for one body part. Shape (n_modes * 4,)."""
    out = np.zeros(n_modes * FEAT_PER_MODE, dtype=np.float32)
    z = np.asarray(eigs).ravel()
    for i in range(n_modes):
        base = i * FEAT_PER_MODE
        out[base + 2] = 1.0
        if i >= z.size:
            continue
        val = complex(z[i])
        ang = np.angle(val)
        out[base] = np.log1p(abs(val))
        out[base + 1] = np.sin(ang)
        out[base + 2] = np.cos(ang)
        if amp is not None and i < amp.shape[0]:
            out[base + 3] = float(np.sqrt(np.mean(np.square(amp[i]))))
    return out


def kdf_joint_features(
    joints: np.ndarray,
    n_modes: int = N_MODES,
    delays: int = N_DELAYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kalman + signer-normalized pose, part-wise Hankel-DMD.

    joints: (T, V, 3)
    returns:
      x:     (3, T, V)  Kalman-smoothed, root-centered pose
      eig:   (EIG_DIM,) part-wise Koopman spectrum + part kinetic energy
      modes: (V, MODE_MAP_DIM) spatial graph-mode energy per joint
    """
    joints = np.asarray(joints, dtype=np.float32)
    t_len, n_j, n_c = joints.shape
    smooth = normalize_joints(kalman_smooth_joints(joints))
    vel = np.diff(smooth, axis=0, prepend=smooth[:1])
    mode_map = np.zeros((n_j, N_PARTS * n_modes), dtype=np.float32)
    spec_parts: list[np.ndarray] = []
    kinetic: list[float] = []
    for p, name in enumerate(PART_ORDER):
        idx = list(PARTS[name])
        sub = smooth[:, idx, :]
        spatial, amp, eigs = hankel_dmd(sub.reshape(t_len, -1).T, delays=delays, rank=n_modes)
        vp = len(idx)
        spat = spatial.reshape(vp, n_c, -1).mean(axis=1)  # (Vp, r)
        for i in range(n_modes):
            col = spat[:, i]
            peak = float(col.max()) + 1e-6
            mode_map[idx, p * n_modes + i] = col / peak
        spec_parts.append(_part_spectrum(eigs, amp, n_modes))
        kinetic.append(float(np.sqrt(np.mean(np.square(vel[:, idx])))))
    eig = np.concatenate([*spec_parts, np.asarray(kinetic, dtype=np.float32)], axis=0)
    x = np.transpose(smooth, (2, 0, 1)).astype(np.float32)
    return x, eig.astype(np.float32), mode_map.astype(np.float32)


class KDFSTGCN(nn.Module):
    """Shared joint/bone GCN (no temporal stride) + transformer + Koopman concat."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = KDF_IN_CHANNELS,
        n_modes: int = N_MODES,
        eig_hidden: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        layers: int = 2,
        dropout: float = 0.25,
        mixup: float = 0.2,
        label_smoothing: float = 0.1,
        **_ignored,
    ):
        super().__init__()
        self.n_modes = int(n_modes)
        self.mixup = float(mixup)
        self.label_smoothing = float(label_smoothing)
        a = torch.from_numpy(spatial_partition_adjacency())
        self.data_bn = nn.BatchNorm1d(in_channels * N_JOINTS)
        self.gcn = nn.Sequential(
            STGCNBlock(in_channels, 64, a, residual=False, dropout=dropout),
            STGCNBlock(64, 64, a, dropout=dropout),
            STGCNBlock(64, 128, a, dropout=dropout),
        )
        self.stream_fuse = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.part_fuse = nn.Sequential(
            nn.Conv1d(128 * N_PARTS, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(d_model, max_len=64, dropout=dropout)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(enc, num_layers=int(layers), enable_nested_tensor=False)
        self.pool_score = nn.Linear(d_model, 1)
        mode_in = N_PARTS * MODE_MAP_DIM
        self.mode_mlp = nn.Sequential(
            nn.LayerNorm(mode_in),
            nn.Linear(mode_in, eig_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(eig_hidden, 64),
        )
        self.eig_mlp = nn.Sequential(
            nn.LayerNorm(EIG_DIM),
            nn.Linear(EIG_DIM, eig_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(eig_hidden, 64),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model + 128),
            nn.Dropout(dropout),
            nn.Linear(d_model + 128, num_classes),
        )
        pairs = bone_pairs()
        self.register_buffer("_bone_src", torch.tensor([p for p, _ in pairs], dtype=torch.long), persistent=False)
        self.register_buffer("_bone_dst", torch.tensor([c for _, c in pairs], dtype=torch.long), persistent=False)
        for name in PART_ORDER:
            self.register_buffer(f"_idx_{name}", torch.tensor(PARTS[name], dtype=torch.long), persistent=False)

    def _bones(self, x: torch.Tensor) -> torch.Tensor:
        bone = torch.zeros_like(x)
        for src, dst in zip(self._bone_src.tolist(), self._bone_dst.tolist()):
            bone[:, :, :, dst] = x[:, :, :, dst] - x[:, :, :, src]
        return bone

    def _part_tokens(self, x: torch.Tensor) -> torch.Tensor:
        chunks = [x.index_select(-1, getattr(self, f"_idx_{name}")).mean(-1) for name in PART_ORDER]
        stacked = torch.cat(chunks, dim=1)  # (N, 128*P, T)
        return self.part_fuse(stacked).transpose(1, 2)  # (N, T, d_model)

    def _part_mode_vec(self, modes: torch.Tensor) -> torch.Tensor:
        parts = [modes.index_select(1, getattr(self, f"_idx_{name}")).mean(1) for name in PART_ORDER]
        return torch.cat(parts, dim=-1)

    def encode_motion(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.shape
        x = self.data_bn(x.reshape(n, c * v, t)).reshape(n, c, t, v)
        joint = self.gcn(x)
        bone = self.gcn(self._bones(x))
        fused = self.stream_fuse(torch.cat([joint, bone], dim=1))
        tokens = self.pos(self._part_tokens(fused))
        tokens = self.temporal(tokens)
        weights = torch.softmax(self.pool_score(tokens), dim=1)
        return (tokens * weights).sum(dim=1) + tokens.mean(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        eig: torch.Tensor | None = None,
        modes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kin = self.encode_motion(x)
        if eig is None:
            eig = x.new_zeros(x.size(0), EIG_DIM)
        if modes is None:
            modes = x.new_zeros(x.size(0), N_JOINTS, MODE_MAP_DIM)
        dyn = torch.cat([self.eig_mlp(eig), self.mode_mlp(self._part_mode_vec(modes))], dim=-1)
        return self.head(torch.cat([kin, dyn], dim=-1))


def _ce(logits, y, criterion, smoothing: float):
    return F.cross_entropy(
        logits,
        y,
        weight=getattr(criterion, "weight", None),
        label_smoothing=float(smoothing),
    )


def kdf_forward(model, batch, criterion, device, train: bool = True):
    """Unpack (pose, koopman_eigs, spatial_modes); mixup while training."""
    x, y = batch
    if isinstance(x, (tuple, list)):
        joints, eig = x[0], x[1]
        modes = x[2] if len(x) > 2 else None
        joints = joints.to(device, non_blocking=True)
        eig = eig.to(device, non_blocking=True)
        if modes is not None:
            modes = modes.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
    else:
        joints = x.to(device, non_blocking=True)
        eig, modes = None, None
        y = y.to(device, non_blocking=True)

    mix = float(getattr(model, "mixup", 0.0) or 0.0) if train and model.training else 0.0
    smooth = float(getattr(model, "label_smoothing", 0.0) or 0.0) if train else 0.0
    if mix > 0.0 and joints.size(0) > 1:
        lam = float(np.random.beta(mix, mix))
        idx = torch.randperm(joints.size(0), device=device)
        joints = lam * joints + (1.0 - lam) * joints[idx]
        if eig is not None:
            eig = lam * eig + (1.0 - lam) * eig[idx]
        if modes is not None:
            modes = lam * modes + (1.0 - lam) * modes[idx]
        logits = model(joints, eig, modes)
        loss = lam * _ce(logits, y, criterion, smooth) + (1.0 - lam) * _ce(logits, y[idx], criterion, smooth)
        return logits, y, loss

    logits = model(joints, eig, modes)
    return logits, y, _ce(logits, y, criterion, smooth)
