# -*- coding: utf-8 -*-
"""Regression models for SpatialChord-Reg.

This module is intentionally small and reuses the existing scChord spatial
attention block instead of forking graph logic.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyGate(nn.Module):
    """Gate RNA features by posterior variance."""

    def __init__(self, var_dim: int = 64, feat_dim: int = 256, momentum: float = 0.01):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(var_dim, feat_dim), nn.Sigmoid())
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(feat_dim))

    def forward(self, h_rna: torch.Tensor, rna_var: torch.Tensor) -> torch.Tensor:
        g = self.gate(rna_var)
        if self.training:
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(
                    h_rna.mean(0) * self.momentum
                )
        return g * h_rna


class FourierSpatialEncoding(nn.Module):
    """Fourier features for relative spatial displacement vectors."""

    def __init__(self, n_freq: int = 8, out_dim: int = 64):
        super().__init__()
        self.n_freq = n_freq
        self.out_dim = out_dim
        freqs = torch.linspace(0.5, 4.0, n_freq)
        self.register_buffer("freq_r", freqs.clone())
        self.register_buffer("freq_theta", freqs.clone())
        self.proj = nn.Linear(4 * n_freq, out_dim)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        r = torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp(min=1e-6)
        theta = torch.atan2(delta[..., 1:2], delta[..., 0:1])
        fr, fth = self.freq_r, self.freq_theta
        sr = torch.sin(r * fr)
        cr = torch.cos(r * fr)
        st = torch.sin(theta * fth)
        ct = torch.cos(theta * fth)
        enc = torch.cat([sr, cr, st, ct], dim=-1)
        return self.proj(enc)


class SpatialNeighborAttention(nn.Module):
    """Single-layer spatial neighbor attention over local kNN subgraphs."""

    def __init__(
        self,
        dim: int = 256,
        n_heads: int = 8,
        max_neighbors: int = 16,
        pos_dim: int = 64,
        dropout: float = 0.1,
        n_freq: int = 8,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_neighbors = max_neighbors

        self.pos_enc = FourierSpatialEncoding(n_freq=n_freq, out_dim=pos_dim)
        self.k_pos_proj = nn.Linear(pos_dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim * 2, dim)
        self.out_ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    @staticmethod
    def _pad_neighbor_tensors(
        edge_index: torch.Tensor,
        coords: torch.Tensor,
        h_feat: torch.Tensor,
        max_neighbors: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]
        device = h_feat.device
        dtype = h_feat.dtype
        n_nodes = h_feat.shape[0]
        dim = h_feat.shape[1]
        k_neighbors = max_neighbors

        if edge_index.numel() == 0:
            z = torch.zeros(n_nodes, k_neighbors, dim, device=device, dtype=dtype)
            dlt = torch.zeros(n_nodes, k_neighbors, 2, device=device, dtype=coords.dtype)
            m = torch.zeros(n_nodes, k_neighbors, dtype=torch.bool, device=device)
            return z, dlt, m

        sorted_dst, perm = torch.sort(dst)
        sorted_src = src[perm]
        n_edges = sorted_dst.numel()
        counts = torch.bincount(sorted_dst, minlength=n_nodes)
        ptr = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)]
        )
        start_edge = ptr[sorted_dst]
        local_idx = torch.arange(n_edges, device=device, dtype=torch.long) - start_edge
        valid = local_idx < k_neighbors

        nb_feat = torch.zeros(n_nodes, k_neighbors, dim, device=device, dtype=dtype)
        delta = torch.zeros(n_nodes, k_neighbors, 2, device=device, dtype=coords.dtype)
        mask = torch.zeros(n_nodes, k_neighbors, dtype=torch.bool, device=device)

        d_idx = sorted_dst[valid]
        l_idx = local_idx[valid]
        s_idx = sorted_src[valid]
        nb_feat[d_idx, l_idx] = h_feat[s_idx]
        delta[d_idx, l_idx] = coords[s_idx] - coords[d_idx]
        mask[d_idx, l_idx] = True
        return nb_feat, delta, mask

    def forward(
        self,
        h_fused: torch.Tensor,
        edge_index: torch.Tensor,
        coords_um: torch.Tensor,
    ) -> torch.Tensor:
        n_nodes, dim = h_fused.shape
        k_neighbors = self.max_neighbors
        n_heads, head_dim = self.n_heads, self.head_dim

        nb_feat, delta, mask = self._pad_neighbor_tensors(
            edge_index, coords_um, h_fused, k_neighbors
        )
        pos_k = self.k_pos_proj(self.pos_enc(delta))
        k_nb = self.k_proj(nb_feat) + pos_k
        v_nb = self.v_proj(nb_feat)
        q_ctr = self.q_proj(h_fused)

        q = q_ctr.view(n_nodes, n_heads, head_dim).unsqueeze(2)
        k = k_nb.view(n_nodes, k_neighbors, n_heads, head_dim).permute(0, 2, 1, 3)
        v = v_nb.view(n_nodes, k_neighbors, n_heads, head_dim).permute(0, 2, 1, 3)

        logits = (q * k).sum(-1) * self.scale
        mask_h = mask.unsqueeze(1).expand(-1, n_heads, -1)
        logits = logits.masked_fill(~mask_h, float("-inf"))
        attn = torch.softmax(logits, dim=2)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        agg = (attn.unsqueeze(-1) * v).sum(dim=2)
        agg = agg.transpose(1, 2).contiguous().view(n_nodes, dim)
        agg = self.out_proj(agg)
        gate = torch.sigmoid(self.gate(torch.cat([h_fused, agg], dim=-1)))
        out = gate * h_fused + (1.0 - gate) * agg
        return self.out_ln(out)


class ProteinSpecificSpatialReadout(nn.Module):
    """Protein-conditioned neighbor readout for target cells.

    The shared spatial token is still available to the fusion module. This block
    adds a later protein-specific residual so each marker can query the local
    neighborhood with a different query vector.
    """

    def __init__(
        self,
        n_proteins: int,
        token_names: Sequence[str],
        dim: int = 256,
        n_heads: int = 4,
        max_neighbors: int = 16,
        pos_dim: int = 64,
        dropout: float = 0.1,
        n_freq: int = 8,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_proteins = n_proteins
        self.token_names = list(token_names)
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_neighbors = max_neighbors
        self.source_indices = [
            i for i, name in enumerate(self.token_names) if name != "spatial"
        ]
        if not self.source_indices:
            self.source_indices = list(range(len(self.token_names)))

        self.protein_embed = nn.Embedding(n_proteins, dim)
        self.pos_enc = FourierSpatialEncoding(n_freq=n_freq, out_dim=pos_dim)
        self.k_pos_proj = nn.Linear(pos_dim, dim)
        self.center_ln = nn.LayerNorm(dim)
        self.protein_ln = nn.LayerNorm(dim)
        self.q_cell = nn.Linear(dim, dim)
        self.q_protein = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.context_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.out_ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    @staticmethod
    def _pad_target_neighbor_tensors(
        edge_index: torch.Tensor,
        coords: torch.Tensor,
        h_feat: torch.Tensor,
        n_targets: int,
        max_neighbors: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]
        device = h_feat.device
        dtype = h_feat.dtype
        dim = h_feat.shape[1]
        k_neighbors = max_neighbors

        nb_feat = torch.zeros(n_targets, k_neighbors, dim, device=device, dtype=dtype)
        delta = torch.zeros(
            n_targets, k_neighbors, 2, device=device, dtype=coords.dtype
        )
        mask = torch.zeros(n_targets, k_neighbors, dtype=torch.bool, device=device)
        if edge_index.numel() == 0 or n_targets == 0:
            return nb_feat, delta, mask

        target_edge = dst < n_targets
        if not target_edge.any():
            return nb_feat, delta, mask

        src_t = src[target_edge]
        dst_t = dst[target_edge]
        sorted_dst, perm = torch.sort(dst_t)
        sorted_src = src_t[perm]
        n_edges = sorted_dst.numel()
        counts = torch.bincount(sorted_dst, minlength=n_targets)
        ptr = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)]
        )
        start_edge = ptr[sorted_dst]
        local_idx = torch.arange(n_edges, device=device, dtype=torch.long) - start_edge
        valid = local_idx < k_neighbors
        if not valid.any():
            return nb_feat, delta, mask

        d_idx = sorted_dst[valid]
        l_idx = local_idx[valid]
        s_idx = sorted_src[valid]
        nb_feat[d_idx, l_idx] = h_feat[s_idx]
        delta[d_idx, l_idx] = coords[s_idx] - coords[d_idx]
        mask[d_idx, l_idx] = True
        return nb_feat, delta, mask

    def _cell_summary(
        self,
        tokens: torch.Tensor,
        token_valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        subset = tokens[:, self.source_indices]
        if token_valid is None:
            return subset.mean(dim=1)

        valid = token_valid[:, self.source_indices].to(tokens.dtype)
        denom = valid.sum(dim=1, keepdim=True)
        summary = (subset * valid.unsqueeze(-1)).sum(dim=1) / denom.clamp_min(1.0)
        return torch.where(denom > 0, summary, torch.zeros_like(summary))

    def forward(
        self,
        tokens: torch.Tensor,
        token_valid: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        coords_um: torch.Tensor,
        fused: torch.Tensor,
        output_size: int,
    ) -> torch.Tensor:
        n_targets = min(output_size, tokens.shape[0], fused.shape[0])
        if n_targets == 0:
            return fused

        source = self._cell_summary(tokens, token_valid)
        nb_feat, delta, mask = self._pad_target_neighbor_tensors(
            edge_index=edge_index,
            coords=coords_um,
            h_feat=source,
            n_targets=n_targets,
            max_neighbors=self.max_neighbors,
        )

        n_heads, head_dim = self.n_heads, self.head_dim
        center = self.center_ln(source[:n_targets])
        protein = self.protein_ln(self.protein_embed.weight)
        q_cell = self.q_cell(center).view(n_targets, 1, n_heads, head_dim)
        q_prot = self.q_protein(protein).view(1, self.n_proteins, n_heads, head_dim)
        q = q_cell + q_prot

        pos_k = self.k_pos_proj(self.pos_enc(delta))
        k_nb = self.k_proj(nb_feat) + pos_k
        v_nb = self.v_proj(nb_feat)
        k = k_nb.view(n_targets, self.max_neighbors, n_heads, head_dim).permute(
            0, 2, 1, 3
        )
        v = v_nb.view(n_targets, self.max_neighbors, n_heads, head_dim).permute(
            0, 2, 1, 3
        )

        logits = torch.einsum("tphd,thkd->tphk", q, k) * self.scale
        logits = logits.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        context = torch.einsum("tphk,thkd->tphd", attn, v)
        context = context.reshape(n_targets, self.n_proteins, self.dim)
        context = self.context_proj(context)

        protein_exp = self.protein_embed.weight.unsqueeze(0).expand(n_targets, -1, -1)
        fused_target = fused[:n_targets]
        gate = self.gate(torch.cat([fused_target, context, protein_exp], dim=-1))
        updated = self.out_ln(fused_target + gate * context)
        if n_targets == fused.shape[0]:
            return updated
        return torch.cat([updated, fused[n_targets:]], dim=0)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FeatureAdapter(nn.Module):
    """Project one precomputed modality feature into the shared token space."""

    def __init__(self, input_dim: int, dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(dim, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _masked_softmax(logits: torch.Tensor, token_valid: Optional[torch.Tensor]) -> torch.Tensor:
    """Softmax over modality tokens with per-sample missing-token masking."""
    if token_valid is None:
        return torch.softmax(logits, dim=-1)
    if logits.dim() == 3:
        mask = token_valid.unsqueeze(1)
    else:
        mask = token_valid
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    attn = torch.softmax(logits, dim=-1)
    return torch.nan_to_num(attn, nan=0.0)


class ProteinConditionedFusion(nn.Module):
    """Protein-query attention over modality/scale tokens."""

    def __init__(
        self,
        n_proteins: int,
        dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_proteins = n_proteins
        self.dim = dim
        self.protein_embed = nn.Embedding(n_proteins, dim)
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            ResidualMLPBlock(dim, dropout=dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        token_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return protein-specific fused features and scale attention.

        Args:
            tokens: [B, S, D]

        Returns:
            fused: [B, P, D]
            attn:  [B, P, S]
        """
        queries = self.query_proj(self.protein_embed.weight)  # [P, D]
        keys = self.key_proj(tokens)                          # [B, S, D]
        values = self.value_proj(tokens)                      # [B, S, D]
        logits = torch.einsum("pd,bsd->bps", queries, keys) / math.sqrt(self.dim)
        attn = _masked_softmax(logits, token_valid)
        attn = self.dropout(attn)
        fused = torch.einsum("bps,bsd->bpd", attn, values)
        return self.out(fused), attn


class SharedAttentionFusion(nn.Module):
    """Shared modality attention plus protein identity embeddings."""

    def __init__(
        self,
        n_proteins: int,
        dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_proteins = n_proteins
        self.dim = dim
        self.shared_query = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.shared_query, std=0.02)
        self.protein_embed = nn.Embedding(n_proteins, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            ResidualMLPBlock(dim, dropout=dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        token_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        keys = self.key_proj(tokens)
        values = self.value_proj(tokens)
        logits = torch.einsum("d,bsd->bs", self.shared_query, keys) / math.sqrt(self.dim)
        attn_shared = _masked_softmax(logits, token_valid)
        attn_shared = self.dropout(attn_shared)
        fused_shared = torch.einsum("bs,bsd->bd", attn_shared, values)
        fused = fused_shared.unsqueeze(1) + self.protein_embed.weight.unsqueeze(0)
        attn = attn_shared.unsqueeze(1).expand(-1, self.n_proteins, -1)
        return self.out(fused), attn


class ConcatProteinFusion(nn.Module):
    """Concatenate modality tokens and condition the output on protein identity."""

    def __init__(
        self,
        n_proteins: int,
        n_tokens: int,
        dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_proteins = n_proteins
        self.n_tokens = n_tokens
        self.protein_embed = nn.Embedding(n_proteins, dim)
        self.net = nn.Sequential(
            nn.LayerNorm(n_tokens * dim + dim),
            nn.Linear(n_tokens * dim + dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            ResidualMLPBlock(dim, dropout=dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        token_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = tokens.shape[0]
        flat = tokens.reshape(bsz, -1)
        protein = self.protein_embed.weight.unsqueeze(0).expand(bsz, -1, -1)
        flat_rep = flat.unsqueeze(1).expand(-1, self.n_proteins, -1)
        fused = self.net(torch.cat([flat_rep, protein], dim=-1))

        if token_valid is None:
            attn_base = tokens.new_full((bsz, self.n_tokens), 1.0 / self.n_tokens)
        else:
            denom = token_valid.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
            attn_base = token_valid.float() / denom
        attn = attn_base.unsqueeze(1).expand(-1, self.n_proteins, -1)
        return fused, attn


class ProteinConditionedInteractionBlock(nn.Module):
    """Low-rank protein-gated interaction block for PCIF variants."""

    def __init__(
        self,
        n_inputs: int,
        n_proteins: int,
        dim: int = 256,
        rank: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_proteins = n_proteins
        self.dim = dim
        self.rank = rank
        self.input_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_inputs)])
        self.input_projs = nn.ModuleList([nn.Linear(dim, rank) for _ in range(n_inputs)])
        self.protein_norm = nn.LayerNorm(dim)
        self.gates = nn.ModuleList([nn.Linear(dim, rank) for _ in range(n_inputs)])
        n_pairs = n_inputs * (n_inputs - 1) // 2
        in_dim = rank * (n_inputs + n_pairs) + dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            ResidualMLPBlock(dim, dropout=dropout),
        )

    @staticmethod
    def _ensure_valid(
        input_valid: Optional[torch.Tensor],
        batch_size: int,
        n_inputs: int,
        device: torch.device,
    ) -> torch.Tensor:
        if input_valid is None:
            return torch.ones(batch_size, n_inputs, dtype=torch.bool, device=device)
        return input_valid.bool()

    def _project_input(
        self,
        x: torch.Tensor,
        input_idx: int,
        protein: torch.Tensor,
    ) -> torch.Tensor:
        gate = 2.0 * torch.sigmoid(self.gates[input_idx](self.protein_norm(protein)))
        if x.dim() == 2:
            h = self.input_projs[input_idx](self.input_norms[input_idx](x))
            return h.unsqueeze(1) * gate.unsqueeze(0)
        if x.dim() == 3:
            bsz, n_proteins, dim = x.shape
            h = self.input_projs[input_idx](
                self.input_norms[input_idx](x.reshape(bsz * n_proteins, dim))
            )
            return h.view(bsz, n_proteins, self.rank) * gate.unsqueeze(0)
        raise ValueError("PCIF inputs must be [B, D] or [B, P, D]")

    @staticmethod
    def _contribution_proxy(
        z: torch.Tensor,
        input_valid: torch.Tensor,
    ) -> torch.Tensor:
        contribution = z.abs().mean(dim=-1)
        n_inputs = z.shape[2]
        for i in range(n_inputs):
            for j in range(i + 1, n_inputs):
                pair = (z[:, :, i] * z[:, :, j]).abs().mean(dim=-1)
                contribution[:, :, i] = contribution[:, :, i] + 0.5 * pair
                contribution[:, :, j] = contribution[:, :, j] + 0.5 * pair

        valid = input_valid.unsqueeze(1)
        contribution = contribution.masked_fill(~valid, 0.0)
        denom = contribution.sum(dim=-1, keepdim=True)
        valid_float = input_valid.float()
        uniform = valid_float / valid_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
        uniform = uniform.unsqueeze(1).expand_as(contribution)
        attn = contribution / denom.clamp_min(torch.finfo(contribution.dtype).eps)
        return torch.where(denom > 0, attn, uniform)

    def forward(
        self,
        inputs: Sequence[torch.Tensor],
        protein: torch.Tensor,
        input_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(inputs) != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} PCIF inputs, got {len(inputs)}")
        bsz = inputs[0].shape[0]
        valid = self._ensure_valid(input_valid, bsz, self.n_inputs, inputs[0].device)

        projected = []
        for i, x in enumerate(inputs):
            z_i = self._project_input(x, i, protein)
            z_i = z_i * valid[:, i].view(bsz, 1, 1).to(z_i.dtype)
            projected.append(z_i)
        z = torch.stack(projected, dim=2)

        terms = [z[:, :, i] for i in range(self.n_inputs)]
        for i in range(self.n_inputs):
            for j in range(i + 1, self.n_inputs):
                terms.append(z[:, :, i] * z[:, :, j])
        protein_rep = protein.unsqueeze(0).expand(bsz, -1, -1)
        fused = self.net(torch.cat(terms + [protein_rep], dim=-1))
        return fused, self._contribution_proxy(z, valid)


class ProteinConditionedInteractionFusion(nn.Module):
    """PCIF-pairwise: explicit low-rank pairwise token interactions."""

    def __init__(
        self,
        n_proteins: int,
        n_tokens: int,
        dim: int = 256,
        rank: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_proteins = n_proteins
        self.n_tokens = n_tokens
        self.dim = dim
        self.rank = rank
        self.protein_embed = nn.Embedding(n_proteins, dim)
        self.block = ProteinConditionedInteractionBlock(
            n_inputs=n_tokens,
            n_proteins=n_proteins,
            dim=dim,
            rank=rank,
            dropout=dropout,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        token_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = [tokens[:, i] for i in range(self.n_tokens)]
        return self.block(inputs, self.protein_embed.weight, input_valid=token_valid)


class HierarchicalProteinConditionedInteractionFusion(nn.Module):
    """PCIF-hier: H&E -> intrinsic RNA/H&E -> optional spatial context."""

    def __init__(
        self,
        n_proteins: int,
        token_names: Sequence[str],
        dim: int = 256,
        rank: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_proteins = n_proteins
        self.token_names = list(token_names)
        self.n_tokens = len(self.token_names)
        self.dim = dim
        self.rank = rank
        self.protein_embed = nn.Embedding(n_proteins, dim)

        self.rna_idx = self.token_names.index("rna")
        self.he_indices = [
            i for i, name in enumerate(self.token_names) if name in ("he_cell", "he_context")
        ]
        self.spatial_idx = (
            self.token_names.index("spatial") if "spatial" in self.token_names else None
        )

        self.he_block = None
        if self.he_indices:
            self.he_block = ProteinConditionedInteractionBlock(
                n_inputs=len(self.he_indices),
                n_proteins=n_proteins,
                dim=dim,
                rank=rank,
                dropout=dropout,
            )
        self.intrinsic_block = ProteinConditionedInteractionBlock(
            n_inputs=2 if self.he_indices else 1,
            n_proteins=n_proteins,
            dim=dim,
            rank=rank,
            dropout=dropout,
        )
        self.contextual_block = None
        if self.spatial_idx is not None:
            self.contextual_block = ProteinConditionedInteractionBlock(
                n_inputs=2,
                n_proteins=n_proteins,
                dim=dim,
                rank=rank,
                dropout=dropout,
            )

    def _valid_or_ones(
        self,
        tokens: torch.Tensor,
        token_valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if token_valid is None:
            return torch.ones(
                tokens.shape[0], self.n_tokens, dtype=torch.bool, device=tokens.device
            )
        return token_valid.bool()

    def forward(
        self,
        tokens: torch.Tensor,
        token_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = self._valid_or_ones(tokens, token_valid)
        protein = self.protein_embed.weight
        bsz = tokens.shape[0]
        token_dist = tokens.new_zeros(bsz, self.n_proteins, self.n_tokens)

        he_summary = None
        he_any_valid = None
        he_local = None
        if self.he_block is not None:
            he_inputs = [tokens[:, i] for i in self.he_indices]
            he_valid = valid[:, self.he_indices]
            he_summary, he_local = self.he_block(he_inputs, protein, input_valid=he_valid)
            he_any_valid = he_valid.any(dim=1)
            he_summary = he_summary * he_any_valid.view(bsz, 1, 1).to(he_summary.dtype)

        intrinsic_inputs = [tokens[:, self.rna_idx]]
        if he_summary is not None:
            intrinsic_inputs.append(he_summary)
            intrinsic_valid = torch.stack([valid[:, self.rna_idx], he_any_valid], dim=1)
        else:
            intrinsic_valid = valid[:, self.rna_idx].view(bsz, 1)
        intrinsic, intrinsic_local = self.intrinsic_block(
            intrinsic_inputs, protein, input_valid=intrinsic_valid
        )

        prev_dist = token_dist.clone()
        prev_dist[:, :, self.rna_idx] = intrinsic_local[:, :, 0]
        if he_summary is not None and he_local is not None:
            he_weight = intrinsic_local[:, :, 1].unsqueeze(-1) * he_local
            for local_i, token_i in enumerate(self.he_indices):
                prev_dist[:, :, token_i] = prev_dist[:, :, token_i] + he_weight[:, :, local_i]

        intrinsic_any_valid = intrinsic_valid.any(dim=1)
        if self.contextual_block is None:
            attn = prev_dist
            denom = attn.sum(dim=-1, keepdim=True)
            attn = attn / denom.clamp_min(torch.finfo(attn.dtype).eps)
            return intrinsic, torch.where(denom > 0, attn, token_dist)

        intrinsic_for_context = intrinsic * intrinsic_any_valid.view(bsz, 1, 1).to(
            intrinsic.dtype
        )
        spatial_valid = valid[:, self.spatial_idx]
        contextual_inputs = [intrinsic_for_context, tokens[:, self.spatial_idx]]
        contextual_valid = torch.stack([intrinsic_any_valid, spatial_valid], dim=1)
        fused, contextual_local = self.contextual_block(
            contextual_inputs, protein, input_valid=contextual_valid
        )

        attn = contextual_local[:, :, 0].unsqueeze(-1) * prev_dist
        attn[:, :, self.spatial_idx] = attn[:, :, self.spatial_idx] + contextual_local[:, :, 1]
        denom = attn.sum(dim=-1, keepdim=True)
        attn = attn / denom.clamp_min(torch.finfo(attn.dtype).eps)
        return fused, torch.where(denom > 0, attn, token_dist)


class HybridProteinConditionedInteractionFusion(nn.Module):
    """Concat backbone plus a gated PCIF residual interaction branch."""

    def __init__(
        self,
        n_proteins: int,
        token_names: Sequence[str],
        dim: int = 256,
        rank: int = 128,
        dropout: float = 0.1,
        interaction_mode: str = "pairwise",
    ):
        super().__init__()
        if interaction_mode not in ("pairwise", "hier"):
            raise ValueError("interaction_mode must be one of: pairwise, hier")
        self.n_proteins = n_proteins
        self.token_names = list(token_names)
        self.n_tokens = len(self.token_names)
        self.dim = dim
        self.rank = rank
        self.interaction_mode = interaction_mode

        self.protein_embed = nn.Embedding(n_proteins, dim)
        self.concat_net = nn.Sequential(
            nn.LayerNorm(self.n_tokens * dim + dim),
            nn.Linear(self.n_tokens * dim + dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            ResidualMLPBlock(dim, dropout=dropout),
        )
        self.pairwise_block = ProteinConditionedInteractionBlock(
            n_inputs=self.n_tokens,
            n_proteins=n_proteins,
            dim=dim,
            rank=rank,
            dropout=dropout,
        )

        self.rna_idx = self.token_names.index("rna")
        self.he_indices = [
            i for i, name in enumerate(self.token_names) if name in ("he_cell", "he_context")
        ]
        self.spatial_idx = (
            self.token_names.index("spatial") if "spatial" in self.token_names else None
        )
        self.he_block = None
        self.intrinsic_block = None
        self.contextual_block = None
        if interaction_mode == "hier":
            if self.he_indices:
                self.he_block = ProteinConditionedInteractionBlock(
                    n_inputs=len(self.he_indices),
                    n_proteins=n_proteins,
                    dim=dim,
                    rank=rank,
                    dropout=dropout,
                )
            self.intrinsic_block = ProteinConditionedInteractionBlock(
                n_inputs=2 if self.he_indices else 1,
                n_proteins=n_proteins,
                dim=dim,
                rank=rank,
                dropout=dropout,
            )
            if self.spatial_idx is not None:
                self.contextual_block = ProteinConditionedInteractionBlock(
                    n_inputs=2,
                    n_proteins=n_proteins,
                    dim=dim,
                    rank=rank,
                    dropout=dropout,
                )

        self.delta = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate[-2].bias, -2.0)
        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            ResidualMLPBlock(dim, dropout=dropout),
        )

    def _valid_or_ones(
        self,
        tokens: torch.Tensor,
        token_valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if token_valid is None:
            return torch.ones(
                tokens.shape[0], self.n_tokens, dtype=torch.bool, device=tokens.device
            )
        return token_valid.bool()

    def _concat_forward(
        self,
        tokens: torch.Tensor,
        protein: torch.Tensor,
        valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = tokens.shape[0]
        flat = tokens.reshape(bsz, -1)
        protein_rep = protein.unsqueeze(0).expand(bsz, -1, -1)
        flat_rep = flat.unsqueeze(1).expand(-1, self.n_proteins, -1)
        fused = self.concat_net(torch.cat([flat_rep, protein_rep], dim=-1))

        denom = valid.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        attn_base = valid.float() / denom
        attn = attn_base.unsqueeze(1).expand(-1, self.n_proteins, -1)
        return fused, attn

    def _pairwise_forward(
        self,
        tokens: torch.Tensor,
        protein: torch.Tensor,
        valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = [tokens[:, i] for i in range(self.n_tokens)]
        return self.pairwise_block(inputs, protein, input_valid=valid)

    def _hier_forward(
        self,
        tokens: torch.Tensor,
        protein: torch.Tensor,
        valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = tokens.shape[0]
        token_dist = tokens.new_zeros(bsz, self.n_proteins, self.n_tokens)

        he_summary = None
        he_any_valid = None
        he_local = None
        if self.he_block is not None:
            he_inputs = [tokens[:, i] for i in self.he_indices]
            he_valid = valid[:, self.he_indices]
            he_summary, he_local = self.he_block(he_inputs, protein, input_valid=he_valid)
            he_any_valid = he_valid.any(dim=1)
            he_summary = he_summary * he_any_valid.view(bsz, 1, 1).to(he_summary.dtype)

        intrinsic_inputs = [tokens[:, self.rna_idx]]
        if he_summary is not None:
            intrinsic_inputs.append(he_summary)
            intrinsic_valid = torch.stack([valid[:, self.rna_idx], he_any_valid], dim=1)
        else:
            intrinsic_valid = valid[:, self.rna_idx].view(bsz, 1)
        intrinsic, intrinsic_local = self.intrinsic_block(
            intrinsic_inputs, protein, input_valid=intrinsic_valid
        )

        prev_dist = token_dist.clone()
        prev_dist[:, :, self.rna_idx] = intrinsic_local[:, :, 0]
        if he_summary is not None and he_local is not None:
            he_weight = intrinsic_local[:, :, 1].unsqueeze(-1) * he_local
            for local_i, token_i in enumerate(self.he_indices):
                prev_dist[:, :, token_i] = prev_dist[:, :, token_i] + he_weight[:, :, local_i]

        intrinsic_any_valid = intrinsic_valid.any(dim=1)
        if self.contextual_block is None:
            attn = prev_dist
            denom = attn.sum(dim=-1, keepdim=True)
            attn = attn / denom.clamp_min(torch.finfo(attn.dtype).eps)
            return intrinsic, torch.where(denom > 0, attn, token_dist)

        intrinsic_for_context = intrinsic * intrinsic_any_valid.view(bsz, 1, 1).to(
            intrinsic.dtype
        )
        spatial_valid = valid[:, self.spatial_idx]
        contextual_inputs = [intrinsic_for_context, tokens[:, self.spatial_idx]]
        contextual_valid = torch.stack([intrinsic_any_valid, spatial_valid], dim=1)
        fused, contextual_local = self.contextual_block(
            contextual_inputs, protein, input_valid=contextual_valid
        )

        attn = contextual_local[:, :, 0].unsqueeze(-1) * prev_dist
        attn[:, :, self.spatial_idx] = attn[:, :, self.spatial_idx] + contextual_local[:, :, 1]
        denom = attn.sum(dim=-1, keepdim=True)
        attn = attn / denom.clamp_min(torch.finfo(attn.dtype).eps)
        return fused, torch.where(denom > 0, attn, token_dist)

    def forward(
        self,
        tokens: torch.Tensor,
        token_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = self._valid_or_ones(tokens, token_valid)
        protein = self.protein_embed.weight
        concat_fused, concat_attn = self._concat_forward(tokens, protein, valid)
        if self.interaction_mode == "hier":
            interaction_fused, interaction_attn = self._hier_forward(tokens, protein, valid)
        else:
            interaction_fused, interaction_attn = self._pairwise_forward(
                tokens, protein, valid
            )

        bsz = tokens.shape[0]
        protein_rep = protein.unsqueeze(0).expand(bsz, -1, -1)
        gate = self.gate(torch.cat([concat_fused, interaction_fused, protein_rep], dim=-1))
        fused = self.out(concat_fused + gate * self.delta(interaction_fused))

        gate_strength = gate.mean(dim=-1, keepdim=True)
        attn = (1.0 - gate_strength) * concat_attn + gate_strength * interaction_attn
        denom = attn.sum(dim=-1, keepdim=True)
        attn = attn / denom.clamp_min(torch.finfo(attn.dtype).eps)
        return fused, attn


class TokenSubsetProteinExpert(nn.Module):
    """Protein-conditioned expert over a selected token group."""

    def __init__(
        self,
        n_proteins: int,
        n_tokens: int,
        token_indices: Sequence[int],
        dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_proteins = n_proteins
        self.n_tokens = n_tokens
        self.token_indices = list(token_indices)
        self.dim = dim
        self.fusion = ProteinConditionedFusion(n_proteins, dim=dim, dropout=dropout)
        self.out = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            ResidualMLPBlock(dim, dropout=dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        token_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        subset = tokens[:, self.token_indices]
        subset_valid = None
        if token_valid is not None:
            subset_valid = token_valid[:, self.token_indices].bool()

        fused, local_attn = self.fusion(subset, token_valid=subset_valid)
        bsz = tokens.shape[0]
        protein = self.fusion.protein_embed.weight.unsqueeze(0).expand(bsz, -1, -1)
        fused = self.out(torch.cat([fused, protein], dim=-1))

        if subset_valid is None:
            subset_valid = torch.ones(
                bsz,
                len(self.token_indices),
                dtype=torch.bool,
                device=tokens.device,
            )
        any_valid = subset_valid.any(dim=1)
        fused = fused * any_valid.view(bsz, 1, 1).to(fused.dtype)

        local_attn = local_attn * subset_valid.unsqueeze(1).to(local_attn.dtype)
        denom = local_attn.sum(dim=-1, keepdim=True)
        local_attn = local_attn / denom.clamp_min(torch.finfo(local_attn.dtype).eps)
        local_attn = torch.where(denom > 0, local_attn, torch.zeros_like(local_attn))

        attn = tokens.new_zeros(bsz, self.n_proteins, self.n_tokens)
        for local_i, token_i in enumerate(self.token_indices):
            attn[:, :, token_i] = local_attn[:, :, local_i]
        return fused, attn


class ProteinConditionedMoEFusion(nn.Module):
    """Concat-anchored protein-conditioned mixture of fusion experts."""

    def __init__(
        self,
        n_proteins: int,
        token_names: Sequence[str],
        dim: int = 256,
        rank: int = 128,
        dropout: float = 0.1,
        include_pairwise: bool = True,
        router_context: bool = True,
    ):
        super().__init__()
        self.n_proteins = n_proteins
        self.token_names = list(token_names)
        self.n_tokens = len(self.token_names)
        self.dim = dim
        self.rank = rank
        self.include_pairwise = include_pairwise
        self.router_context = router_context

        self.concat = ConcatProteinFusion(
            n_proteins, n_tokens=self.n_tokens, dim=dim, dropout=dropout
        )
        self.experts = nn.ModuleList()
        self.delta_nets = nn.ModuleList()
        self.expert_names: List[str] = []

        if include_pairwise:
            self._add_expert(
                "pairwise",
                ProteinConditionedInteractionFusion(
                    n_proteins,
                    n_tokens=self.n_tokens,
                    dim=dim,
                    rank=rank,
                    dropout=dropout,
                ),
                dropout,
            )
        self._add_expert(
            "hier",
            HierarchicalProteinConditionedInteractionFusion(
                n_proteins,
                token_names=self.token_names,
                dim=dim,
                rank=rank,
                dropout=dropout,
            ),
            dropout,
        )

        self._add_expert(
            "rna",
            TokenSubsetProteinExpert(
                n_proteins,
                self.n_tokens,
                [self.token_names.index("rna")],
                dim=dim,
                dropout=dropout,
            ),
            dropout,
        )

        he_indices = [
            i for i, name in enumerate(self.token_names) if name in ("he_cell", "he_context")
        ]
        if he_indices:
            self._add_expert(
                "he",
                TokenSubsetProteinExpert(
                    n_proteins,
                    self.n_tokens,
                    he_indices,
                    dim=dim,
                    dropout=dropout,
                ),
                dropout,
            )

        if "spatial" in self.token_names:
            self._add_expert(
                "spatial",
                TokenSubsetProteinExpert(
                    n_proteins,
                    self.n_tokens,
                    [self.token_names.index("spatial")],
                    dim=dim,
                    dropout=dropout,
                ),
                dropout,
            )

        self.router_protein = nn.Embedding(n_proteins, dim)
        router_in_dim = dim * 2 if router_context else dim
        self.router = nn.Sequential(
            nn.LayerNorm(router_in_dim),
            nn.Linear(router_in_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, len(self.experts) + 1),
        )
        with torch.no_grad():
            self.router[-1].bias.fill_(-1.0)
            self.router[-1].bias[0] = 2.0

        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            ResidualMLPBlock(dim, dropout=dropout),
        )

    def _add_expert(self, name: str, expert: nn.Module, dropout: float) -> None:
        self.expert_names.append(name)
        self.experts.append(expert)
        self.delta_nets.append(
            nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, self.dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.dim, self.dim),
            )
        )

    def _valid_or_ones(
        self,
        tokens: torch.Tensor,
        token_valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if token_valid is None:
            return torch.ones(
                tokens.shape[0], self.n_tokens, dtype=torch.bool, device=tokens.device
            )
        return token_valid.bool()

    def _cell_summary(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.to(tokens.dtype)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (tokens * weights.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        tokens: torch.Tensor,
        token_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = self._valid_or_ones(tokens, token_valid)
        bsz = tokens.shape[0]

        concat_fused, concat_attn = self.concat(tokens, token_valid=valid)
        protein = self.router_protein.weight.unsqueeze(0).expand(bsz, -1, -1)
        if self.router_context:
            cell_summary = self._cell_summary(tokens, valid)
            summary = cell_summary.unsqueeze(1).expand(-1, self.n_proteins, -1)
            route_input = torch.cat([summary, protein], dim=-1)
        else:
            route_input = protein
        route_logits = self.router(route_input)
        route_weight = torch.softmax(route_logits, dim=-1)

        fused = concat_fused
        attn = route_weight[:, :, :1] * concat_attn
        for expert_i, expert in enumerate(self.experts):
            expert_fused, expert_attn = expert(tokens, token_valid=valid)
            w = route_weight[:, :, expert_i + 1 : expert_i + 2]
            fused = fused + w * self.delta_nets[expert_i](expert_fused)
            attn = attn + w * expert_attn

        denom = attn.sum(dim=-1, keepdim=True)
        attn = attn / denom.clamp_min(torch.finfo(attn.dtype).eps)
        return self.out(fused), torch.where(denom > 0, attn, concat_attn)


class HurdleProteinHead(nn.Module):
    """Predict protein foreground probability and standardized abundance."""

    def __init__(
        self,
        dim: int = 256,
        dropout: float = 0.1,
        direct: bool = False,
    ):
        super().__init__()
        self.direct = direct
        self.shared = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.abundance = nn.Linear(dim, 1)
        self.foreground = None if direct else nn.Linear(dim, 1)

    def forward(
        self,
        fused: torch.Tensor,
        zero_std: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        h = self.shared(fused)
        abundance = self.abundance(h).squeeze(-1)
        if self.direct:
            return {
                "pred_std": abundance,
                "abundance_std": abundance,
                "fg_logits": None,
                "fg_prob": None,
            }

        fg_logits = self.foreground(h).squeeze(-1)
        fg_prob = torch.sigmoid(fg_logits)
        pred_std = fg_prob * abundance + (1.0 - fg_prob) * zero_std.view(1, -1)
        return {
            "pred_std": pred_std,
            "abundance_std": abundance,
            "fg_logits": fg_logits,
            "fg_prob": fg_prob,
        }


class SpatialChordReg(nn.Module):
    """Cell-centric regression model with protein-conditioned fusion."""

    def __init__(
        self,
        feature_dims: Dict[str, int],
        n_proteins: int,
        dim: int = 256,
        dropout: float = 0.1,
        use_unc_gate: bool = True,
        direct_head: bool = False,
        use_spatial: bool = False,
        spatial_k: int = 16,
        fusion_type: str = "protein_conditioned",
        pcif_rank: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.n_proteins = n_proteins
        self.use_unc_gate = use_unc_gate and "rna_var" in feature_dims
        self.use_spatial = use_spatial
        self.fusion_type = fusion_type
        self.base_fusion_type = (
            fusion_type[: -len("_psr")] if fusion_type.endswith("_psr") else fusion_type
        )
        self.use_protein_spatial_readout = use_spatial and fusion_type.endswith("_psr")
        self.pcif_rank_arg = pcif_rank
        self.pcif_rank = pcif_rank if pcif_rank > 0 else max(1, dim // 2)

        self.rna_adapter = FeatureAdapter(feature_dims["rna_latent"], dim, dropout)
        self.rna_gate = (
            UncertaintyGate(var_dim=feature_dims["rna_var"], feat_dim=dim)
            if self.use_unc_gate
            else None
        )

        self.he_cell_adapter = None
        if "he_cell" in feature_dims:
            self.he_cell_adapter = FeatureAdapter(feature_dims["he_cell"], dim, dropout)

        self.context_key = next(
            (k for k in ("he_context", "he_context_224", "he_micro") if k in feature_dims),
            None,
        )
        self.he_context_adapter = None
        if self.context_key is not None:
            self.he_context_adapter = FeatureAdapter(
                feature_dims[self.context_key], dim, dropout
            )

        self.spatial_attn = None
        if use_spatial:
            self.spatial_attn = SpatialNeighborAttention(
                dim=dim,
                n_heads=8,
                max_neighbors=spatial_k,
                pos_dim=64,
                dropout=dropout,
            )

        self._token_names = self._build_token_names()
        self.missing_embeddings = nn.Parameter(torch.zeros(len(self._token_names), dim))
        nn.init.normal_(self.missing_embeddings, std=0.02)

        self.protein_spatial_readout = None
        if self.use_protein_spatial_readout:
            self.protein_spatial_readout = ProteinSpecificSpatialReadout(
                n_proteins=n_proteins,
                token_names=self._token_names,
                dim=dim,
                n_heads=4,
                max_neighbors=spatial_k,
                pos_dim=64,
                dropout=dropout,
            )

        if self.base_fusion_type == "protein_conditioned":
            self.fusion = ProteinConditionedFusion(n_proteins, dim=dim, dropout=dropout)
        elif self.base_fusion_type == "shared_attention":
            self.fusion = SharedAttentionFusion(n_proteins, dim=dim, dropout=dropout)
        elif self.base_fusion_type == "concat":
            self.fusion = ConcatProteinFusion(
                n_proteins, n_tokens=len(self._token_names), dim=dim, dropout=dropout
            )
        elif self.base_fusion_type == "pcif_pairwise":
            self.fusion = ProteinConditionedInteractionFusion(
                n_proteins,
                n_tokens=len(self._token_names),
                dim=dim,
                rank=self.pcif_rank,
                dropout=dropout,
            )
        elif self.base_fusion_type == "pcif_hier":
            self.fusion = HierarchicalProteinConditionedInteractionFusion(
                n_proteins,
                token_names=self._token_names,
                dim=dim,
                rank=self.pcif_rank,
                dropout=dropout,
            )
        elif self.base_fusion_type == "hybrid_pcif":
            self.fusion = HybridProteinConditionedInteractionFusion(
                n_proteins,
                token_names=self._token_names,
                dim=dim,
                rank=self.pcif_rank,
                dropout=dropout,
                interaction_mode="pairwise",
            )
        elif self.base_fusion_type == "hybrid_pcif_hier":
            self.fusion = HybridProteinConditionedInteractionFusion(
                n_proteins,
                token_names=self._token_names,
                dim=dim,
                rank=self.pcif_rank,
                dropout=dropout,
                interaction_mode="hier",
            )
        elif self.base_fusion_type == "pc_moe":
            self.fusion = ProteinConditionedMoEFusion(
                n_proteins,
                token_names=self._token_names,
                dim=dim,
                rank=self.pcif_rank,
                dropout=dropout,
                include_pairwise=True,
            )
        elif self.base_fusion_type == "pc_moe_hier":
            self.fusion = ProteinConditionedMoEFusion(
                n_proteins,
                token_names=self._token_names,
                dim=dim,
                rank=self.pcif_rank,
                dropout=dropout,
                include_pairwise=False,
            )
        elif self.base_fusion_type == "pc_moe_static":
            self.fusion = ProteinConditionedMoEFusion(
                n_proteins,
                token_names=self._token_names,
                dim=dim,
                rank=self.pcif_rank,
                dropout=dropout,
                include_pairwise=True,
                router_context=False,
            )
        elif self.base_fusion_type == "pc_moe_hier_static":
            self.fusion = ProteinConditionedMoEFusion(
                n_proteins,
                token_names=self._token_names,
                dim=dim,
                rank=self.pcif_rank,
                dropout=dropout,
                include_pairwise=False,
                router_context=False,
            )
        else:
            raise ValueError(
                "fusion_type must be one of: protein_conditioned, shared_attention, "
                "concat, pcif_pairwise, pcif_hier, hybrid_pcif, hybrid_pcif_hier, "
                "pc_moe, pc_moe_hier, pc_moe_static, pc_moe_hier_static, or a "
                "supported *_psr variant"
            )
        self.head = HurdleProteinHead(dim=dim, dropout=dropout, direct=direct_head)

    def _build_token_names(self) -> List[str]:
        names = ["rna"]
        if self.he_cell_adapter is not None:
            names.append("he_cell")
        if self.he_context_adapter is not None:
            names.append("he_context")
        if self.use_spatial:
            names.append("spatial")
        return names

    def token_names(self) -> List[str]:
        return list(self._token_names)

    def _token_valid(
        self,
        batch_size: int,
        device: torch.device,
        missing_modalities: Optional[Sequence[str]] = None,
        modality_mask_train: str = "none",
        mask_prob: float = 0.0,
    ) -> torch.Tensor:
        names = self.token_names()
        valid = torch.ones(batch_size, len(names), dtype=torch.bool, device=device)
        missing = set(missing_modalities or [])
        for i, name in enumerate(names):
            if name in missing:
                valid[:, i] = False

        if self.training and modality_mask_train == "single_drop" and mask_prob > 0:
            eligible = [i for i, name in enumerate(names) if name not in missing]
            if eligible:
                drop_rows = torch.rand(batch_size, device=device) < mask_prob
                row_idx = drop_rows.nonzero(as_tuple=False).flatten()
                if row_idx.numel() > 0:
                    eligible_t = torch.tensor(eligible, dtype=torch.long, device=device)
                    choice = eligible_t[
                        torch.randint(0, len(eligible), (row_idx.numel(),), device=device)
                    ]
                    valid[row_idx, choice] = False
        return valid

    def _apply_missing_embeddings(
        self,
        tokens: torch.Tensor,
        token_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        missing = ~token_valid
        if missing.any():
            repl = self.missing_embeddings.unsqueeze(0).expand(tokens.shape[0], -1, -1)
            tokens = torch.where(token_valid.unsqueeze(-1), tokens, repl)

        # If an evaluation intentionally masks the only available modality,
        # attend over the learned missing token(s) instead of producing NaNs.
        attn_valid = token_valid.clone()
        all_missing = ~attn_valid.any(dim=1)
        if all_missing.any():
            attn_valid[all_missing] = True
        return tokens, attn_valid

    def encode_tokens(
        self,
        batch: Dict[str, torch.Tensor],
        edge_index: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        token_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens: List[torch.Tensor] = []

        h_rna = self.rna_adapter(batch["rna_latent"])
        if self.rna_gate is not None:
            h_rna = self.rna_gate(h_rna, batch["rna_var"])
        tokens.append(h_rna)

        if self.he_cell_adapter is not None:
            tokens.append(self.he_cell_adapter(batch["he_cell"]))

        if self.he_context_adapter is not None and self.context_key is not None:
            tokens.append(self.he_context_adapter(batch[self.context_key]))

        if self.spatial_attn is not None:
            base_tokens = torch.stack(tokens, dim=1)
            if token_valid is not None:
                base_valid = token_valid[:, : base_tokens.shape[1]]
                base_weights = base_valid.float()
                denom = base_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
                spatial_source = (base_tokens * base_weights.unsqueeze(-1)).sum(dim=1) / denom
                no_base = ~base_valid.any(dim=1)
                if no_base.any():
                    spatial_source[no_base] = base_tokens[no_base].mean(dim=1)
            else:
                spatial_source = base_tokens.mean(dim=1)
            h_spatial = self.spatial_attn(spatial_source, edge_index, pos)
            tokens.append(h_spatial)

        out = torch.stack(tokens, dim=1)
        if token_valid is None:
            token_valid = torch.ones(
                out.shape[0], out.shape[1], dtype=torch.bool, device=out.device
            )
        return self._apply_missing_embeddings(out, token_valid)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        zero_std: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        missing_modalities: Optional[Sequence[str]] = None,
        modality_mask_train: str = "none",
        mask_prob: float = 0.0,
        output_size: Optional[int] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        first = batch["rna_latent"]
        token_valid = self._token_valid(
            batch_size=first.shape[0],
            device=first.device,
            missing_modalities=missing_modalities,
            modality_mask_train=modality_mask_train,
            mask_prob=mask_prob,
        )
        tokens, attn_valid = self.encode_tokens(
            batch, edge_index=edge_index, pos=pos, token_valid=token_valid
        )
        all_tokens = tokens
        all_token_valid = token_valid
        if output_size is not None:
            if output_size < 1 or output_size > tokens.shape[0]:
                raise ValueError(
                    f"output_size must be in [1, {tokens.shape[0]}], got {output_size}"
                )
            tokens = tokens[:output_size]
            token_valid = token_valid[:output_size]
            attn_valid = attn_valid[:output_size]
        fusion_valid = (
            token_valid
            if "pcif" in self.base_fusion_type or "moe" in self.base_fusion_type
            else attn_valid
        )
        fused, attn = self.fusion(tokens, token_valid=fusion_valid)
        if (
            self.protein_spatial_readout is not None
            and edge_index is not None
            and pos is not None
        ):
            fused = self.protein_spatial_readout(
                all_tokens,
                all_token_valid,
                edge_index=edge_index,
                coords_um=pos,
                fused=fused,
                output_size=fused.shape[0],
            )
        out = self.head(fused, zero_std=zero_std)
        out["scale_attention"] = attn
        return out
