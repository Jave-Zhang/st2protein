# -*- coding: utf-8 -*-
"""
Spatial Multi-Omics Dataset for Protein Prediction

Extensible data loading pipeline supporting multiple modalities:
  - rna_latent: scVI posterior mean μ
  - he_cell: UNI2-h cell-centric CLS features
  - he_context: UNI2-h context patch features
  - spatial: Xenium μm spatial coordinates

Design principle: each modality is a named numpy array aligned by cell index.
Adding a new modality only requires providing the .npy file and registering it.
"""

import numpy as np
import scipy.sparse
import scanpy as sc
import torch
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from torch_geometric.data import Data as PyGData
    from torch_geometric.loader import NeighborLoader as PyGNeighborLoader


def _import_pyg():
    """Lazy import so non-graph ablations keep working without PyG."""
    try:
        from torch_geometric.data import Data
        from torch_geometric.loader import NeighborLoader
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "PyG runtime is required for +SpatialAttn. "
            "Please install torch_geometric in the HISTEX environment."
        ) from exc
    return Data, NeighborLoader


def _build_knn_edge_index(coords: np.ndarray, knn_k: int, loop: bool = False) -> torch.Tensor:
    """
    Build a symmetrized kNN graph from spatial coordinates with sklearn.
    Returns edge_index in neighbor->center format expected by SpatialNeighborAttention.
    """
    n_cells = coords.shape[0]
    n_neighbors = knn_k + (0 if loop else 1)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nbrs.fit(coords)
    indices = nbrs.kneighbors(coords, return_distance=False)

    src_list = []
    dst_list = []
    for center_idx in range(n_cells):
        nbr_idx = indices[center_idx]
        if not loop:
            nbr_idx = nbr_idx[nbr_idx != center_idx]
        nbr_idx = nbr_idx[:knn_k]
        dst_list.append(np.full_like(nbr_idx, center_idx))
        src_list.append(nbr_idx)

    src = np.concatenate(src_list, axis=0)
    dst = np.concatenate(dst_list, axis=0)
    edge_index = np.stack([src, dst], axis=0)

    # Symmetrize and remove duplicates while preserving neighbor->center semantics.
    edge_rev = edge_index[[1, 0], :]
    edge_all = np.concatenate([edge_index, edge_rev], axis=1)
    edge_all = np.unique(edge_all, axis=1)
    edge_all = np.ascontiguousarray(edge_all)
    return torch.from_numpy(edge_all).long().contiguous()


def _build_random_edge_index(
    n_cells: int,
    knn_k: int,
    seed: int,
    loop: bool = False,
) -> torch.Tensor:
    """Build a symmetrized random graph with approximately k neighbors per node."""
    rng = np.random.RandomState(seed)
    src = rng.randint(0, n_cells, size=(n_cells, knn_k), dtype=np.int64)
    dst = np.repeat(np.arange(n_cells, dtype=np.int64), knn_k).reshape(n_cells, knn_k)
    if not loop:
        self_mask = src == dst
        src[self_mask] = (src[self_mask] + 1) % n_cells

    edge_index = np.stack([src.reshape(-1), dst.reshape(-1)], axis=0)
    edge_rev = edge_index[[1, 0], :]
    edge_all = np.concatenate([edge_index, edge_rev], axis=1)
    edge_all = np.unique(edge_all, axis=1)
    edge_all = np.ascontiguousarray(edge_all)
    return torch.from_numpy(edge_all).long().contiguous()


def _median_knn_distance(coords: np.ndarray, knn_k: int = 8) -> float:
    """Median distance to the kth nearest non-self neighbor."""
    n_neighbors = min(knn_k + 1, coords.shape[0])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nbrs.fit(coords)
    distances = nbrs.kneighbors(coords, return_distance=True)[0]
    kth = distances[:, -1]
    return float(np.median(kth))


def _spatial_block_split(
    coords: np.ndarray,
    train_ratio: float,
    seed: int,
    grid_size: int = 5,
    buffer_knn: int = 8,
    buffer_mult: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Coordinate-grid holdout with a nearest-test-cell buffer around test blocks."""
    if coords is None:
        raise ValueError("spatial_block split requires adata.obsm['spatial']")
    if grid_size < 2:
        raise ValueError("spatial_block grid_size must be >= 2")

    n_cells = coords.shape[0]
    target_test = max(1, int(round(n_cells * (1.0 - train_ratio))))
    rng = np.random.RandomState(seed)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    xy = np.floor((coords - mins) / spans * grid_size).astype(np.int64)
    xy = np.clip(xy, 0, grid_size - 1)
    block_ids = xy[:, 0] * grid_size + xy[:, 1]

    unique_blocks = np.unique(block_ids)
    rng.shuffle(unique_blocks)
    chosen_blocks = []
    chosen_count = 0
    for block_id in unique_blocks:
        chosen_blocks.append(int(block_id))
        chosen_count += int(np.sum(block_ids == block_id))
        if chosen_count >= target_test:
            break

    test_mask = np.isin(block_ids, np.asarray(chosen_blocks, dtype=np.int64))
    test_idx = np.flatnonzero(test_mask)
    train_candidates = np.flatnonzero(~test_mask)

    buffer_radius = buffer_mult * _median_knn_distance(coords, knn_k=buffer_knn)
    if len(test_idx) > 0 and len(train_candidates) > 0 and buffer_radius > 0:
        nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nbrs.fit(coords[test_idx])
        nearest_dist = nbrs.kneighbors(coords[train_candidates], return_distance=True)[0][:, 0]
        keep_train = nearest_dist > buffer_radius
        train_idx = train_candidates[keep_train]
        n_buffer_excluded = int((~keep_train).sum())
    else:
        train_idx = train_candidates
        n_buffer_excluded = 0

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            "spatial_block split produced an empty train or test set; "
            "adjust grid_size or buffer settings"
        )

    meta = {
        'split_strategy': 'spatial_block',
        'spatial_block_grid_size': int(grid_size),
        'spatial_block_buffer_knn': int(buffer_knn),
        'spatial_block_buffer_mult': float(buffer_mult),
        'spatial_block_buffer_radius': float(buffer_radius),
        'spatial_block_test_blocks': chosen_blocks,
        'spatial_block_buffer_excluded': n_buffer_excluded,
    }
    return np.sort(train_idx), np.sort(test_idx), meta


def _rect_sum(prefix: np.ndarray, x0: int, y0: int, w: int, h: int) -> int:
    x1 = x0 + w - 1
    y1 = y0 + h - 1
    total = prefix[x1, y1]
    if x0 > 0:
        total -= prefix[x0 - 1, y1]
    if y0 > 0:
        total -= prefix[x1, y0 - 1]
    if x0 > 0 and y0 > 0:
        total += prefix[x0 - 1, y0 - 1]
    return int(total)


def _assign_tiles(
    coords: np.ndarray,
    tile_um: float,
) -> Tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
    mins = coords.min(axis=0)
    tile_xy = np.floor((coords - mins) / max(tile_um, 1e-6)).astype(np.int64)
    tile_x = tile_xy[:, 0]
    tile_y = tile_xy[:, 1]
    n_tiles_x = int(tile_x.max()) + 1
    n_tiles_y = int(tile_y.max()) + 1
    return tile_x, tile_y, n_tiles_x, n_tiles_y, mins


def _count_grid(
    tile_x: np.ndarray,
    tile_y: np.ndarray,
    n_tiles_x: int,
    n_tiles_y: int,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    counts = np.zeros((n_tiles_x, n_tiles_y), dtype=np.int64)
    if mask is None:
        sel_x = tile_x
        sel_y = tile_y
    else:
        sel_x = tile_x[mask]
        sel_y = tile_y[mask]
    if sel_x.size > 0:
        np.add.at(counts, (sel_x, sel_y), 1)
    return counts


def _window_bounds_um(
    mins: np.ndarray,
    tile_um: float,
    x0: int,
    y0: int,
    w: int,
    h: int,
) -> Dict[str, List[float]]:
    x_min = float(mins[0] + x0 * tile_um)
    x_max = float(mins[0] + (x0 + w) * tile_um)
    y_min = float(mins[1] + y0 * tile_um)
    y_max = float(mins[1] + (y0 + h) * tile_um)
    return {
        "x": [x_min, x_max],
        "y": [y_min, y_max],
    }


def _window_mask(
    tile_x: np.ndarray,
    tile_y: np.ndarray,
    x0: int,
    y0: int,
    w: int,
    h: int,
    base_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    mask = (
        (tile_x >= x0)
        & (tile_x < x0 + w)
        & (tile_y >= y0)
        & (tile_y < y0 + h)
    )
    if base_mask is not None:
        mask &= base_mask
    return mask


def _select_region_window(
    tile_x: np.ndarray,
    tile_y: np.ndarray,
    n_tiles_x: int,
    n_tiles_y: int,
    window_w: int,
    window_h: int,
    target_count: int,
    seed: int,
    base_mask: Optional[np.ndarray] = None,
) -> Tuple[int, int, int]:
    if n_tiles_x < window_w or n_tiles_y < window_h:
        raise ValueError(
            f"Window {window_w}x{window_h} does not fit inside tile grid "
            f"{n_tiles_x}x{n_tiles_y}"
        )

    counts = _count_grid(tile_x, tile_y, n_tiles_x, n_tiles_y, mask=base_mask)
    prefix = counts.cumsum(axis=0).cumsum(axis=1)

    candidates: List[Tuple[int, int, int]] = []
    best_diff: Optional[int] = None
    for x0 in range(n_tiles_x - window_w + 1):
        for y0 in range(n_tiles_y - window_h + 1):
            count = _rect_sum(prefix, x0, y0, window_w, window_h)
            if count <= 0:
                continue
            diff = abs(count - target_count)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                candidates = [(x0, y0, count)]
            elif diff == best_diff:
                candidates.append((x0, y0, count))

    if not candidates:
        raise ValueError("No non-empty candidate region window was found")

    rng = np.random.RandomState(seed)
    pick = candidates[int(rng.randint(len(candidates)))]
    return int(pick[0]), int(pick[1]), int(pick[2])


def _buffer_indices(
    coords: np.ndarray,
    anchor_idx: np.ndarray,
    candidate_idx: np.ndarray,
    radius_um: float,
) -> np.ndarray:
    if radius_um <= 0 or len(anchor_idx) == 0 or len(candidate_idx) == 0:
        return np.empty(0, dtype=np.int64)
    nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nbrs.fit(coords[anchor_idx])
    nearest_dist = nbrs.kneighbors(coords[candidate_idx], return_distance=True)[0][:, 0]
    return np.sort(candidate_idx[nearest_dist <= radius_um].astype(np.int64))


def _region_holdout_split(
    coords: np.ndarray,
    seed: int,
    tile_um: float = 800.0,
    test_window_w: int = 4,
    test_window_h: int = 3,
    test_ratio: float = 0.1,
    buffer_um: float = 80.0,
    val_window_w: int = 2,
    val_window_h: int = 2,
    val_ratio: float = 0.1,
    val_buffer_um: float = 40.0,
) -> Dict[str, np.ndarray]:
    if coords is None:
        raise ValueError("region_holdout split requires adata.obsm['spatial']")

    n_cells = coords.shape[0]
    tile_x, tile_y, n_tiles_x, n_tiles_y, mins = _assign_tiles(coords, tile_um=tile_um)

    target_test = max(1, int(round(n_cells * test_ratio)))
    test_x0, test_y0, test_count = _select_region_window(
        tile_x=tile_x,
        tile_y=tile_y,
        n_tiles_x=n_tiles_x,
        n_tiles_y=n_tiles_y,
        window_w=test_window_w,
        window_h=test_window_h,
        target_count=target_test,
        seed=seed,
    )
    test_mask = _window_mask(tile_x, tile_y, test_x0, test_y0, test_window_w, test_window_h)
    test_idx = np.flatnonzero(test_mask).astype(np.int64)

    non_test_idx = np.flatnonzero(~test_mask).astype(np.int64)
    buffer_idx = _buffer_indices(coords, test_idx, non_test_idx, radius_um=buffer_um)
    buffer_mask = np.zeros(n_cells, dtype=bool)
    buffer_mask[buffer_idx] = True

    train_pool_mask = ~(test_mask | buffer_mask)
    train_pool_idx = np.flatnonzero(train_pool_mask).astype(np.int64)
    if len(train_pool_idx) == 0:
        raise ValueError("region_holdout produced an empty train pool after buffer removal")

    target_val = max(1, int(round(len(train_pool_idx) * val_ratio)))
    val_x0, val_y0, val_count = _select_region_window(
        tile_x=tile_x,
        tile_y=tile_y,
        n_tiles_x=n_tiles_x,
        n_tiles_y=n_tiles_y,
        window_w=val_window_w,
        window_h=val_window_h,
        target_count=target_val,
        seed=seed + 1,
        base_mask=train_pool_mask,
    )
    val_mask = _window_mask(
        tile_x,
        tile_y,
        val_x0,
        val_y0,
        val_window_w,
        val_window_h,
        base_mask=train_pool_mask,
    )
    val_idx = np.flatnonzero(val_mask).astype(np.int64)
    if len(val_idx) == 0:
        raise ValueError("region_holdout produced an empty validation region")

    train_candidates_mask = train_pool_mask & ~val_mask
    train_candidates_idx = np.flatnonzero(train_candidates_mask).astype(np.int64)
    val_buffer_idx = _buffer_indices(
        coords,
        val_idx,
        train_candidates_idx,
        radius_um=val_buffer_um,
    )
    val_buffer_mask = np.zeros(n_cells, dtype=bool)
    val_buffer_mask[val_buffer_idx] = True

    train_mask = train_candidates_mask & ~val_buffer_mask
    train_idx = np.flatnonzero(train_mask).astype(np.int64)
    if len(train_idx) == 0:
        raise ValueError("region_holdout produced an empty training set")

    meta = {
        "split_strategy": "region_holdout",
        "region_tile_um": float(tile_um),
        "region_window_shape": [int(test_window_w), int(test_window_h)],
        "region_buffer_um": float(buffer_um),
        "test_region_bounds_um": _window_bounds_um(
            mins, tile_um, test_x0, test_y0, test_window_w, test_window_h
        ),
        "test_region_tile_origin": [int(test_x0), int(test_y0)],
        "test_region_count": int(test_count),
        "test_ratio_actual": float(len(test_idx) / max(n_cells, 1)),
        "region_val_window_shape": [int(val_window_w), int(val_window_h)],
        "region_val_buffer_um": float(val_buffer_um),
        "val_region_bounds_um": _window_bounds_um(
            mins, tile_um, val_x0, val_y0, val_window_w, val_window_h
        ),
        "val_region_tile_origin": [int(val_x0), int(val_y0)],
        "val_region_count": int(val_count),
        "val_ratio_actual": float(len(val_idx) / max(len(train_pool_idx), 1)),
        "n_buffer": int(len(buffer_idx)),
        "n_val_buffer": int(len(val_buffer_idx)),
        "n_train_pool": int(len(train_pool_idx)),
        "n_tiles_x": int(n_tiles_x),
        "n_tiles_y": int(n_tiles_y),
    }
    return {
        "train_idx": np.sort(train_idx),
        "val_idx": np.sort(val_idx),
        "test_idx": np.sort(test_idx),
        "buffer_idx": np.sort(buffer_idx),
        "val_buffer_idx": np.sort(val_buffer_idx),
        "train_pool_idx": np.sort(train_pool_idx),
        "split_meta": meta,
    }


def compute_split_indices(
    coords: Optional[np.ndarray],
    n_cells: int,
    train_ratio: float,
    seed: int,
    split_strategy: str = 'random',
    spatial_block_grid_size: int = 5,
    spatial_block_buffer_k: int = 8,
    spatial_block_buffer_mult: float = 2.0,
    region_tile_um: float = 800.0,
    region_test_window_w: int = 4,
    region_test_window_h: int = 3,
    region_buffer_um: float = 80.0,
    region_val_window_w: int = 2,
    region_val_window_h: int = 2,
    region_val_buffer_um: float = 40.0,
) -> Dict[str, np.ndarray]:
    if split_strategy == 'random':
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_cells)
        n_train = int(n_cells * train_ratio)
        train_idx = np.sort(indices[:n_train]).astype(np.int64)
        test_idx = np.sort(indices[n_train:]).astype(np.int64)
        split_meta = {'split_strategy': 'random'}
        return {
            'train_idx': train_idx,
            'val_idx': test_idx.copy(),
            'test_idx': test_idx,
            'buffer_idx': np.empty(0, dtype=np.int64),
            'val_buffer_idx': np.empty(0, dtype=np.int64),
            'train_pool_idx': train_idx.copy(),
            'split_meta': split_meta,
        }

    if split_strategy == 'spatial_block':
        train_idx, test_idx, split_meta = _spatial_block_split(
            coords,
            train_ratio=train_ratio,
            seed=seed,
            grid_size=spatial_block_grid_size,
            buffer_knn=spatial_block_buffer_k,
            buffer_mult=spatial_block_buffer_mult,
        )
        return {
            'train_idx': train_idx,
            'val_idx': test_idx.copy(),
            'test_idx': test_idx,
            'buffer_idx': np.empty(0, dtype=np.int64),
            'val_buffer_idx': np.empty(0, dtype=np.int64),
            'train_pool_idx': train_idx.copy(),
            'split_meta': split_meta,
        }

    if split_strategy == 'region_holdout':
        return _region_holdout_split(
            coords,
            seed=seed,
            tile_um=region_tile_um,
            test_window_w=region_test_window_w,
            test_window_h=region_test_window_h,
            test_ratio=1.0 - train_ratio,
            buffer_um=region_buffer_um,
            val_window_w=region_val_window_w,
            val_window_h=region_val_window_h,
            val_ratio=1.0 - train_ratio,
            val_buffer_um=region_val_buffer_um,
        )

    raise ValueError("split_strategy must be one of: random, spatial_block, region_holdout")


MODALITY_REGISTRY = {
    'rna_latent':      {'file': 'scvi128_h512/rna_mu.npy',          'desc': 'default scVI-128 hidden512 posterior mean'},
    'rna_var':         {'file': 'scvi128_h512/rna_var.npy',          'desc': 'default scVI-128 hidden512 posterior variance'},
    'he_cell':         {'file': 'feats_he_cell_cellmask-masked.npy', 'desc': 'UNI2-h cell CLS'},
    'he_cell_nomask':  {'file': 'feats_he_cell_cellmask-nomask.npy', 'desc': 'UNI2-h cell CLS (no mask)'},
    'he_context':      {'file': 'feats_he_context_rawctx-409.npy',   'desc': 'UNI2-h context CLS (409)'},
    'he_context_224':  {'file': 'feats_he_context_rawctx-224.npy',   'desc': 'UNI2-h context CLS (224)'},
    'he_micro':        {'file': 'feats_he_context_rawctx-224.npy',                 'desc': 'UNI2-h micro env CLS (134.4μm)'},
}


DEFAULT_RNA_FEATURE = 'scvi128_h512'
DEFAULT_RNA_FEATURE_SUBDIR = 'scvi_variants'


RNA_FEATURE_REGISTRY = {
    'scvi128_h512':    {'file': 'scvi128_h512/rna_mu.npy',                 'desc': 'scVI-128 hidden512 posterior mean'},
}


def _resolve_rna_feature_info(
    feature_dir: Path,
    rna_feature: str,
    rna_feature_subdir: str,
) -> Tuple[Path, str]:
    if rna_feature not in RNA_FEATURE_REGISTRY:
        raise ValueError(
            f"Unknown rna_feature '{rna_feature}'. Registered: {list(RNA_FEATURE_REGISTRY)}"
        )
    info = RNA_FEATURE_REGISTRY[rna_feature]
    return feature_dir / rna_feature_subdir / info['file'], info['desc']


def _resolve_rna_var_feature(rna_feature: str, rna_var_feature: str) -> str:
    if rna_var_feature != 'auto':
        return rna_var_feature
    return 'matched'


def _resolve_rna_var_feature_info(
    feature_dir: Path,
    rna_feature: str,
    rna_var_feature: str,
    rna_feature_subdir: str,
) -> Tuple[Path, str]:
    if rna_var_feature == 'matched':
        fpath, _ = _resolve_rna_feature_info(feature_dir, rna_feature, rna_feature_subdir)
        return fpath.with_name('rna_var.npy'), 'matched scVI posterior variance'
    raise ValueError("rna_var_feature must be one of: auto, matched, zeros, none")


class SpatialOmicsDataset(Dataset):
    """
    Extensible spatial multi-omics dataset.

    Each sample returns a dict:
      {
        '<modality_name>': Tensor,    # for each loaded modality
        'protein':         Tensor,    # prediction target (standardized)
        'batch_id':        Tensor,    # batch identifier
      }
    """

    def __init__(
        self,
        modalities: Dict[str, np.ndarray],
        protein: np.ndarray,
        batch_ids: np.ndarray,
    ):
        self.modalities = {k: torch.from_numpy(v).float() for k, v in modalities.items()}
        self.protein = torch.from_numpy(protein).float()
        self.batch_ids = torch.from_numpy(batch_ids).long()
        self._n = len(protein)

    def __len__(self):
        return self._n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {k: v[idx] for k, v in self.modalities.items()}
        sample['protein'] = self.protein[idx]
        sample['batch_id'] = self.batch_ids[idx]
        return sample


def _load_spatial_arrays(
    h5ad_path: str,
    feature_dir: Path,
    modalities: List[str],
    extra_features: Optional[Dict[str, np.ndarray]],
    rna_feature: str,
    rna_var_feature: str,
    rna_feature_subdir: str,
    train_ratio: float,
    seed: int,
    split_strategy: str = 'random',
    spatial_block_grid_size: int = 5,
    spatial_block_buffer_k: int = 8,
    spatial_block_buffer_mult: float = 2.0,
    region_tile_um: float = 800.0,
    region_test_window_w: int = 4,
    region_test_window_h: int = 3,
    region_buffer_um: float = 80.0,
    region_val_window_w: int = 2,
    region_val_window_h: int = 2,
    region_val_buffer_um: float = 40.0,
) -> Dict:
    """
    Shared loader: read h5ad + modality .npy, split, standardize protein on train.
    Returns dict with full-graph arrays and split indices.
    """
    print(f"Loading h5ad from {h5ad_path} ...")
    adata = sc.read_h5ad(h5ad_path)
    n_cells = adata.n_obs

    if 'protein_expression' not in adata.obsm:
        raise ValueError("adata.obsm['protein_expression'] not found")
    prot_df = adata.obsm['protein_expression']
    protein_names = list(prot_df.columns)
    prot_raw = prot_df.values.astype(np.float32)
    prot_log = np.arcsinh(prot_raw).astype(np.float32)

    if 'batch_id' in adata.obs.columns:
        batch_col = adata.obs['batch_id']
        if hasattr(batch_col, 'cat'):
            batch_ids = batch_col.cat.codes.to_numpy(dtype=np.int64)
        elif np.issubdtype(batch_col.dtype, np.integer):
            batch_ids = batch_col.to_numpy(dtype=np.int64)
        else:
            batch_ids = batch_col.astype('category').cat.codes.to_numpy(dtype=np.int64)
    else:
        batch_ids = np.zeros(n_cells, dtype=np.int64)
    n_batches = int(batch_ids.max()) + 1

    spatial_coords = adata.obsm['spatial'].astype(np.float32) if 'spatial' in adata.obsm else None

    split_info = compute_split_indices(
        coords=spatial_coords,
        n_cells=n_cells,
        train_ratio=train_ratio,
        seed=seed,
        split_strategy=split_strategy,
        spatial_block_grid_size=spatial_block_grid_size,
        spatial_block_buffer_k=spatial_block_buffer_k,
        spatial_block_buffer_mult=spatial_block_buffer_mult,
        region_tile_um=region_tile_um,
        region_test_window_w=region_test_window_w,
        region_test_window_h=region_test_window_h,
        region_buffer_um=region_buffer_um,
        region_val_window_w=region_val_window_w,
        region_val_window_h=region_val_window_h,
        region_val_buffer_um=region_val_buffer_um,
    )
    train_idx = split_info['train_idx']
    val_idx = split_info['val_idx']
    test_idx = split_info['test_idx']
    buffer_idx = split_info['buffer_idx']
    val_buffer_idx = split_info['val_buffer_idx']
    train_pool_idx = split_info['train_pool_idx']
    split_meta = split_info['split_meta']

    prot_mean = prot_log[train_idx].mean(axis=0)
    prot_std = prot_log[train_idx].std(axis=0)
    prot_std[prot_std < 1e-6] = 1.0
    prot_standardized = ((prot_log - prot_mean) / prot_std).astype(np.float32)

    loaded_features: Dict[str, np.ndarray] = {}
    resolved_rna_var = _resolve_rna_var_feature(rna_feature, rna_var_feature)
    for mod_name in modalities:
        if mod_name not in MODALITY_REGISTRY:
            raise ValueError(f"Unknown modality '{mod_name}'. Registered: {list(MODALITY_REGISTRY)}")
        if mod_name == 'rna_latent':
            fpath, desc = _resolve_rna_feature_info(
                feature_dir,
                rna_feature,
                rna_feature_subdir,
            )
            if not fpath.exists():
                raise FileNotFoundError(
                    f"RNA feature file not found: {fpath}. "
                    f"Generate it with spatialchord/train_scvi_feature.py "
                    f"or use the default extracted feature: {DEFAULT_RNA_FEATURE}."
                )
            arr = np.load(fpath).astype(np.float32)
        elif mod_name == 'rna_var':
            if resolved_rna_var == 'zeros':
                if 'rna_latent' not in loaded_features:
                    raise ValueError("rna_var=zeros requires rna_latent to be loaded first")
                arr = np.zeros_like(loaded_features['rna_latent'], dtype=np.float32)
                desc = 'zero RNA variance placeholder'
            else:
                fpath, desc = _resolve_rna_var_feature_info(
                    feature_dir,
                    rna_feature,
                    resolved_rna_var,
                    rna_feature_subdir,
                )
                if not fpath.exists():
                    raise FileNotFoundError(f"RNA variance feature file not found: {fpath}")
                arr = np.load(fpath).astype(np.float32)
        else:
            info = MODALITY_REGISTRY[mod_name]
            fpath = feature_dir / info['file']
            desc = info['desc']
            if not fpath.exists():
                raise FileNotFoundError(f"Feature file not found: {fpath}")
            arr = np.load(fpath).astype(np.float32)
        if arr.shape[0] != n_cells:
            raise ValueError(f"Feature {mod_name} has {arr.shape[0]} rows, expected {n_cells}")
        loaded_features[mod_name] = arr
        print(f"  Loaded {mod_name}: {arr.shape}  ({desc})")

    if extra_features:
        for feat_name, feat_arr in extra_features.items():
            feat_arr = feat_arr.astype(np.float32)
            if feat_arr.shape[0] != n_cells:
                raise ValueError(f"Extra feature '{feat_name}' has {feat_arr.shape[0]} rows, expected {n_cells}")
            loaded_features[feat_name] = feat_arr
            print(f"  Loaded {feat_name}: {feat_arr.shape}  (extra)")

    return {
        'n_cells': n_cells,
        'protein_names': protein_names,
        'prot_raw': prot_raw,
        'prot_log': prot_log,
        'prot_standardized': prot_standardized,
        'batch_ids': batch_ids,
        'n_batches': n_batches,
        'spatial_coords': spatial_coords,
        'loaded_features': loaded_features,
        'prot_mean': prot_mean,
        'prot_std': prot_std,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'buffer_idx': buffer_idx,
        'val_buffer_idx': val_buffer_idx,
        'train_pool_idx': train_pool_idx,
        'split_meta': split_meta,
    }


def load_spatial_data(
    h5ad_path: str,
    feature_dir: str,
    modalities: List[str] = ('rna_latent',),
    extra_features: Optional[Dict[str, np.ndarray]] = None,
    rna_feature: str = DEFAULT_RNA_FEATURE,
    rna_var_feature: str = 'auto',
    rna_feature_subdir: str = DEFAULT_RNA_FEATURE_SUBDIR,
    train_ratio: float = 0.9,
    seed: int = 42,
    split_strategy: str = 'random',
    spatial_block_grid_size: int = 5,
    spatial_block_buffer_k: int = 8,
    spatial_block_buffer_mult: float = 2.0,
    region_tile_um: float = 800.0,
    region_test_window_w: int = 4,
    region_test_window_h: int = 3,
    region_buffer_um: float = 80.0,
    region_val_window_w: int = 2,
    region_val_window_h: int = 2,
    region_val_buffer_um: float = 40.0,
) -> Tuple[SpatialOmicsDataset, SpatialOmicsDataset, SpatialOmicsDataset, Dict]:
    """
    Load spatial omics data and create train/test datasets.

    Protein preprocessing: log normalize → per-protein standardize (on train set).

    Args:
        h5ad_path:      path to xenium_rna_prot.h5ad
        feature_dir:    directory containing pre-computed .npy feature files
        modalities:     which modalities to load (keys of MODALITY_REGISTRY)
        extra_features: additional {name: np.ndarray} features to inject
                        (bypasses registry, e.g. for switching HE files at CLI)
        train_ratio:    fraction used for training (rest for test)
        seed:           random seed for reproducible split

    Returns:
        train_dataset, test_dataset, data_info dict
    """
    feature_dir = Path(feature_dir)
    core = _load_spatial_arrays(
        h5ad_path,
        feature_dir,
        list(modalities),
        extra_features,
        rna_feature,
        rna_var_feature,
        rna_feature_subdir,
        train_ratio,
        seed,
        split_strategy=split_strategy,
        spatial_block_grid_size=spatial_block_grid_size,
        spatial_block_buffer_k=spatial_block_buffer_k,
        spatial_block_buffer_mult=spatial_block_buffer_mult,
        region_tile_um=region_tile_um,
        region_test_window_w=region_test_window_w,
        region_test_window_h=region_test_window_h,
        region_buffer_um=region_buffer_um,
        region_val_window_w=region_val_window_w,
        region_val_window_h=region_val_window_h,
        region_val_buffer_um=region_val_buffer_um,
    )
    n_cells = core['n_cells']
    train_idx = core['train_idx']
    val_idx = core['val_idx']
    test_idx = core['test_idx']
    loaded_features = core['loaded_features']
    prot_standardized = core['prot_standardized']
    prot_log = core['prot_log']
    prot_raw = core['prot_raw']
    batch_ids = core['batch_ids']

    def _subset(feats, idx):
        return {k: v[idx] for k, v in feats.items()}

    train_dataset = SpatialOmicsDataset(
        modalities=_subset(loaded_features, train_idx),
        protein=prot_standardized[train_idx],
        batch_ids=batch_ids[train_idx],
    )
    val_dataset = SpatialOmicsDataset(
        modalities=_subset(loaded_features, val_idx),
        protein=prot_standardized[val_idx],
        batch_ids=batch_ids[val_idx],
    )
    test_dataset = SpatialOmicsDataset(
        modalities=_subset(loaded_features, test_idx),
        protein=prot_standardized[test_idx],
        batch_ids=batch_ids[test_idx],
    )

    data_info = {
        'n_cells': n_cells,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'n_buffer': int(len(core['buffer_idx'])),
        'n_val_buffer': int(len(core['val_buffer_idx'])),
        'n_proteins': len(core['protein_names']),
        'protein_names': core['protein_names'],
        'n_batches': core['n_batches'],
        'prot_mean': core['prot_mean'],
        'prot_std': core['prot_std'],
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'buffer_idx': core['buffer_idx'],
        'val_buffer_idx': core['val_buffer_idx'],
        'train_pool_idx': core['train_pool_idx'],
        'modalities': list(loaded_features.keys()),
        'modality_dims': {k: v.shape[1] for k, v in loaded_features.items()},
        'spatial_coords': core['spatial_coords'],
        'prot_log_train': prot_log[train_idx],
        'prot_log_val': prot_log[val_idx],
        'prot_log_test': prot_log[test_idx],
        'prot_raw_train': prot_raw[train_idx],
        'prot_raw_val': prot_raw[val_idx],
        'prot_raw_test': prot_raw[test_idx],
        **core['split_meta'],
    }

    print(
        f"Dataset ready: {data_info['n_train']} train / "
        f"{data_info['n_val']} val / {data_info['n_test']} test"
    )
    print(f"Proteins: {data_info['n_proteins']}, Batches: {core['n_batches']}")
    return train_dataset, val_dataset, test_dataset, data_info


def load_spatial_graph_data(
    h5ad_path: str,
    feature_dir: str,
    modalities: List[str],
    extra_features: Optional[Dict[str, np.ndarray]] = None,
    rna_feature: str = DEFAULT_RNA_FEATURE,
    rna_var_feature: str = 'auto',
    rna_feature_subdir: str = DEFAULT_RNA_FEATURE_SUBDIR,
    train_ratio: float = 0.9,
    seed: int = 42,
    knn_k: int = 16,
    loop: bool = False,
    split_strategy: str = 'random',
    spatial_block_grid_size: int = 5,
    spatial_block_buffer_k: int = 8,
    spatial_block_buffer_mult: float = 2.0,
    region_tile_um: float = 800.0,
    region_test_window_w: int = 4,
    region_test_window_h: int = 3,
    region_buffer_um: float = 80.0,
    region_val_window_w: int = 2,
    region_val_window_h: int = 2,
    region_val_buffer_um: float = 40.0,
    spatial_control: str = 'true_knn',
) -> Tuple['Data', Dict]:
    """
    Full-graph tensors + kNN edge_index for PyG NeighborLoader (+SpatialAttn).

    Requires torch_geometric. Builds undirected kNN from ``adata.obsm['spatial']`` (μm).

    Returns:
        data: torch_geometric.data.Data with node features and ``edge_index``.
        data_info: same keys as ``load_spatial_data`` plus ``knn_k``, ``edge_index`` stats.
    """
    Data, _ = _import_pyg()
    feature_dir = Path(feature_dir)
    core = _load_spatial_arrays(
        h5ad_path,
        feature_dir,
        list(modalities),
        extra_features,
        rna_feature,
        rna_var_feature,
        rna_feature_subdir,
        train_ratio,
        seed,
        split_strategy=split_strategy,
        spatial_block_grid_size=spatial_block_grid_size,
        spatial_block_buffer_k=spatial_block_buffer_k,
        spatial_block_buffer_mult=spatial_block_buffer_mult,
        region_tile_um=region_tile_um,
        region_test_window_w=region_test_window_w,
        region_test_window_h=region_test_window_h,
        region_buffer_um=region_buffer_um,
        region_val_window_w=region_val_window_w,
        region_val_window_h=region_val_window_h,
        region_val_buffer_um=region_val_buffer_um,
    )
    coords = core['spatial_coords']
    if coords is None:
        raise ValueError("adata.obsm['spatial'] required for +SpatialAttn / kNN graph")

    n_cells = core['n_cells']
    loaded_features = core['loaded_features']
    prot_standardized = core['prot_standardized']
    batch_ids = core['batch_ids']
    train_idx = core['train_idx']
    test_idx = core['test_idx']

    if spatial_control == 'true_knn':
        graph_coords = coords
        edge_index = _build_knn_edge_index(graph_coords, knn_k=knn_k, loop=loop)
    elif spatial_control == 'random_neighbors':
        graph_coords = coords
        edge_index = _build_random_edge_index(n_cells, knn_k=knn_k, seed=seed, loop=loop)
    elif spatial_control == 'permuted_coords':
        rng = np.random.RandomState(seed)
        graph_coords = coords[rng.permutation(n_cells)]
        edge_index = _build_knn_edge_index(graph_coords, knn_k=knn_k, loop=loop)
    else:
        raise ValueError(
            "spatial_control must be one of: true_knn, random_neighbors, permuted_coords"
        )

    pos = torch.from_numpy(np.ascontiguousarray(graph_coords)).float().contiguous()

    data_dict = {
        'pos': pos,
        'edge_index': edge_index,
        'protein': torch.from_numpy(np.ascontiguousarray(prot_standardized)).float().contiguous(),
        'batch_id': torch.from_numpy(np.ascontiguousarray(batch_ids)).long().contiguous(),
    }
    for k, arr in loaded_features.items():
        data_dict[k] = torch.from_numpy(np.ascontiguousarray(arr)).float().contiguous()

    data = Data(**data_dict)
    data.num_nodes = int(n_cells)
    data.train_mask = torch.zeros(n_cells, dtype=torch.bool)
    data.train_mask[torch.from_numpy(np.ascontiguousarray(train_idx)).long()] = True
    data.test_mask = torch.zeros(n_cells, dtype=torch.bool)
    data.test_mask[torch.from_numpy(np.ascontiguousarray(test_idx)).long()] = True

    data_info = {
        'n_cells': n_cells,
        'n_train': len(train_idx),
        'n_val': len(core['val_idx']),
        'n_test': len(test_idx),
        'n_buffer': int(len(core['buffer_idx'])),
        'n_proteins': len(core['protein_names']),
        'protein_names': core['protein_names'],
        'n_batches': core['n_batches'],
        'prot_mean': core['prot_mean'],
        'prot_std': core['prot_std'],
        'train_idx': train_idx,
        'val_idx': core['val_idx'],
        'test_idx': test_idx,
        'buffer_idx': core['buffer_idx'],
        'val_buffer_idx': core['val_buffer_idx'],
        'train_pool_idx': core['train_pool_idx'],
        'modalities': list(loaded_features.keys()),
        'modality_dims': {k: v.shape[1] for k, v in loaded_features.items()},
        'spatial_coords': coords,
        'prot_log_train': core['prot_log'][train_idx],
        'prot_log_val': core['prot_log'][core['val_idx']],
        'prot_log_test': core['prot_log'][test_idx],
        'prot_raw_train': core['prot_raw'][train_idx],
        'prot_raw_val': core['prot_raw'][core['val_idx']],
        'prot_raw_test': core['prot_raw'][test_idx],
        'knn_k': knn_k,
        'num_edges': int(edge_index.shape[1]),
        'spatial_control': spatial_control,
        'graph_eval_index_mode': 'global',
        **core['split_meta'],
    }
    print(
        f"Graph data ready: {n_cells} nodes, {data_info['num_edges']} edges "
        f"(k={knn_k}, control={spatial_control}, symmetrized)"
    )
    print(f"Dataset ready: {data_info['n_train']} train / {data_info['n_test']} test")
    print(f"Proteins: {data_info['n_proteins']}, Batches: {core['n_batches']}")
    return data, data_info


def _build_graph_subset(
    data_cls,
    coords: np.ndarray,
    loaded_features: Dict[str, np.ndarray],
    prot_standardized: np.ndarray,
    batch_ids: np.ndarray,
    subset_idx: np.ndarray,
    knn_k: int,
    loop: bool,
    spatial_control: str,
    seed: int,
):
    subset_idx = np.asarray(subset_idx, dtype=np.int64)
    subset_coords = np.ascontiguousarray(coords[subset_idx]).astype(np.float32)
    if spatial_control == 'true_knn':
        graph_coords = subset_coords
        edge_index = _build_knn_edge_index(graph_coords, knn_k=knn_k, loop=loop)
    elif spatial_control == 'random_neighbors':
        graph_coords = subset_coords
        edge_index = _build_random_edge_index(
            len(subset_idx), knn_k=knn_k, seed=seed, loop=loop
        )
    elif spatial_control == 'permuted_coords':
        rng = np.random.RandomState(seed)
        graph_coords = subset_coords[rng.permutation(len(subset_idx))]
        edge_index = _build_knn_edge_index(graph_coords, knn_k=knn_k, loop=loop)
    else:
        raise ValueError(
            "spatial_control must be one of: true_knn, random_neighbors, permuted_coords"
        )

    data_dict = {
        'pos': torch.from_numpy(graph_coords).float().contiguous(),
        'edge_index': edge_index,
        'protein': torch.from_numpy(
            np.ascontiguousarray(prot_standardized[subset_idx])
        ).float().contiguous(),
        'batch_id': torch.from_numpy(
            np.ascontiguousarray(batch_ids[subset_idx])
        ).long().contiguous(),
        'global_idx': torch.from_numpy(np.ascontiguousarray(subset_idx)).long().contiguous(),
    }
    for key, arr in loaded_features.items():
        data_dict[key] = torch.from_numpy(np.ascontiguousarray(arr[subset_idx])).float().contiguous()

    data = data_cls(**data_dict)
    data.num_nodes = int(len(subset_idx))
    return data


def load_spatial_graph_split_data(
    h5ad_path: str,
    feature_dir: str,
    modalities: List[str],
    extra_features: Optional[Dict[str, np.ndarray]] = None,
    rna_feature: str = DEFAULT_RNA_FEATURE,
    rna_var_feature: str = 'auto',
    rna_feature_subdir: str = DEFAULT_RNA_FEATURE_SUBDIR,
    train_ratio: float = 0.9,
    seed: int = 42,
    knn_k: int = 16,
    loop: bool = False,
    split_strategy: str = 'region_holdout',
    spatial_block_grid_size: int = 5,
    spatial_block_buffer_k: int = 8,
    spatial_block_buffer_mult: float = 2.0,
    region_tile_um: float = 800.0,
    region_test_window_w: int = 4,
    region_test_window_h: int = 3,
    region_buffer_um: float = 80.0,
    region_val_window_w: int = 2,
    region_val_window_h: int = 2,
    region_val_buffer_um: float = 40.0,
    spatial_control: str = 'true_knn',
) -> Tuple[Dict[str, 'Data'], Dict]:
    Data, _ = _import_pyg()
    feature_dir = Path(feature_dir)
    core = _load_spatial_arrays(
        h5ad_path,
        feature_dir,
        list(modalities),
        extra_features,
        rna_feature,
        rna_var_feature,
        rna_feature_subdir,
        train_ratio,
        seed,
        split_strategy=split_strategy,
        spatial_block_grid_size=spatial_block_grid_size,
        spatial_block_buffer_k=spatial_block_buffer_k,
        spatial_block_buffer_mult=spatial_block_buffer_mult,
        region_tile_um=region_tile_um,
        region_test_window_w=region_test_window_w,
        region_test_window_h=region_test_window_h,
        region_buffer_um=region_buffer_um,
        region_val_window_w=region_val_window_w,
        region_val_window_h=region_val_window_h,
        region_val_buffer_um=region_val_buffer_um,
    )
    coords = core['spatial_coords']
    if coords is None:
        raise ValueError("adata.obsm['spatial'] required for +SpatialAttn / kNN graph")

    loaded_features = core['loaded_features']
    prot_standardized = core['prot_standardized']
    batch_ids = core['batch_ids']
    train_idx = core['train_idx']
    val_idx = core['val_idx']
    test_idx = core['test_idx']

    split_data = {
        'train': _build_graph_subset(
            Data,
            coords,
            loaded_features,
            prot_standardized,
            batch_ids,
            train_idx,
            knn_k=knn_k,
            loop=loop,
            spatial_control=spatial_control,
            seed=seed,
        ),
        'val': _build_graph_subset(
            Data,
            coords,
            loaded_features,
            prot_standardized,
            batch_ids,
            val_idx,
            knn_k=knn_k,
            loop=loop,
            spatial_control=spatial_control,
            seed=seed + 1,
        ),
        'test': _build_graph_subset(
            Data,
            coords,
            loaded_features,
            prot_standardized,
            batch_ids,
            test_idx,
            knn_k=knn_k,
            loop=loop,
            spatial_control=spatial_control,
            seed=seed + 2,
        ),
    }

    data_info = {
        'n_cells': core['n_cells'],
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'n_buffer': int(len(core['buffer_idx'])),
        'n_val_buffer': int(len(core['val_buffer_idx'])),
        'n_proteins': len(core['protein_names']),
        'protein_names': core['protein_names'],
        'n_batches': core['n_batches'],
        'prot_mean': core['prot_mean'],
        'prot_std': core['prot_std'],
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'buffer_idx': core['buffer_idx'],
        'val_buffer_idx': core['val_buffer_idx'],
        'train_pool_idx': core['train_pool_idx'],
        'modalities': list(loaded_features.keys()),
        'modality_dims': {k: v.shape[1] for k, v in loaded_features.items()},
        'spatial_coords': coords,
        'prot_log_train': core['prot_log'][train_idx],
        'prot_log_val': core['prot_log'][val_idx],
        'prot_log_test': core['prot_log'][test_idx],
        'prot_raw_train': core['prot_raw'][train_idx],
        'prot_raw_val': core['prot_raw'][val_idx],
        'prot_raw_test': core['prot_raw'][test_idx],
        'knn_k': knn_k,
        'num_edges_train': int(split_data['train'].edge_index.shape[1]),
        'num_edges_val': int(split_data['val'].edge_index.shape[1]),
        'num_edges_test': int(split_data['test'].edge_index.shape[1]),
        'spatial_control': spatial_control,
        'graph_eval_index_mode': 'local',
        **core['split_meta'],
    }
    print(
        f"Graph split data ready: train={len(train_idx)}, val={len(val_idx)}, "
        f"test={len(test_idx)}, control={spatial_control}, k={knn_k}"
    )
    return split_data, data_info


def get_neighbor_loader(
    data: 'Data',
    input_nodes: torch.Tensor,
    batch_size: int,
    num_neighbors: List[int],
    shuffle: bool = True,
    num_workers: int = 4,
    subgraph_type: str = 'induced',
) -> 'NeighborLoader':
    """PyG NeighborLoader on full-graph ``data``; ``input_nodes`` = seed global indices.

    Default ``subgraph_type='induced'`` keeps all edges between sampled nodes so
    SpatialNeighborAttention sees a consistent local neighborhood.
    """
    _, NeighborLoader = _import_pyg()
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=input_nodes.contiguous(),
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        subgraph_type=subgraph_type,
    )


def get_dataloader(
    dataset: SpatialOmicsDataset,
    batch_size: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
