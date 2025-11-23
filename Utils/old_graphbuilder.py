import torch
from torch_geometric.data import Data
import numpy as np

def _to_tensor(x, dtype=torch.float32, device=None):
    if isinstance(x, torch.Tensor):
        t = x.to(device) if device is not None else x
    else:
        try:
            import numpy as _np
            t = torch.tensor(x, dtype=dtype, device=device)
        except Exception:
            t = torch.as_tensor(x, dtype=dtype, device=device)
    return t

def _knn_edge_index_from_numpy(x_np, k):
    """
    Build edge_index (2, N*k) from numpy array x_np using sklearn NearestNeighbors.
    Returns a torch.LongTensor on CPU (we'll move to device later).
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as e:
        raise RuntimeError("scikit-learn is required for the fallback KNN. Install it with `pip install scikit-learn`.") from e

    # set n_neighbors = k+1 to include self, then drop self
    n_neighbors = min(k + 1, x_np.shape[0])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean').fit(x_np)
    distances, indices = nbrs.kneighbors(x_np)  # indices: (N, n_neighbors)
    # drop the first column which is the point itself (if present)
    if indices.shape[1] > 1:
        knn = indices[:, 1:n_neighbors]  # (N, k) or fewer if small N
    else:
        knn = np.zeros((x_np.shape[0], 0), dtype=np.int64)
    N = x_np.shape[0]
    if knn.size == 0:
        # no neighbors (e.g., N==1), return empty edge_index
        return torch.empty((2, 0), dtype=torch.long)
    src = np.repeat(np.arange(N), knn.shape[1])
    dst = knn.reshape(-1)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)  # (2, N*k)
    return torch.from_numpy(edge_index)

def build_sim_graph(x, y=None, k: int = 5):
    """
    Build a similarity graph (k-NN) from node feature vectors `x`.
    Returns a torch_geometric.data.Data object with fields:
      - x: node features (Tensor[N, F])
      - edge_index: long Tensor[2, E]
      - y: optional node labels (kept as-is if provided)
    """
    x = _to_tensor(x, dtype=torch.float32)
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    num_nodes = x.shape[0]
    if num_nodes == 0:
        raise ValueError("build_sim_graph received empty x")

    # Prefer torch_geometric.knn_graph if available (requires torch-cluster)
    try:
        from torch_geometric.nn import knn_graph
        edge_index = knn_graph(x, k=k, batch=None, loop=False)
    except Exception:
        # Fallback: use sklearn NearestNeighbors (memory-efficient)
        x_np = x.cpu().numpy()
        edge_index = _knn_edge_index_from_numpy(x_np, k).to(x.device)

    data = Data(x=x, edge_index=edge_index)
    if y is not None:
        data.y = _to_tensor(y, dtype=torch.long)
    return data

def build_spat_graph(patch_centers, x_features=None, y=None, k: int = 4):
    """
    Build a spatial graph (k-NN) from patch_centers (coordinates).
    - patch_centers: array-like or Tensor of shape (N, D) (D typically 2 or 3)
    - x_features: optional node features to attach as data.x; if None, patch_centers used as features
    - y: optional node labels
    - k: number of neighbors for k-NN
    Returns: torch_geometric.data.Data
    """
    coords = _to_tensor(patch_centers, dtype=torch.float32)
    if coords.dim() == 1:
        coords = coords.unsqueeze(-1)
    num_nodes = coords.shape[0]
    if num_nodes == 0:
        raise ValueError("build_spat_graph received empty patch_centers")

    try:
        from torch_geometric.nn import knn_graph
        edge_index = knn_graph(coords, k=k, batch=None, loop=False)
    except Exception:
        coords_np = coords.cpu().numpy()
        edge_index = _knn_edge_index_from_numpy(coords_np, k).to(coords.device)

    # choose node features
    if x_features is None:
        x = coords
    else:
        x = _to_tensor(x_features, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

    data = Data(x=x, edge_index=edge_index)
    if y is not None:
        data.y = _to_tensor(y, dtype=torch.long)
    return data