import numpy as np
from new_pipeline import config


def _build_csr(neighbors, conductances):
    """Convert neighbor lists + dict to flat CSR arrays with precomputed weights.

    Returns:
        nbr_offsets: (n+1,) int32 cumulative offsets into flat arrays
        nbr_indices: (total_edges,) int32 neighbor face indices
        nbr_weights: (total_edges,) float64 normalized G_ij / Σ_k G_ik
    """
    n = len(neighbors)
    counts = np.array([len(nbrs) for nbrs in neighbors], dtype=np.int32)
    total = int(counts.sum())

    offsets = np.empty(n + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    indices = np.empty(total, dtype=np.int32)
    weights = np.empty(total, dtype=np.float64)

    for i, nbrs in enumerate(neighbors):
        start = offsets[i]
        end = offsets[i + 1]
        G_sum = 0.0
        for k, j in enumerate(nbrs):
            G = conductances.get((i, j), 0.0)
            indices[start + k] = j
            weights[start + k] = G
            G_sum += G
        if G_sum > 1e-20:
            inv = 1.0 / G_sum
            weights[start:end] *= inv
        else:
            weights[start:end] = 0.0

    return offsets, indices, weights


def gauss_seidel(T, neighbors, conductances, fixed_faces=None, tol=None, max_iter=None):
    """Run Gauss-Seidel diffusion on a mesh graph until steady state.

    Gauss-Seidel update for free face i:
        T_i_new = Σ_j (w_ij * T_j)    where w_ij = G_ij / Σ_k G_ik

    Fixed faces (Dirichlet BCs) are skipped. Boundary faces with fewer neighbors
    are handled naturally (equivalent to adiabatic edge condition).

    Uses precomputed CSR arrays with normalized weights to avoid dict lookups
    and repeated summation during iteration.
    """
    if tol is None:
        tol = config.DIFFUSION_TOL
    if max_iter is None:
        max_iter = config.MAX_ITERATIONS
    if fixed_faces is None:
        fixed_faces = set()

    n = len(T)
    T = T.copy()

    # Build CSR sparse system (one-time cost)
    offsets, nbr_idx, nbr_w = _build_csr(neighbors, conductances)

    # Boolean mask for O(1) fixed-face check (vs O(1) hash set is also fine,
    # but bool array avoids per-element hash in inner loop)
    is_fixed = np.zeros(n, dtype=bool)
    if fixed_faces:
        fixed_arr = np.fromiter(fixed_faces, dtype=np.int32)
        is_fixed[fixed_arr] = True

    for iteration in range(1, max_iter + 1):
        max_change = 0.0

        for i in range(n):
            if is_fixed[i]:
                continue

            start = int(offsets[i])
            end = int(offsets[i + 1])
            if start == end:
                continue

            # Dot product: T_new = Σ w_k * T[nbr_k]
            w_slice = nbr_w[start:end]
            t_slice = T[nbr_idx[start:end]]
            T_new = np.dot(w_slice, t_slice)

            change = abs(T_new - T[i])
            if change > max_change:
                max_change = change
            T[i] = T_new

        if max_change < tol:
            return T, iteration, max_change

    return T, max_iter, max_change
