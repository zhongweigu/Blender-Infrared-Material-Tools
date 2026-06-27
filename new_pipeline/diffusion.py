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


def gauss_seidel(T, neighbors, conductances, fixed_faces=None, tol=None, max_iter=None,
                 decay=0.0, T_amb=280.0):
    """Run Gauss-Seidel diffusion on a mesh graph until steady state.

    Gauss-Seidel update for free face i:
        T_i_new = (1-α)·Σ_j (w_ij · T_j) + α·T_amb

    where w_ij = G_ij / Σ_k G_ik, and α = decay pulls toward ambient.

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

    one_minus_decay = 1.0 - decay

    def _sweep(indices):
        max_c = 0.0
        for i in indices:
            if is_fixed[i]:
                continue
            start = int(offsets[i])
            end = int(offsets[i + 1])
            if start == end:
                continue
            w_slice = nbr_w[start:end]
            t_slice = T[nbr_idx[start:end]]
            T_neighbor = np.dot(w_slice, t_slice)
            T_new = one_minus_decay * T_neighbor + decay * T_amb
            c = abs(T_new - T[i])
            if c > max_c:
                max_c = c
            T[i] = T_new
        return max_c

    for iteration in range(1, max_iter + 1):
        # Alternate which sweep goes first to cancel directional bias
        if iteration % 2 == 1:
            c1 = _sweep(range(n))                      # forward first
            c2 = _sweep(range(n - 1, -1, -1))           # reverse second
        else:
            c2 = _sweep(range(n - 1, -1, -1))           # reverse first
            c1 = _sweep(range(n))                      # forward second
        if max(c1, c2) < tol:
            return T, iteration, max(c1, c2)

    return T, max_iter, max(c1, c2)
