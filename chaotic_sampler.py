import numpy as np
import copy
from typing import Dict, Any, Optional


def chaotic_sampler_run(
    data: np.ndarray,
    *,
    B: int = 100,
    B_finos: int = 50,
    use_simulated_annealing: bool = True,
    sa_initial_temp: float = 1.0,
    sa_final_temp: float = 1e-3,
    sa_alpha: float = 0.99,
    sa_iterations: int = 20,
    # single-seed
    single_seed_index: int = 340,
    N_iter: int = 100000,
    # multi-seed
    num_rounds: int = 1,
    # params finos
    expand_if_tie: bool = True,
    eps_rel: float = 4.0,
    eps_abs: float = 2.0,
    span_umbral: float = 0.2,
    # reproducibilidad
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the deterministic chaotic latent-space sampler.

    Pipeline:
      1) Compute per-dimension min/max and coarse thresholds
         (optionally optimized via simulated annealing).
      2) Build coarse discretization and context hashes.
      3) Construct conditional fine histograms and local chaotic maps.
      4) Generate:
         - a single-seed trajectory (XX)
         - a multi-seed ensemble evolved for several rounds (SAL2)

    Parameters
    ----------
    data : ndarray, shape (N, D)
        Latent vectors (e.g., autoencoder latent codes).
    B : int
        Number of coarse bins per dimension.
    B_finos : int
        Number of fine bins per conditional chaotic map.
    use_simulated_annealing : bool
        Whether to optimize coarse thresholds.
    single_seed_index : int
        Index of the latent vector used for single-seed dynamics.
    N_iter : int
        Number of iterations for the single-seed trajectory.
    num_rounds : int
        Number of global iterations in multi-seed mode.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    results : dict
        Dictionary containing thresholds, trajectories, and metadata.
    """
    if seed is not None:
        np.random.seed(seed)

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data debe ser un array 2D (N, D)")
    N, D = data.shape
    if B < 2:
        raise ValueError("B debe ser >= 2")
    if B_finos < 1:
        raise ValueError("B_finos debe ser >= 1")
    if not (0 <= single_seed_index < N):
        raise ValueError("single_seed_index fuera de rango")

    # ============================================================
    # 1. Utilidades básicas (tal cual tu código)
    # ============================================================

    def obtener_minimos_maximos(matriz):
        Dloc = len(matriz[0])
        minimos = [float('inf')] * Dloc
        maximos = [float('-inf')] * Dloc
        for fila in matriz:
            for i in range(Dloc):
                if fila[i] < minimos[i]:
                    minimos[i] = fila[i]
                if fila[i] > maximos[i]:
                    maximos[i] = fila[i]
        return minimos, maximos

    def compute_hashes(discretized_matrix, B):
        M = discretized_matrix.shape[1]
        exponents = np.array([B ** (M - 1 - i) for i in range(M)], dtype=object)
        return discretized_matrix.dot(exponents)

    def discretize_column(values, thresholds, B):
        bins = np.digitize(values, thresholds, right=False) - 1
        bins = np.clip(bins, 0, B - 1)
        return bins

    def evaluate_coverage(data, thresholds, B, left_out):
        Dloc = data.shape[1]
        used = [d for d in range(Dloc) if d != left_out]
        Nloc = data.shape[0]
        M = len(used)

        discretized = np.empty((Nloc, M), dtype=int)
        for j, d in enumerate(used):
            discretized[:, j] = discretize_column(data[:, d], thresholds[d], B)

        hashes = compute_hashes(discretized, B)
        unique_hashes = np.unique(hashes)
        total_possible = B ** M
        return len(unique_hashes), total_possible, hashes

    def objective(thresholds, data, B):
        total_missing = 0
        Dloc = data.shape[1]
        for left_out in range(Dloc):
            unique_count, total_possible, _ = evaluate_coverage(data, thresholds, B, left_out)
            total_missing += (total_possible - unique_count)
        return total_missing

    def random_move(thresholds, column, move_scale=0.01):
        new_thresholds = copy.deepcopy(thresholds)
        t = new_thresholds[column].copy()
        i = np.random.randint(1, len(t) - 1)
        min_val = t[0]
        max_val = t[-1]
        delta = (max_val - min_val) * np.random.uniform(-move_scale, move_scale)
        t[i] = np.clip(t[i] + delta, t[i - 1] + 1e-8, t[i + 1] - 1e-8)
        new_thresholds[column] = t
        return new_thresholds

    def simulated_annealing(data, initial_thresholds, B,
                            initial_temp=1.0, final_temp=1e-3,
                            alpha=0.99, iterations=20):
        current_thresholds = copy.deepcopy(initial_thresholds)
        current_obj = objective(current_thresholds, data, B)
        best_thresholds = copy.deepcopy(current_thresholds)
        best_obj = current_obj
        temp = initial_temp

        while temp > final_temp:
            for _ in range(iterations):
                column = np.random.choice(list(current_thresholds.keys()))
                new_thresholds = random_move(current_thresholds, column)
                new_obj = objective(new_thresholds, data, B)
                delta = new_obj - current_obj
                if delta < 0 or np.random.rand() < np.exp(-delta / temp):
                    current_thresholds = new_thresholds
                    current_obj = new_obj
                    if current_obj < best_obj:
                        best_thresholds = copy.deepcopy(current_thresholds)
                        best_obj = current_obj
            temp *= alpha
        return best_thresholds, best_obj

    # ============================================================
    # 2) min/max, thresholds y (opcional) SA
    # ============================================================
    minimos, maximos = obtener_minimos_maximos(data)
    thresholds = {d: np.linspace(minimos[d], maximos[d], B + 1) for d in range(D)}

    if use_simulated_annealing:
        limites1, best_obj = simulated_annealing(
            data, thresholds, B,
            initial_temp=sa_initial_temp,
            final_temp=sa_final_temp,
            alpha=sa_alpha,
            iterations=sa_iterations
        )
        limites = np.array([limites1[k] for k in sorted(limites1.keys())], dtype=float)
    else:
        limites = np.array([thresholds[d] for d in range(D)], dtype=float)
        best_obj = None

    # ============================================================
    # 3) Discretización gruesa + comb_info_por_esel (tal cual)
    # ============================================================
    discs = np.stack([discretize_column(data[:, d], limites[d], B) for d in range(D)], axis=1)
    M = max(D - 1, 0)

    if M > 0:
        exponents_global = np.array([B ** (M - 1 - i) for i in range(M)], dtype=object)
    else:
        exponents_global = np.array([], dtype=object)

    def compute_hashes_local(mat):
        if mat.shape[1] == 0:
            return np.zeros((mat.shape[0],), dtype=object)
        return mat.dot(exponents_global)

    comb_info_por_esel = []
    for d in range(D):
        if D == 1:
            comb_indices = np.zeros(N, dtype=object)
        else:
            otras = [j for j in range(D) if j != d]
            comb_indices = compute_hashes_local(discs[:, otras])

        order = np.argsort(comb_indices)
        comb_sorted = comb_indices[order]
        keys, idx_start, counts = np.unique(comb_sorted, return_index=True, return_counts=True)

        if M > 0 and keys.size > 0:
            digits_keys = np.zeros((keys.size, M), dtype=int)
            for ki, key in enumerate(keys):
                tmp = int(key)
                for pos in range(M - 1, -1, -1):
                    digits_keys[ki, pos] = tmp % B
                    tmp //= B
        else:
            digits_keys = np.zeros((keys.size, 0), dtype=int)

        comb_info_por_esel.append({
            "keys": keys,
            "idx_start": idx_start,
            "counts": counts,
            "order": order,
            "digits_keys": digits_keys
        })

    # ============================================================
    # 4) Mapa local (A), normalización y selección de caja
    # ============================================================
    def matriz_A(valores_de_bin):
        Dloc = len(valores_de_bin)
        beta = 0.9
        alfa = np.zeros(Dloc)
        for i in range(Dloc):
            alfa[i] = valores_de_bin[i] * beta
        if np.sum(alfa) == 0:
            alfa[:] = 1.0
        alfa = alfa / np.sum(alfa)

        A = np.zeros((Dloc, Dloc))
        for fil in range(Dloc):
            for col in range(Dloc):
                A[fil, col] = alfa[fil] * 0.9
                if fil == col:
                    A[fil, col] = A[fil, col] + 0.9
        if np.sum(A[:, 0]) != 0:
            A = A / np.sum(A[:, 0])
        return A

    def vector_a_decimal(vector, B):
        vector_invertido = vector[::-1]
        numero_decimal = 0
        for i, digito in enumerate(vector_invertido):
            numero_decimal += digito * (B ** i)
        return numero_decimal

    def normalizar_0_1(valor, minimo, maximo):
        if maximo == minimo:
            raise ValueError("El máximo y el mínimo no pueden ser iguales.")
        if valor <= minimo:
            return 0.0
        elif valor >= maximo:
            return 1.0
        else:
            return (valor - minimo) / (maximo - minimo)

    def desnormalizar_0_1(valor_normalizado, minimo, maximo):
        return valor_normalizado * (maximo - minimo) + minimo

    def determinar_A(esel, x, limites, B):
        Dloc = len(x)
        dims_restantes = [i for i in range(Dloc) if i != esel]
        resultados = []
        for dim in dims_restantes:
            valor = x[dim]
            bordes = limites[dim]
            if valor <= bordes[0]:
                bin_id = 0
            elif valor >= bordes[-1]:
                bin_id = B - 1
            else:
                bin_id = None
                for b in range(B):
                    if bordes[b] <= valor < bordes[b+1]:
                        bin_id = b
                        break
                if bin_id is None:
                    diffs = np.abs(bordes - valor)
                    idx_cercano = int(np.argmin(diffs))
                    bin_id = min(max(idx_cercano, 0), B - 1)
            resultados.append(bin_id)
        return vector_a_decimal(resultados, B)

    # ============================================================
    # 5) get_caja_stats / itera_mapa / get_A_para_caja (closures)
    # ============================================================
    def get_caja_stats(esel, cualA):
        esel = int(esel)
        cualA = int(cualA)

        info = comb_info_por_esel[esel]
        keys = np.asarray(info["keys"], dtype=object)
        idx_start = info["idx_start"]
        counts = info["counts"]
        order = info["order"]
        digits_keys = info["digits_keys"]

        usando_vecino = False
        dist_nn = 0.0

        if keys.size == 0:
            vals = data[:, esel]
        else:
            pos = np.searchsorted(keys, cualA)
            if pos < keys.size and int(keys[pos]) == cualA:
                s = idx_start[pos]; c = counts[pos]
                rows = order[s:s + c]
                vals = data[rows, esel]
                usando_vecino = False
            else:
                usando_vecino = True
                if D == 1:
                    pos_nn = 0; dist_nn = 0.0
                else:
                    Mloc = D - 1
                    digits = np.zeros((Mloc,), dtype=int)
                    tmp = cualA
                    for j in range(Mloc - 1, -1, -1):
                        digits[j] = tmp % B
                        tmp //= B
                    diffs = np.abs(digits_keys - digits).sum(axis=1)
                    pos_nn = int(np.argmin(diffs))
                    dist_nn = float(diffs[pos_nn])

                s = idx_start[pos_nn]; c = counts[pos_nn]
                rows = order[s:s + c]
                vals = data[rows, esel]

        if vals.size == 0:
            vals = data[:, esel]
            usando_vecino = True
            dist_nn = 0.0

        minimo = float(vals.min())
        maximo = float(vals.max())
        span = maximo - minimo

        if expand_if_tie and (span == 0.0 or abs(span) < span_umbral):
            centro = 0.5 * (maximo + minimo)
            delta = span_umbral
            minimo = centro - delta
            maximo = centro + 3.0 * delta

        if maximo <= minimo:
            maximo = minimo + eps_abs

        edges = np.linspace(minimo, maximo, B_finos + 1)
        hist, _ = np.histogram(vals, bins=edges)
        hist = hist.astype(float)

        if usando_vecino:
            hist_marg, _ = np.histogram(data[:, esel], bins=edges)
            hist_marg = hist_marg.astype(float)
            if hist_marg.sum() <= 0:
                hist_marg = np.ones_like(hist_marg)
            alpha = 1.0 / (1.0 + dist_nn)
            hist = alpha * hist + (1.0 - alpha) * hist_marg

        # “relleno gaussiano” / piso (tal cual)
        num_bins = len(hist)
        ocupados = hist > 0
        k = int(ocupados.sum())

        if k == 0:
            hist = np.ones_like(hist)
        else:
            if k < num_bins / 2.0:
                idx_peak = int(np.argmax(hist))
                xs = np.arange(num_bins)
                sigma = max(1.0, num_bins / 6.0)
                gauss = np.exp(-0.5 * ((xs - idx_peak) / sigma) ** 2)
                gauss[gauss <= 0] = 1e-6
                amp = max(hist.max(), 3.0)
                gauss = amp * gauss / gauss.max()
                hist = gauss
            else:
                hist = hist + 1.0

        if hist.sum() <= 0:
            hist = np.ones_like(hist)

        return float(minimo), float(maximo), edges.astype(float), hist.astype(float), usando_vecino

    def get_A_para_caja(esel, cualA):
        _, _, _, hist_local, _ = get_caja_stats(esel, cualA)
        if hist_local.sum() <= 0:
            hist_local = np.ones_like(hist_local)
        return matriz_A(hist_local)

    def itera_mapa(esel, x_actual_norm, cualA, x_actual_denorm):
        esel = int(esel)
        cualA = int(cualA)

        minimo, maximo, limites_pre, _, _ = get_caja_stats(esel, cualA)
        A_local = get_A_para_caja(esel, cualA)
        B_loc = A_local.shape[0]

        e_min = limites_pre.min()
        e_max = limites_pre.max()
        if e_max == e_min:
            raise ValueError("limites finos degenerados en esa caja")

        limitesa = (limites_pre - e_min) / (e_max - e_min)
        x_nw = 0.0

        if not np.any(A_local != 0):
            x_nw = 0.0
        else:
            Lim0 = 0.0
            Lim1 = 0.0
            x0 = 0.0
            y0 = 0.0
            x1 = 0.0
            y1 = 0.0

            for c in range(1, B_loc + 1):
                Lim0 = limitesa[c - 1]
                Lim1 = limitesa[c - 1]
                x0 = limitesa[c - 1]
                y0 = 0.0
                x1 = limitesa[c - 1]
                y1 = 0.0

                for f in range(1, B_loc + 1):
                    w = (limitesa[c] - limitesa[c - 1])
                    Lim1 += A_local[f - 1, c - 1] * w
                    y1 = limitesa[f]
                    x1 += A_local[f - 1, c - 1] * w

                    if Lim0 <= x_actual_norm < Lim1:
                        m = (y1 - y0) / (x1 - x0)
                        x_nw = m * (x_actual_norm - x0) + y0

                    x0 = x1
                    y0 = y1
                    Lim0 = Lim1

        x_nw_denorm = desnormalizar_0_1(x_nw, minimo, maximo)
        return float(x_nw), float(x_nw_denorm)

    # ============================================================
    # 6) SINGLE-SEED
    # ============================================================
    x_norm = np.zeros(D, dtype=float)
    x = data[single_seed_index, :].astype(float).copy()
    xx_no_norm = [x.copy()]

    # inicialización x0i por dimensión en la primera iteración (igual que vos)
    x0i = None

    for ii in range(N_iter):
        x_prev = x.copy()

        for esel in range(D):
            cualA = determinar_A(esel, x, limites, B)

            if ii == 0:
                minimo, maximo, _, _, _ = get_caja_stats(esel, cualA)
                x0i = normalizar_0_1(x[esel], minimo, maximo)

            x_norm[esel], x[esel] = itera_mapa(esel, x0i, cualA, x[esel])
            x0i = x_norm[esel]

        xx_no_norm.append(x.copy())

        if np.array_equal(x, x_prev):
            break

    XX = np.array(xx_no_norm, dtype=float)

    # ============================================================
    # 7) MULTI-SEED (rounds)
    # ============================================================
    SAL2 = np.zeros((num_rounds + 1, N, D), dtype=float)
    SAL2[0] = data.copy()

    x_norm_ms = np.zeros((N, D), dtype=float)

    # init x_norm desde data
    for i in range(N):
        x0 = SAL2[0, i]
        for esel in range(D):
            cualA = determinar_A(esel, x0, limites, B)
            minimo, maximo, _, _, _ = get_caja_stats(esel, cualA)
            x_norm_ms[i, esel] = normalizar_0_1(x0[esel], minimo, maximo)

    for k in range(1, num_rounds + 1):
        X_prev = SAL2[k - 1].copy()
        X_new = X_prev.copy()

        for i in range(N):
            x0 = X_prev[i].copy()
            for esel in range(D):
                cualA = determinar_A(esel, x0, limites, B)
                x_norm_ms[i, esel], x0[esel] = itera_mapa(esel, x_norm_ms[i, esel], cualA, x0[esel])
            X_new[i] = x0

        SAL2[k] = X_new

    return {
        "limites": limites,
        "best_obj": best_obj,
        "comb_info_por_esel": comb_info_por_esel,
        "XX": XX,              # single-seed trajectory
        "SAL2": SAL2,          # multi-seed rounds
        "B": B,
        "B_finos": B_finos,
        "N": N,
        "D": D,
        "single_seed_index": single_seed_index,
        "N_iter": N_iter,
        "num_rounds": num_rounds,
        "use_simulated_annealing": use_simulated_annealing,
    }
