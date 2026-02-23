#%%
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors


def _log_beta(a: int, b: int) -> float:
    # log Beta(a,b) = lgamma(a)+lgamma(b)-lgamma(a+b)
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _log_comb(n: int, k: int) -> float:
    # log nCk
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _fexpected_table(N: int, M: int, k: int) -> np.ndarray:
    """
    Build fexpected(m) for m=0..M where m is the number of 'good' synthetic samples.
    Uses Lemma-1 formula (paper eq. 9) with M replaced by m.
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    if N <= k:
        raise ValueError(f"Need N > k for Beta(k, N-k). Got N={N}, k={k}.")

    log_beta_den = _log_beta(k, N - k)
    f = np.zeros(M + 1, dtype=np.float64)

    # m=0 => no synthetic samples => coverage 0
    f[0] = 0.0

    for m in range(1, M + 1):
        s = 0.0
        for j in range(1, m + 1):
            # term = min(j/k,1) * C(m,j) * Beta(k+j, m-j + N-k) / Beta(k, N-k)
            weight = min(j / k, 1.0)

            log_term = (
                _log_comb(m, j)
                + _log_beta(k + j, (m - j) + (N - k))
                - log_beta_den
            )
            s += weight * math.exp(log_term)
        f[m] = s
    return f


def _calibrate_g(unnorm_score: float, f_table: np.ndarray) -> float:
    """
    Paper: reverse expected curve values into a sorted list; find insertion index i(s);
    g(s)=1 - i(s)/M. (paper Sec.4.2.2)
    """
    M = len(f_table) - 1
    # descending list: [f(M), f(M-1), ..., f(0)]
    f_desc = f_table[::-1]

    # searchsorted works on increasing arrays -> use -f_desc to make it increasing
    idx = np.searchsorted(-f_desc, -unnorm_score, side="right")
    idx = int(np.clip(idx, 0, M))
    return float(np.clip(1.0 - idx / M, 0.0, 1.0))


def clipped_coverage(real: np.ndarray, fake: np.ndarray, k: int = 5, calibrate: bool = True) -> float:
    """
    real: (N,d), fake: (M,d)
    Returns ClippedCoverage (calibrated) by default; set calibrate=False to return unnorm.
    """
    real = np.asarray(real, dtype=np.float64)
    fake = np.asarray(fake, dtype=np.float64)

    N = real.shape[0]
    M = fake.shape[0]
    if N == 0 or M == 0:
        return 0.0
    if k <= 0:
        raise ValueError("k must be positive.")
    if N <= k:
        raise ValueError(f"Need N > k. Got N={N}, k={k}.")

    # 1) real k-NN radius: NNDr_k(x_i^r) (exclude self by using k+1 neighbors)
    nn_r = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(real)
    dist_r, _ = nn_r.kneighbors(real, return_distance=True)
    radius = dist_r[:, k]  # kth neighbor distance (0th is itself)

    # 2) Count how many of the k-closest fake samples are within each real ball
    #    (equivalent to counting all, because we cap at k anyway)
    nn_f = NearestNeighbors(n_neighbors=min(k, M), algorithm="auto").fit(fake)
    dist_rf, _ = nn_f.kneighbors(real, return_distance=True)

    within = (dist_rf <= radius[:, None])
    count = within.sum(axis=1)

    # eq (8): mean_i min(count_i/k, 1)
    unnorm = np.minimum(count / k, 1.0).mean()

    if not calibrate:
        return float(unnorm)

    # 3) calibration g using Lemma-1 expected curve table
    f_table = _fexpected_table(N=N, M=M, k=k)
    return _calibrate_g(unnorm_score=float(unnorm), f_table=f_table)

#%%
""" Examples """
if __name__== "__main__":
    num_real_samples = num_fake_samples = 10000
    feature_dim = 1000
    nearest_k = 5
    real_features = np.random.normal(loc=0.0, scale=1.0,
                                    size=[num_real_samples, feature_dim])

    fake_features = np.random.normal(loc=0.0, scale=1.0,
                                    size=[num_fake_samples, feature_dim])

    metrics = clipped_coverage(real=real_features,
                        fake=fake_features,
                        k=nearest_k)
    
    print(metrics)