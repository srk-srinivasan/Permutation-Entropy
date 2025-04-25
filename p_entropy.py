import numpy as np
import math
import itertools

def s_entropy(p):
    """Shannon entropy (base 2) for a probability distribution p."""
    p = np.asarray(p)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def ordinal_patterns(ts, embdim, embdelay):
    """
    Computes normalized frequency of all ordinal patterns (including zeros).
    Based on Bandt-Pompe method.
    """
    ts = np.asarray(ts)
    m, t = embdim, embdelay
    n = len(ts)
    n_patterns = math.factorial(m)

    # All possible patterns in lexicographic order
    patterns = list(itertools.permutations(range(m)))
    pattern_index = {pat: idx for idx, pat in enumerate(patterns)}
    counts = np.zeros(n_patterns)

    for i in range(n - (m - 1) * t):
        window = ts[i:i + t * m:t]
        order = tuple(np.argsort(window))
        counts[pattern_index[order]] += 1

    total = np.sum(counts)
    return counts / total if total > 0 else counts

def permutation_entropy(ts, embdim, embdelay):
    """
    Returns normalized permutation entropy (log base 2), using all possible patterns.
    Matches Bandt-Pompe + Rosso et al. (2007).
    """
    p = ordinal_patterns(ts, embdim, embdelay)
    max_entropy = np.log2(len(p)) if len(p) > 0 else 0
    return s_entropy(p) / max_entropy if max_entropy > 0 else 0.0

def complexity(ts, embdim, embdelay):
    """
    Computes statistical complexity (Rosso et al.):
    - Q_J: Jensen-Shannon divergence between observed and uniform over ALL patterns
    - H_S: Normalized permutation entropy
    - C_JS = Q_J * H_S
    """
    p = ordinal_patterns(ts, embdim, embdelay)
    pe = permutation_entropy(ts, embdim, embdelay)

    n = len(p)
    uniform = np.ones(n) / n
    avg = 0.5 * (p + uniform)

    js_div = s_entropy(avg) - 0.5 * s_entropy(p) - 0.5 * s_entropy(uniform)
    normalized_js = js_div / np.log2(n) if np.log2(n) > 0 else 0

    return normalized_js * pe
