import numpy as np

def midprice_paths(n_paths=3, n_steps=300, dt=1.0, mid0=100.0, ann_vol=0.20, seed=123):
    rng = np.random.default_rng(seed)
    secs_per_year = 252.0 * 14400.0
    sigma = ann_vol / np.sqrt(secs_per_year)
    s = np.empty((n_paths, n_steps), dtype=float)
    s[:, 0] = mid0
    for t in range(1, n_steps):
        z = rng.standard_normal(n_paths)
        s[:, t] = s[:, t-1] + mid0 * sigma * np.sqrt(dt) * z
    return s

def ac_schedule(N_total=10000, n_steps=300, T=300.0, chi=0.03):
    t = np.linspace(0.0, T, n_steps)
    num = np.sinh(chi * (T - t))
    den = np.sinh(chi * T)
    inv = N_total * (1.0 - num / den)
    inv[0] = max(inv[0], 0.0)
    return inv

def twap_avg_price_series(mid, schedule, Y=0.01):
    n_steps = mid.shape[-1]
    amt = np.empty(n_steps)
    amt[0] = schedule[0]
    amt[1:] = np.diff(schedule)
    money = np.cumsum(amt * (mid + Y))
    with np.errstate(divide='ignore', invalid='ignore'):
        avg = np.where(schedule > 0, money / schedule, money * 0.0)
    return avg

def algo_step_from_band(schedule, band=500, lmax=1200):
    n_steps = schedule.shape[0]
    q = 0.0
    q_step = np.zeros(n_steps)
    mkt_size = np.zeros(n_steps, dtype=float)
    for t in range(n_steps):
        deficit = schedule[t] - q
        if deficit > band:
            size = min(lmax, deficit)
            q += size
            mkt_size[t] = size
        q_step[t] = q
    return q_step, mkt_size
