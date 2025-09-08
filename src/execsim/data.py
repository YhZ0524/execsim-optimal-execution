import numpy as np
import pandas as pd

def u_shape(n, low, high):
    x = np.linspace(0, 1, n)
    y = low + (high - low) * (4 * (x - 0.5) ** 2)
    return y

def generate_profile(n_steps=300, dt=1.0, mid0=100.0, ann_vol=0.2, tick=0.01, spread_ticks=1.0, kappa=4.0, lam_low=0.15, lam_high=0.6, seed=42):
    rng = np.random.default_rng(seed)
    vol_per_step = ann_vol / np.sqrt(252 * 6.5 * 3600 / dt)
    half_spread = spread_ticks * tick / 2.0
    lam = u_shape(n_steps, lam_low, lam_high)
    mid = np.empty(n_steps)
    mid[0] = mid0
    for t in range(1, n_steps):
        z = rng.standard_normal()
        mid[t] = mid[t - 1] * np.exp(-0.5 * (vol_per_step ** 2) + vol_per_step * z)
    kappa_series = np.full(n_steps, kappa)
    df = pd.DataFrame({"t": np.arange(n_steps) * dt, "mid": mid, "half_spread": np.full(n_steps, half_spread), "lambda_mo": lam, "kappa": kappa_series})
    return df, dt

def twap_schedule(q_total, n_steps):
    per = q_total / n_steps
    cum = np.cumsum(np.full(n_steps, per))
    return np.clip(cum, 0, q_total)

def ac_schedule(q_total, n_steps, eta=3.0):
    x = np.linspace(0, 1, n_steps)
    w = np.exp(eta * (1 - x))
    w = w / w.sum()
    cum = np.cumsum(q_total * w)
    return np.clip(cum, 0, q_total)
