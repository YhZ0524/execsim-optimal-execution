import numpy as np
import pandas as pd
from .sim import Simulator
from .policy import ParamPolicy

def grid_search(profile_df, dt, schedule, q_total, base_depth_grid, alpha_depth_grid, band_grid, market_frac_grid, lmax_grid, n_mc=64, seed=123):
    rows = []
    seeds = np.arange(n_mc) + seed
    for base in base_depth_grid:
        for alpha in alpha_depth_grid:
            for band in band_grid:
                for mf in market_frac_grid:
                    for lmax in lmax_grid:
                        shorts = []
                        lim_shares = []
                        mkt_shares = []
                        for s in seeds:
                            sim = Simulator(profile_df, dt, seed=int(s))
                            pol = ParamPolicy(base_depth=base, alpha_depth=alpha, band=band, market_frac=mf, lmax=lmax)
                            out = sim.run(pol, q_total=q_total, schedule=schedule, force_finish=True)
                            shorts.append(out["shortfall"])
                            lim_shares.append(out["fills_lim"].sum())
                            mkt_shares.append(out["fills_mkt"].sum())
                        row = {"base_depth": base, "alpha_depth": alpha, "band": band, "market_frac": mf, "lmax": lmax, "mean_shortfall": float(np.mean(shorts)), "p25_shortfall": float(np.quantile(shorts, 0.25)), "p75_shortfall": float(np.quantile(shorts, 0.75)), "lim_share_mean": float(np.mean(lim_shares)), "mkt_share_mean": float(np.mean(mkt_shares))}
                        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_shortfall")
