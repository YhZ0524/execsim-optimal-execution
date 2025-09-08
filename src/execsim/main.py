import numpy as np
import pandas as pd
from .data import generate_profile, twap_schedule, ac_schedule
from .policy import ParamPolicy
from .benchmarks import twap_market_policy, ac_market_policy
from .sim import Simulator
from .eval import grid_search
from .plotting import plot_price_and_inventory
import os

def run_all(seed=2025):
    n_steps = 300
    q_total = 10000
    profile, dt = generate_profile(n_steps=n_steps, dt=1.0, mid0=100.0, ann_vol=0.25, tick=0.01, spread_ticks=1.0, kappa=4.0, lam_low=0.2, lam_high=0.7, seed=seed)
    schedule_twap = twap_schedule(q_total, n_steps)
    schedule_ac = ac_schedule(q_total, n_steps, eta=3.5)
    base_grid = np.linspace(profile["half_spread"].iloc[0] * 0.5, profile["half_spread"].iloc[0] * 2.0, 4)
    alpha_grid = np.linspace(0.0, 0.02, 4)
    band_grid = [50, 100, 200]
    mf_grid = [0.5, 0.75, 1.0]
    lmax_grid = [50, 100, 200]
    df = grid_search(profile, dt, schedule_twap, q_total, base_grid, alpha_grid, band_grid, mf_grid, lmax_grid, n_mc=64, seed=seed)
    best = df.iloc[0].to_dict()
    pol = ParamPolicy(base_depth=float(best["base_depth"]), alpha_depth=float(best["alpha_depth"]), band=int(best["band"]), market_frac=float(best["market_frac"]), lmax=int(best["lmax"]))
    sim = Simulator(profile, dt, seed=seed + 999)
    out = sim.run(pol, q_total=q_total, schedule=schedule_twap, force_finish=True)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/grid_search.csv", index=False)
    plot_price_and_inventory(profile, schedule_twap, out, "results/plots.png")
    sim_twap = Simulator(profile, dt, seed=seed + 7)
    pol_twap = twap_market_policy(q_total, n_steps)
    out_twap = sim_twap.run(pol_twap, q_total=q_total, schedule=schedule_twap, force_finish=True)
    sim_ac = Simulator(profile, dt, seed=seed + 8)
    pol_ac = ac_market_policy(schedule_ac)
    out_ac = sim_ac.run(pol_ac, q_total=q_total, schedule=schedule_ac, force_finish=True)
    rows = []
    rows.append({"policy": "ParamPolicy", "avg_price": out["avg_price"], "shortfall": out["shortfall"], "lim_shares": int(out["fills_lim"].sum()), "mkt_shares": int(out["fills_mkt"].sum())})
    rows.append({"policy": "TWAP_Market", "avg_price": out_twap["avg_price"], "shortfall": out_twap["shortfall"], "lim_shares": int(out_twap["fills_lim"].sum()), "mkt_shares": int(out_twap["fills_mkt"].sum())})
    rows.append({"policy": "AC_Market", "avg_price": out_ac["avg_price"], "shortfall": out_ac["shortfall"], "lim_shares": int(out_ac["fills_lim"].sum()), "mkt_shares": int(out_ac["fills_mkt"].sum())})
    pd.DataFrame(rows).to_csv("results/summary.csv", index=False)

if __name__ == "__main__":
    run_all()
