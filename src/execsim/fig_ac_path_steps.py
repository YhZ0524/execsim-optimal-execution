import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from .data import generate_profile, ac_schedule
from .eval import grid_search
from .policy import ParamPolicy
from .sim import Simulator

def build_step_threshold(t, cum, block):
    xs=[t[0]]; ys=[0.0]; last=0.0
    for i in range(1, len(t)):
        if cum[i]-last >= block or i==len(t)-1:
            xs += [t[i], t[i]]
            ys += [last, float(cum[i])]
            last = float(cum[i])
    return np.array(xs), np.array(ys)

def main(seed=2025, n_steps=300, q_total=10000, blocks=12):
    profile, dt = generate_profile(n_steps=n_steps, dt=1.0, mid0=100.0, ann_vol=0.25, tick=0.01, spread_ticks=1.0, kappa=4.0, lam_low=0.2, lam_high=0.7, seed=seed)
    sched_ac = ac_schedule(q_total, n_steps, eta=3.5)
    base_grid = np.linspace(profile["half_spread"].iloc[0]*0.5, profile["half_spread"].iloc[0]*2.0, 4)
    alpha_grid = np.linspace(0.0, 0.02, 4)
    band_grid = [50,100,200]
    mf_grid = [0.5,0.75,1.0]
    lmax_grid = [50,100,200]
    tune = grid_search(profile, dt, sched_ac, q_total, base_grid, alpha_grid, band_grid, mf_grid, lmax_grid, n_mc=64, seed=seed)
    best = tune.iloc[0].to_dict()
    pol = ParamPolicy(base_depth=float(best["base_depth"]), alpha_depth=float(best["alpha_depth"]), band=int(best["band"]), market_frac=float(best["market_frac"]), lmax=int(best["lmax"]))
    sim = Simulator(profile, dt, seed=seed+321)
    out = sim.run(pol, q_total=q_total, schedule=sched_ac, force_finish=True)
    t = profile["t"].values
    cum_all = (out["fills_lim"] + out["fills_mkt"]).cumsum()
    trig = np.where(out["fills_mkt"]>0)[0]
    block = max(1, int(q_total/blocks))
    xs, ys = build_step_threshold(t, cum_all, block)
    os.makedirs("results", exist_ok=True)
    fig = plt.figure()
    plt.plot(t, sched_ac, linestyle="--", label="Almgren–Chriss target")
    plt.plot(xs, ys, drawstyle="steps-post", label="Optimal execution (step process)")
    if len(trig)>0:
        plt.scatter(t[trig], cum_all[trig], s=28, color="red", label="Market-order triggers")
    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative inventory")
    plt.legend()
    fig.savefig("results/ac_path_steps.png", dpi=160, bbox_inches="tight")

if __name__=="__main__":
    main()
