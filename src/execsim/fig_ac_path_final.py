import numpy as np, matplotlib.pyplot as plt, os
from .data import generate_profile, ac_schedule
from .eval import grid_search
from .policy import ParamPolicy
from .sim import Simulator

def compress_runs(idx):
    if len(idx)==0: return idx
    keep=[idx[0]]
    for i in range(1,len(idx)):
        if idx[i]!=idx[i-1]+1:
            keep.append(idx[i])
    return np.array(keep)

def build_step_from_triggers(t, cum, trig_idx, force_terminal=True):
    xs=[t[0]]; ys=[0.0]; last=0.0
    for i in trig_idx:
        xs += [t[i], t[i]]
        ys += [last, float(cum[i])]
        last=float(cum[i])
    if force_terminal and last!=float(cum[-1]):
        xs += [t[-1], t[-1]]
        ys += [last, float(cum[-1])]
        last=float(cum[-1])
    if xs[-1]!=t[-1]:
        xs.append(t[-1]); ys.append(last)
    return np.array(xs), np.array(ys)

def main(seed=2025, n_steps=300, q_total=10000, y_scale="kshares"):
    profile, dt = generate_profile(n_steps=n_steps, dt=1.0, mid0=100.0, ann_vol=0.25, tick=0.01, spread_ticks=1.0, kappa=4.0, lam_low=0.2, lam_high=0.7, seed=seed)
    sched = ac_schedule(q_total, n_steps, eta=3.5)
    base_grid = np.linspace(profile["half_spread"].iloc[0]*0.5, profile["half_spread"].iloc[0]*2.0, 4)
    alpha_grid = np.linspace(0.0, 0.02, 4)
    band_grid  = [50,100,200]
    mf_grid    = [0.5,0.75,1.0]
    lmax_grid  = [50,100,200]
    tune = grid_search(profile, dt, sched, q_total, base_grid, alpha_grid, band_grid, mf_grid, lmax_grid, n_mc=64, seed=seed)
    best = tune.iloc[0].to_dict()
    band = int(best["band"])
    pol = ParamPolicy(base_depth=float(best["base_depth"]), alpha_depth=float(best["alpha_depth"]), band=band, market_frac=float(best["market_frac"]), lmax=int(best["lmax"]))
    sim = Simulator(profile, dt, seed=seed+321)
    out = sim.run(pol, q_total=q_total, schedule=sched, force_finish=True)
    t    = profile["t"].values
    cum  = (out["fills_lim"] + out["fills_mkt"]).cumsum()
    mkt  = out["fills_mkt"]
    trig_mask = np.zeros_like(mkt, dtype=bool)
    for i in range(1, len(t)):
        deficit_prev = sched[i] - cum[i-1]
        trig_mask[i] = (deficit_prev > band) and (mkt[i] > 0)
    if cum[-1] < q_total or mkt[-1] > 0:
        trig_mask[-1] = True
    trig_idx = compress_runs(np.where(trig_mask)[0])
    xs, ys = build_step_from_triggers(t, cum, trig_idx, force_terminal=True)
    boundary = np.maximum(0, sched - band)
    if y_scale=="kshares":
        unit=1000.0; sched_y=sched/unit; boundary_y=boundary/unit; ys_y=ys/unit; ylab="Cumulative inventory (k shares)"
    elif y_scale=="lots":
        unit=q_total/10.0; sched_y=sched/unit; boundary_y=boundary/unit; ys_y=ys/unit; ylab="Cumulative inventory (×10 of total)"
    else:
        unit=1.0; sched_y=sched; boundary_y=boundary; ys_y=ys; ylab="Cumulative inventory"
    os.makedirs("results", exist_ok=True)
    fig = plt.figure()
    plt.plot(t, sched_y, linestyle="--", label="Almgren–Chriss target")
    plt.plot(xs, ys_y, drawstyle="steps-post", label="Optimal execution (step)")
    plt.plot(t, boundary_y, linestyle=":", label="Trigger boundary = target − band")
    if len(trig_idx)>0:
        plt.scatter(t[trig_idx], boundary_y[trig_idx], s=28, color="red", label="Boundary hits")
    plt.xlabel("Time (s)"); plt.ylabel(ylab); plt.legend()
    fig.savefig("results/ac_path_final.png", dpi=160, bbox_inches="tight")

if __name__=="__main__":
    main()
