import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from .data import generate_profile, twap_schedule, ac_schedule
from .policy import ParamPolicy
from .benchmarks import twap_market_policy, ac_market_policy
from .sim import Simulator
from .eval import grid_search

def eval_policy(profile, dt, schedule, q_total, policy, seeds):
    rows=[]
    for s in seeds:
        sim=Simulator(profile, dt, seed=int(s))
        out=sim.run(policy, q_total=q_total, schedule=schedule, force_finish=True)
        cum=(out["fills_lim"]+out["fills_mkt"]).cumsum()
        err=cum-schedule
        rmse=float(np.sqrt(np.mean(err**2)))
        devmax=float(np.max(np.abs(err)))
        lim_share=float(out["fills_lim"].sum()/q_total*100.0)
        rows.append({"shortfall":float(out["shortfall"]), "avg_price":float(out["avg_price"]), "lim_share":lim_share, "dev_rmse":rmse, "dev_max":devmax})
    return pd.DataFrame(rows)

def main(seed=2025, n_steps=300, q_total=10000, n_tune=32, n_eval=200):
    profile, dt = generate_profile(n_steps=n_steps, dt=1.0, mid0=100.0, ann_vol=0.25, tick=0.01, spread_ticks=1.0, kappa=4.0, lam_low=0.2, lam_high=0.7, seed=seed)
    sched_twap = twap_schedule(q_total, n_steps)
    sched_ac = ac_schedule(q_total, n_steps, eta=3.5)
    base_grid = np.linspace(profile["half_spread"].iloc[0]*0.5, profile["half_spread"].iloc[0]*2.0, 4)
    alpha_grid = np.linspace(0.0, 0.02, 4)
    band_grid = [50,100,200]
    mf_grid = [0.5,0.75,1.0]
    lmax_grid = [50,100,200]
    tune = grid_search(profile, dt, sched_twap, q_total, base_grid, alpha_grid, band_grid, mf_grid, lmax_grid, n_mc=n_tune, seed=seed)
    best = tune.iloc[0].to_dict()
    pol = ParamPolicy(base_depth=float(best["base_depth"]), alpha_depth=float(best["alpha_depth"]), band=int(best["band"]), market_frac=float(best["market_frac"]), lmax=int(best["lmax"]))
    twap_pol = twap_market_policy(q_total, n_steps)
    ac_pol = ac_market_policy(sched_ac)
    seeds = np.arange(n_eval)+seed+1000
    df_pol = eval_policy(profile, dt, sched_twap, q_total, pol, seeds)
    df_twap = eval_policy(profile, dt, sched_twap, q_total, twap_pol, seeds)
    df_ac = eval_policy(profile, dt, sched_ac, q_total, ac_pol, seeds)
    df_pol["policy"]="ParamPolicy"; df_twap["policy"]="TWAP_Market"; df_ac["policy"]="AC_Market"
    comp = pd.concat([df_pol, df_twap, df_ac], ignore_index=True)
    os.makedirs("results", exist_ok=True)
    comp.to_csv("results/compare.csv", index=False)
    fig = plt.figure()
    data=[comp[comp.policy=="ParamPolicy"]["shortfall"], comp[comp.policy=="TWAP_Market"]["shortfall"], comp[comp.policy=="AC_Market"]["shortfall"]]
    plt.boxplot(data, labels=["Policy","TWAP","AC"])
    plt.ylabel("Implementation shortfall")
    fig.savefig("results/compare_box.png", dpi=160, bbox_inches="tight")
    means=comp.groupby("policy")["shortfall"].mean().reindex(["ParamPolicy","TWAP_Market","AC_Market"])
    q25=comp.groupby("policy")["shortfall"].quantile(0.25).reindex(["ParamPolicy","TWAP_Market","AC_Market"])
    q75=comp.groupby("policy")["shortfall"].quantile(0.75).reindex(["ParamPolicy","TWAP_Market","AC_Market"])
    x=np.arange(3)
    fig=plt.figure()
    plt.bar(x, means.values)
    for i,(l,u,m) in enumerate(zip(q25.values,q75.values,means.values)):
        plt.plot([i,i],[l,u])
    plt.xticks(x, ["Policy","TWAP","AC"])
    plt.ylabel("IS mean with P25–P75")
    fig.savefig("results/compare_bar.png", dpi=160, bbox_inches="tight")
    fig=plt.figure()
    ls=comp.groupby("policy")["lim_share"].mean().reindex(["ParamPolicy","TWAP_Market","AC_Market"])
    plt.bar(x, ls.values)
    plt.xticks(x, ["Policy","TWAP","AC"])
    plt.ylabel("Limit-order share (%)")
    fig.savefig("results/limit_share_bar.png", dpi=160, bbox_inches="tight")

if __name__=="__main__":
    main()
