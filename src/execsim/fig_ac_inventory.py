import numpy as np, matplotlib.pyplot as plt, os
from .mm_data import ac_schedule, algo_step_from_band

def main(seed=2025):
    n_steps=300
    sch = ac_schedule(N_total=10000, n_steps=n_steps, T=300.0, chi=0.03)
    band = 800
    lmax = 1200
    q_step, mkt = algo_step_from_band(sch, band=band, lmax=lmax)
    t = np.arange(n_steps)
    boundary = np.maximum(0.0, sch - band)
    hit_idx = np.where(mkt>0)[0]
    os.makedirs("results", exist_ok=True)
    fig = plt.figure(figsize=(8,3.8))
    plt.step(t, q_step/1000.0, where="post", color=(0.05,0.25,0.95), linewidth=2.0, label="Algo")
    plt.plot(t, sch/1000.0, linestyle="--", color=(0.55,0.6,0.7), linewidth=2.0, label="AC Targets")
    if hit_idx.size>0:
        plt.scatter(t[hit_idx], boundary[hit_idx]/1000.0, s=28, color="red", label="Trigger Boundary")
    plt.xlabel("Time(Sec.)")
    plt.ylabel("Inventory")
    plt.legend(loc="center right")
    fig.savefig("results/mm_ac_inventory.png", dpi=160, bbox_inches="tight")

if __name__=="__main__":
    main()
