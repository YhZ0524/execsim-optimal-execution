import numpy as np, matplotlib.pyplot as plt, os
from .mm_data import midprice_paths, ac_schedule, twap_avg_price_series

def main(seed=7):
    n_steps = 300
    s = midprice_paths(n_paths=3, n_steps=n_steps, dt=1.0, mid0=100.0, ann_vol=0.20, seed=seed)
    sch = ac_schedule(N_total=10000, n_steps=n_steps, T=300.0, chi=0.03)
    Y = 0.01
    twap_curves = []
    for k in range(3):
        twap_curves.append(twap_avg_price_series(s[k], sch, Y))
    os.makedirs("results", exist_ok=True)
    fig = plt.figure(figsize=(8,3.2))
    ax1 = plt.subplot(1,2,1)
    for k in range(3):
        ax1.plot(np.arange(n_steps), s[k])
    ax1.set_title("MidPrice")
    ax2 = plt.subplot(1,2,2)
    for k in range(3):
        ax2.plot(np.arange(n_steps), twap_curves[k])
    ax2.set_title("TWAP")
    fig.tight_layout()
    fig.savefig("results/mm_midprice_twap.png", dpi=160, bbox_inches="tight")

if __name__=="__main__":
    main()
