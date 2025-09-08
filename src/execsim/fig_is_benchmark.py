import numpy as np, matplotlib.pyplot as plt, os
from .mm_data import midprice_paths, ac_schedule, twap_avg_price_series, algo_step_from_band

def main(seed=11):
    n_steps=300
    s = midprice_paths(n_paths=1, n_steps=n_steps, dt=1.0, mid0=100.0, ann_vol=0.18, seed=seed)[0]
    sch = ac_schedule(N_total=10000, n_steps=n_steps, T=300.0, chi=0.03)
    Y = 0.01
    twap_avg = twap_avg_price_series(s, sch, Y)
    band = 800
    lmax = 1200
    q_step, mkt = algo_step_from_band(sch, band=band, lmax=lmax)
    amt = np.zeros(n_steps); amt[mkt>0] = mkt[mkt>0]
    trade_px = s + Y
    money = np.cumsum(amt * trade_px)
    with np.errstate(divide='ignore', invalid='ignore'):
        algo_avg = np.where(q_step>0, money / q_step, 0.0)
    os.makedirs("results", exist_ok=True)
    fig = plt.figure(figsize=(8,3.4))
    plt.step(np.arange(n_steps), algo_avg, where="post", linewidth=2.0, label="Price Per Share")
    plt.plot(np.arange(n_steps), twap_avg, linestyle="--", linewidth=2.0, label="TWAP")
    plt.xlabel("Time (s)")
    plt.ylabel("Price")
    plt.legend(loc="upper right")
    fig.text(0.02, -0.02, "IS benchmark", fontsize=12)
    fig.savefig("results/mm_is_benchmark.png", dpi=160, bbox_inches="tight")

if __name__=="__main__":
    main()
