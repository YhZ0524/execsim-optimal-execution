import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_price_and_inventory(profile_df, schedule, out, path_png):
    t = profile_df["t"].values
    mid = profile_df["mid"].values
    cum_lim = out["fills_lim"].cumsum()
    cum_mkt = out["fills_mkt"].cumsum()
    cum_all = cum_lim + cum_mkt
    fig = plt.figure()
    plt.plot(t, mid, label="mid")
    plt.xlabel("t")
    plt.ylabel("price")
    fig.savefig(path_png.replace(".png", "_price.png"), dpi=160, bbox_inches="tight")
    fig = plt.figure()
    plt.plot(t, schedule, label="target")
    plt.plot(t, cum_all, label="filled")
    plt.xlabel("t")
    plt.ylabel("shares")
    fig.savefig(path_png.replace(".png", "_inv.png"), dpi=160, bbox_inches="tight")
    fig = plt.figure()
    plt.plot(t, out["depths"], label="depth")
    plt.xlabel("t")
    plt.ylabel("depth")
    fig.savefig(path_png.replace(".png", "_depth.png"), dpi=160, bbox_inches="tight")
