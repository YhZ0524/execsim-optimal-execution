import numpy as np
from .policy import TWAPMarketOnly, FrontLoadedMarketOnly

def twap_market_policy(q_total, n_steps):
    slice_size = int(np.ceil(q_total / n_steps))
    return TWAPMarketOnly(slice_size=slice_size)

def ac_market_policy(schedule):
    return FrontLoadedMarketOnly(schedule=schedule)
