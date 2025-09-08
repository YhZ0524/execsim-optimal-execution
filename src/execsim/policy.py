import numpy as np

class ParamPolicy:
    def __init__(self, base_depth, alpha_depth, band, market_frac, lmax):
        self.base_depth = base_depth
        self.alpha_depth = alpha_depth
        self.band = band
        self.market_frac = market_frac
        self.lmax = lmax

    def decide(self, t, q_filled, q_target_t, remaining, half_spread):
        dev = q_filled - q_target_t
        buy_remaining = remaining
        mkt = 0
        if dev < -self.band:
            need = int(np.ceil((-dev) * self.market_frac))
            need = min(need, buy_remaining)
            mkt = max(0, need)
        lim = min(self.lmax, buy_remaining - mkt)
        depth = max(0.0, self.base_depth + self.alpha_depth * (-dev))
        depth = float(depth)
        return lim, depth, mkt

class TWAPMarketOnly:
    def __init__(self, slice_size):
        self.slice_size = slice_size

    def decide(self, t, q_filled, q_target_t, remaining, half_spread):
        mkt = min(self.slice_size, remaining)
        return 0, half_spread, mkt

class FrontLoadedMarketOnly:
    def __init__(self, schedule):
        self.schedule = schedule

    def decide(self, t, q_filled, q_target_t, remaining, half_spread):
        target_now = int(np.round(self.schedule[t]))
        deficit = max(0, target_now - q_filled)
        mkt = min(deficit, remaining)
        return 0, half_spread, mkt
