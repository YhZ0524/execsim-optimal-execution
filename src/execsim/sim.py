import numpy as np

class Simulator:
    def __init__(self, profile_df, dt, epsilon=0.0005, seed=None):
        self.profile = profile_df.reset_index(drop=True)
        self.dt = dt
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def step_fills(self, t, lim_qty, depth, mkt_qty):
        lam = self.profile.loc[t, "lambda_mo"]
        kappa = self.profile.loc[t, "kappa"]
        p_mo = 1.0 - np.exp(-lam * self.dt)
        lam_eff = lam * np.exp(-kappa * depth)
        p_lim = 1.0 - np.exp(-lam_eff * self.dt)
        lim_fill = 0
        if lim_qty > 0:
            lim_fill = self.rng.binomial(lim_qty, p_lim)
        mkt_fill = mkt_qty
        return lim_fill, mkt_fill

    def step_costs(self, t, lim_fill, depth, mkt_fill, side=1):
        mid = self.profile.loc[t, "mid"]
        hs = self.profile.loc[t, "half_spread"]
        px_lim = mid - hs - depth
        px_mkt = mid + hs + self.epsilon
        cost = 0.0
        if side == 1:
            cost += lim_fill * px_lim + mkt_fill * px_mkt
        else:
            cost -= lim_fill * px_lim + mkt_fill * px_mkt
        return cost

    def run(self, policy, q_total, schedule, force_finish=True):
        n = len(self.profile)
        cash = 0.0
        q = 0
        fills_lim = []
        fills_mkt = []
        depths = []
        mids = self.profile["mid"].values.copy()
        for t in range(n):
            remaining = q_total - q
            if remaining <= 0:
                fills_lim.append(0)
                fills_mkt.append(0)
                depths.append(0.0)
                continue
            q_target_t = int(np.round(schedule[t]))
            lim_qty, depth, mkt_qty = policy.decide(t, q, q_target_t, remaining, self.profile.loc[t, "half_spread"])
            lim_fill, mkt_fill = self.step_fills(t, int(lim_qty), float(depth), int(mkt_qty))
            depths.append(float(depth))
            fills_lim.append(int(lim_fill))
            fills_mkt.append(int(mkt_fill))
            trade_cost = self.step_costs(t, lim_fill, float(depth), mkt_fill, side=1)
            cash -= trade_cost
            q += lim_fill + mkt_fill
        if force_finish and q < q_total:
            t = n - 1
            add = q_total - q
            hs = self.profile.loc[t, "half_spread"]
            mid = self.profile.loc[t, "mid"]
            epsilon = self.epsilon
            cash -= add * (mid + hs + epsilon)
            q += add
            fills_mkt[-1] += add
        avg_price = cash / q_total
        arrival_ref = self.profile.loc[0, "mid"]
        shortfall = avg_price - arrival_ref
        out = {"avg_price": avg_price, "shortfall": shortfall, "q_filled": q, "fills_lim": np.array(fills_lim), "fills_mkt": np.array(fills_mkt), "depths": np.array(depths), "mids": mids}
        return out
