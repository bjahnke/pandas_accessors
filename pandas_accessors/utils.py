import pandas as pd
import numpy as np
from dataclasses import dataclass
import typing as t


@dataclass
class TrailStop:
    """
    pos_price_col: price column to base trail stop movement off of
    neg_price_col: price column to check if stop was crossed
    cum_extreme: cummin/cummax, name of function to use to calculate trailing stop direction
    """

    neg_price_col: str
    pos_price_col: str
    cum_extreme: str
    dir: int

    # def init_trail_stop(
    #     self, price: pd.DataFrame, initial_trail_price, entry_date, rg_end_date
    # ) -> pd.Series:
    #     """
    #     :param rg_end_date:
    #     :param entry_date:
    #     :param price:
    #     :param initial_trail_price:
    #     :return:
    #     """
    #     # trail = price.close.loc[entry_date: rg_end_date] - (initial_trail_price * self.dir)
    #     # trail = (trail * self.dir).cummax() * self.dir
    #
    #     entry_price = price.close.loc[entry_date]
    #     trail_pct_from_entry = (entry_price - initial_trail_price) / entry_price
    #     extremes = price.loc[entry_date:rg_end_date, self.pos_price_col]
    #
    #     # when short, pct should be negative, pushing modifier above one
    #     trail_modifier = 1 - trail_pct_from_entry
    #     # trail stop reaction must be delayed one bar since same bar reaction cannot be determined
    #     trail_stop: pd.Series = (
    #         getattr(extremes, self.cum_extreme)() * trail_modifier
    #     ).shift(1)
    #     # back patch nan after shifting
    #     trail_stop.iat[0] = trail_stop.iat[1]
    #
    #     return trail_stop

    def init_atr_stop(self, stop_line, entry_date, rg_end_date, modified_atr):
        _sl = stop_line - (self.dir * modified_atr.loc[entry_date: rg_end_date])
        return _sl

    def init_trail_stop(self, price, initial_trail_price, entry_date, rg_end_date):
        init_trail_diff = abs(initial_trail_price - price.close.loc[entry_date])
        trail = price.close.loc[entry_date: rg_end_date] - (init_trail_diff * self.dir)
        trail = (trail * self.dir).cummax() * self.dir
        return trail

    def init_stop_loss(self, price, stop_price, entry_date, rg_end_date):
        stop_line = pd.Series(index=price.index)
        stop_line.loc[entry_date: rg_end_date] = stop_price
        return stop_line

    def exit_signal(self, price: pd.DataFrame, trail_stop: pd.Series) -> pd.Series:
        """detect where price has crossed price"""
        return ((trail_stop - price[self.neg_price_col]) * self.dir) >= 0

    def target_exit_signal(self, price: pd.DataFrame, target_price) -> pd.Series:
        """detect where price has crossed price"""
        return ((target_price - price['close']) * self.dir) <= 0

    def get_stop_price(
        self, price: pd.DataFrame, stop_date, offset_pct: float,
    ) -> float:
        """calculate stop price given a date and percent to offset the stop point from the peak"""
        pct_from_peak = 1 - (offset_pct * self.dir)
        return price['close'].at[stop_date] * pct_from_peak

    def cap_trail_stop(self, trail_data: pd.Series, cap_price, set_price=None) -> pd.Series:
        """apply cap to trail stop"""
        if set_price is None:
            set_price = cap_price
        trail_data.loc[((trail_data - cap_price) * self.dir) > 0] = set_price
        return trail_data


def pyramid(position, root=2):
    return 1 / (1 + position) ** (1 / root)


def kelly(win_rate, avg_win, avg_loss):
    """kelly position sizer, returns risk budget as percent"""
    return win_rate / np.abs(avg_loss) - (1 - win_rate) / avg_win


def other_kelly(win_rate, avg_win, avg_loss):
    return win_rate - (1 - win_rate) / (avg_win/abs(avg_loss))


def kelly_fractional():
    pass


def eqty_risk_shares(px, r_pct, eqty, risk, lot=1, fx=0):
    nominal_sizes, clamped_sizes = init_eqty_risk_nominal_sizes(px, r_pct, eqty, risk, fx)
    return nominal_size_to_shares(clamped_sizes, px=px, lot=lot)


def nominal_size_to_shares(nominal_sizes, px, lot=1):
    """translate nominal position size to shares"""
    return round(((nominal_sizes // (px * lot)) * lot), 0)


def init_eqty_risk_nominal_sizes(r_pct, eqty, risk, fx=0, leverage=2) -> t.Tuple[pd.Series, pd.Series]:
    """
    get the initial size of the position, not considering lot resolution or
    entry price
    cap size to less than equity x 2 (leverage)
    """
    budget = eqty * risk
    if fx > 0:
        budget *= fx

    nominal_sizes = budget / r_pct
    nominal_clamped_sizes = nominal_sizes.copy()
    exceed_limit = nominal_sizes.loc[nominal_sizes > (eqty * leverage)]
    if not exceed_limit.empty:
        nominal_clamped_sizes.loc[exceed_limit.index] = eqty
    return nominal_sizes, nominal_clamped_sizes


def adjusted_nominal(nominal, px):
    return (nominal / px).apply(np.ceil) * px


def concave(ddr, floor):
    """
    For demo purpose only
    """
    if floor == 0:
        concave_res = ddr
    else:
        concave_res = ddr**floor
    return concave_res


def convex(ddr, floor):
    """
    # obtuse
    obtuse = 1 - acute
    """
    if floor == 0:
        convex_res = ddr
    else:
        convex_res = ddr ** (1 / floor)
    return convex_res


def risk_appetite(eqty, tolerance, mn, mx, span, shape) -> pd.Series:
    """
    position sizer

    eqty: equity curve series
    tolerance: tolerance for drawdown (<0)
    mn: min risk
    mx: max risk
    span: exponential moving average to smooth the risk_appetite
    shape: convex (>45 deg diagonal) = 1, concave (<diagonal) = -1, else: simple risk_appetite
    """
    # drawdown rebased
    eqty = pd.Series(eqty)
    watermark = eqty.expanding().max()
    # all-time-high peak equity
    drawdown = eqty / watermark - 1
    # drawdown from peak
    ddr = 1 - np.minimum(drawdown / tolerance, 1)
    # drawdown rebased to tolerance from 0 to 1
    avg_ddr = ddr.ewm(span=span).mean()
    # span rebased drawdown

    # Shape of the curve
    if shape == 1:  #
        _power = mx / mn  # convex
    elif shape == -1:
        _power = mn / mx  # concave
    else:
        _power = 1  # raw, straight line
    ddr_power = avg_ddr**_power  # ddr

    # mn + adjusted delta
    risk = mn + (mx - mn) * ddr_power

    return risk


def test_eqty_risk():
    # px = 2000
    # sl = 2222
    px = 2222
    sl = 2000

    eqty = 100000
    risk = -0.005
    fx = 110
    lot = 100

    res = eqty_risk_shares(px, sl, eqty, risk, fx, lot)
    print('done')


def test_risk_app():
    equity_curve = 25000
    tolerance = -0.1
    min_risk = -0.0025
    max_risk = -0.0075
    span = 5
    shape = 1

    convex_risk = risk_appetite(
        equity_curve, tolerance, min_risk, max_risk, span, shape
    )
    # convex_risk = -convex_risk * peak_equity
    print('d')


def simple_log_returns(prices: pd.Series) -> pd.Series:
    """calculates log returns of a price series"""
    return np.log(prices / prices.shift(1))


def rolling_grit(cumul_returns, window):
    rolling_peak = cumul_returns.rolling(window).max()
    draw_down_squared = (cumul_returns - rolling_peak) ** 2
    ulcer = draw_down_squared.rolling(window).sum() ** 0.5
    grit = cumul_returns / ulcer
    return grit.replace([-np.inf, np.inf], np.NAN)


def expanding_grit(cumul_returns):
    tt_peak = cumul_returns.expanding().max()
    draw_down_squared = (cumul_returns - tt_peak) ** 2
    ulcer = draw_down_squared.expanding().sum() ** 0.5
    grit = cumul_returns / ulcer
    return grit.replace([-np.inf, np.inf], np.NAN)


def rolling_profits(returns, window):
    profit_roll = returns.copy()
    profit_roll[profit_roll < 0] = 0
    profit_roll_sum = profit_roll.rolling(window).sum().fillna(method="ffill")
    return profit_roll_sum


def rolling_losses(returns, window):
    loss_roll = returns.copy()
    loss_roll[loss_roll > 0] = 0
    loss_roll_sum = loss_roll.rolling(window).sum().fillna(method="ffill")
    return loss_roll_sum


def expanding_profits(returns):
    profit_roll = returns.copy()
    profit_roll[profit_roll < 0] = 0
    profit_roll_sum = profit_roll.expanding().sum().fillna(method="ffill")
    return profit_roll_sum


def expanding_losses(returns):
    loss_roll = returns.copy()
    loss_roll[loss_roll > 0] = 0
    loss_roll_sum = loss_roll.expanding().sum().fillna(method="ffill")
    return loss_roll_sum


def profit_ratio(profits, losses):
    """
    # if losses goes to zero, ratio is a factor of profits
    # if profits is zero, ratio should be zero
    :param profits:
    :param losses:
    :return:
    """
    _losses = losses.copy()
    _losses.loc[_losses == 0] = np.nan
    _losses = abs(_losses.fillna(method="ffill"))
    pr = profits.fillna(method="ffill") / _losses
    return pr


def rolling_profit_ratio(returns, window):
    return profit_ratio(
        profits=rolling_profits(returns, window), losses=rolling_losses(returns, window)
    )


def expanding_profit_ratio(returns):
    return profit_ratio(
        profits=expanding_profits(returns), losses=expanding_losses(returns)
    )


def rolling_tail_ratio(cumul_returns, window, percentile, limit):
    left_tail = np.abs(cumul_returns.rolling(window).quantile(percentile))
    right_tail = cumul_returns.rolling(window).quantile(1 - percentile)
    np.seterr(all="ignore")
    tail = np.maximum(np.minimum(right_tail / left_tail, limit), -limit)
    return tail


def expanding_tail_ratio(cumul_returns, percentile, limit):
    left_tail = np.abs(cumul_returns.expanding().quantile(percentile))
    right_tail = cumul_returns.expanding().quantile(1 - percentile)
    np.seterr(all="ignore")
    tail = np.maximum(np.minimum(right_tail / left_tail, limit), -limit)
    return tail


def common_sense_ratio(pr, tr):
    return pr * tr


def expectancy(win_rate, avg_win, avg_loss):
    # win% * avg_win% - loss% * abs(avg_loss%)
    return win_rate * avg_win + (1 - win_rate) * avg_loss


def t_stat(signal_count, trading_edge):
    sqn = (signal_count**0.5) * trading_edge / trading_edge.std(ddof=0)
    return sqn


def t_stat_expanding(signal_count, edge):
    """"""
    sqn = (signal_count**0.5) * edge / edge.expanding().std(ddof=0)
    return sqn


def robustness_score(grit, csr, sqn):
    """
    clamp constituents of robustness score to >=0 to avoid positive scores when 2 values are negative
    exclude zeros when finding start date for rebase to avoid infinite score (divide by zero)
    """
    # TODO should it start at 1?
    _grit = grit.copy()
    _csr = csr.copy()
    _sqn = sqn.copy()
    # the score will be zero if on metric is negative
    _grit.loc[_grit < 0] = 0
    _csr.loc[_csr < 0] = 0
    _sqn.loc[_sqn < 0] = 0
    exclude_zeros = (_grit != 0) & (_csr != 0) & (_sqn != 0)
    try:
        start_date = max(
            _grit[pd.notnull(_grit) & exclude_zeros].index[0],
            _csr[pd.notnull(_csr) & exclude_zeros].index[0],
            _sqn[pd.notnull(_sqn) & exclude_zeros].index[0],
        )
    except IndexError:
        score = pd.Series(data=np.NaN, index=grit.index)
    else:
        _grit.loc[_grit.index < start_date] = np.nan
        _csr.loc[_csr.index < start_date] = np.nan
        _sqn.loc[_sqn.index < start_date] = np.nan
        score = (
            _grit * _csr * _sqn / (_grit[start_date] * _csr[start_date] * _sqn[start_date])
        )
    return score


def cumulative_returns_pct(returns, min_periods):
    return returns.expanding(min_periods=min_periods).sum().apply(np.exp) - 1


if __name__ == "__main__":
    test_risk_app()
