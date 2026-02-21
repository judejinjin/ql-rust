"""
Comprehensive test suite for the ql-python bindings.

Run with:
    cd crates/ql-python
    maturin develop
    pytest tests/ -v
"""

import math
import pytest

import ql_python as ql


# =========================================================================
# Date
# =========================================================================

class TestDate:
    def test_create_date(self):
        d = ql.Date(2025, 6, 15)
        assert d.year == 2025
        assert d.month == 6
        assert d.day == 15

    def test_serial_round_trip(self):
        d = ql.Date(2025, 1, 1)
        assert d.serial > 0

    def test_add_days(self):
        d = ql.Date(2025, 1, 1)
        d2 = d.add_days(30)
        assert d2.month == 1
        assert d2.day == 31

    def test_days_between(self):
        d1 = ql.Date(2025, 1, 1)
        d2 = ql.Date(2025, 12, 31)
        assert d1.days_between(d2) == 364

    def test_end_of_month(self):
        d = ql.Date(2025, 2, 15)
        eom = d.end_of_month()
        assert eom.day == 28  # non-leap year

    def test_is_end_of_month(self):
        d = ql.Date(2025, 2, 28)
        assert d.is_end_of_month()

    def test_leap_year_end_of_month(self):
        d = ql.Date(2024, 2, 15)
        eom = d.end_of_month()
        assert eom.day == 29  # 2024 is a leap year

    def test_comparison_operators(self):
        d1 = ql.Date(2025, 1, 1)
        d2 = ql.Date(2025, 6, 15)
        assert d1 < d2
        assert d2 > d1
        assert d1 != d2
        assert d1 == ql.Date(2025, 1, 1)

    def test_invalid_date_raises(self):
        with pytest.raises(ValueError):
            ql.Date(2025, 13, 1)  # month 13

    def test_repr(self):
        d = ql.Date(2025, 6, 15)
        assert "2025" in repr(d)

    def test_today(self):
        d = ql.Date.today()
        assert d.year >= 2025


# =========================================================================
# Period
# =========================================================================

class TestPeriod:
    def test_create_days(self):
        p = ql.Period(30, "D")
        assert "30" in repr(p)
        assert "D" in repr(p)

    def test_create_months(self):
        p = ql.Period(6, "M")
        assert "6" in repr(p)
        assert "M" in repr(p)

    def test_create_years(self):
        p = ql.Period(5, "Y")
        assert "5" in repr(p)

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError):
            ql.Period(1, "X")


# =========================================================================
# Schedule
# =========================================================================

class TestSchedule:
    def test_semiannual_schedule(self):
        start = ql.Date(2025, 1, 15)
        end = ql.Date(2030, 1, 15)
        sched = ql.Schedule(start, end, "Semiannual", "TARGET", "ModifiedFollowing")
        assert len(sched) >= 2
        dates = sched.dates()
        assert len(dates) == len(sched)

    def test_annual_schedule(self):
        start = ql.Date(2025, 1, 1)
        end = ql.Date(2030, 1, 1)
        sched = ql.Schedule(start, end, "Annual", "NullCalendar", "Unadjusted")
        dates = sched.dates()
        assert len(dates) == 6  # 5 years + 1

    def test_date_access(self):
        start = ql.Date(2025, 1, 15)
        end = ql.Date(2026, 1, 15)
        sched = ql.Schedule(start, end, "Quarterly")
        d0 = sched.date(0)
        assert d0.year == 2025

    def test_out_of_range_raises(self):
        start = ql.Date(2025, 1, 1)
        end = ql.Date(2025, 7, 1)
        sched = ql.Schedule(start, end, "Monthly")
        with pytest.raises(ValueError):
            sched.date(999)


# =========================================================================
# Calendar utilities
# =========================================================================

class TestCalendar:
    def test_is_business_day_target(self):
        # 2025-01-01 is New Year's Day → holiday in TARGET
        d = ql.Date(2025, 1, 1)
        assert not ql.is_business_day(d, "TARGET")

    def test_is_business_day_weekday(self):
        # 2025-06-16 (Monday) should be a business day
        d = ql.Date(2025, 6, 16)
        assert ql.is_business_day(d, "TARGET")

    def test_advance_date(self):
        d = ql.Date(2025, 1, 1)
        p = ql.Period(1, "M")
        result = ql.advance_date(d, p, "TARGET", "Following")
        assert result.month >= 1

    def test_year_fraction_act365(self):
        d1 = ql.Date(2025, 1, 1)
        d2 = ql.Date(2026, 1, 1)
        yf = ql.year_fraction(d1, d2, "Actual365Fixed")
        assert abs(yf - 1.0) < 0.01

    def test_business_days_between(self):
        d1 = ql.Date(2025, 6, 2)  # Monday
        d2 = ql.Date(2025, 6, 6)  # Friday
        bdays = ql.business_days_between(d1, d2, "TARGET")
        assert bdays == 4  # Mon-Fri = 4 business days (exclusive of start)


# =========================================================================
# FlatForward
# =========================================================================

class TestFlatForward:
    def test_create_and_rate(self):
        ref = ql.Date(2025, 1, 15)
        ff = ql.FlatForward(ref, 0.05)
        assert abs(ff.rate - 0.05) < 1e-12

    def test_discount_at_zero(self):
        ref = ql.Date(2025, 1, 15)
        ff = ql.FlatForward(ref, 0.05)
        assert abs(ff.discount_t(0.0) - 1.0) < 1e-12

    def test_discount_at_one_year(self):
        ref = ql.Date(2025, 1, 15)
        ff = ql.FlatForward(ref, 0.05)
        df = ff.discount_t(1.0)
        expected = math.exp(-0.05)
        assert abs(df - expected) < 1e-10

    def test_forward_rate_constant(self):
        ref = ql.Date(2025, 1, 15)
        ff = ql.FlatForward(ref, 0.03)
        assert abs(ff.forward_rate_t(0.5) - 0.03) < 1e-10
        assert abs(ff.forward_rate_t(5.0) - 0.03) < 1e-10

    def test_repr(self):
        ref = ql.Date(2025, 1, 15)
        ff = ql.FlatForward(ref, 0.05)
        assert "FlatForward" in repr(ff)


# =========================================================================
# NelsonSiegelCurve
# =========================================================================

class TestNelsonSiegel:
    def test_direct_construction(self):
        ns = ql.NelsonSiegelCurve(0.05, -0.02, 0.01, 1.5)
        params = ns.params
        assert len(params) == 4
        assert abs(params[0] - 0.05) < 1e-12

    def test_zero_rate(self):
        ns = ql.NelsonSiegelCurve(0.05, -0.02, 0.01, 1.5)
        rate = ns.zero_rate(1.0)
        assert rate > 0

    def test_discount(self):
        ns = ql.NelsonSiegelCurve(0.05, 0.0, 0.0, 1.0)
        df = ns.discount(1.0)
        assert 0 < df < 1

    def test_fit(self):
        maturities = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        rates = [0.045, 0.046, 0.047, 0.048, 0.050, 0.052, 0.054]
        ns = ql.NelsonSiegelCurve.fit(maturities, rates)
        params = ns.params
        assert len(params) == 4
        # The fitted curve should reproduce rates reasonably
        for m, r in zip(maturities, rates):
            fitted = ns.zero_rate(m)
            assert abs(fitted - r) < 0.005  # within 50bp

    def test_repr(self):
        ns = ql.NelsonSiegelCurve(0.05, -0.02, 0.01, 1.5)
        assert "NelsonSiegel" in repr(ns)


# =========================================================================
# SvenssonCurve
# =========================================================================

class TestSvensson:
    def test_direct_construction(self):
        sv = ql.SvenssonCurve(0.05, -0.02, 0.01, 0.005, 1.5, 5.0)
        params = sv.params
        assert len(params) == 6
        assert abs(params[0] - 0.05) < 1e-12

    def test_zero_rate(self):
        sv = ql.SvenssonCurve(0.05, -0.02, 0.01, 0.005, 1.5, 5.0)
        rate = sv.zero_rate(5.0)
        assert rate > 0

    def test_discount(self):
        sv = ql.SvenssonCurve(0.05, 0.0, 0.0, 0.0, 1.0, 2.0)
        df = sv.discount(1.0)
        assert 0 < df < 1


# =========================================================================
# VanillaOption
# =========================================================================

class TestVanillaOption:
    def test_create_call(self):
        expiry = ql.Date(2026, 1, 15)
        opt = ql.VanillaOption(100.0, expiry, is_call=True)
        assert opt.strike == 100.0
        assert opt.is_call

    def test_create_put(self):
        expiry = ql.Date(2026, 1, 15)
        opt = ql.VanillaOption(90.0, expiry, is_call=False)
        assert opt.strike == 90.0
        assert not opt.is_call

    def test_repr(self):
        expiry = ql.Date(2026, 1, 15)
        opt = ql.VanillaOption(100.0, expiry, is_call=True)
        assert "Call" in repr(opt)
        assert "100" in repr(opt)


# =========================================================================
# Black-Scholes pricing
# =========================================================================

class TestBlackScholes:
    def test_atm_call_positive(self):
        res = ql.price_european_bs(spot=100, strike=100, r=0.05, q=0.0, vol=0.2, t=1.0)
        assert res.npv > 0
        assert 0 < res.delta < 1
        assert res.gamma > 0
        assert res.vega > 0

    def test_put_call_parity(self):
        S, K, r, q, vol, t = 100.0, 100.0, 0.05, 0.02, 0.2, 1.0
        call = ql.price_european_bs(S, K, r, q, vol, t, is_call=True)
        put = ql.price_european_bs(S, K, r, q, vol, t, is_call=False)
        # C - P = S*e^{-qT} - K*e^{-rT}
        lhs = call.npv - put.npv
        rhs = S * math.exp(-q * t) - K * math.exp(-r * t)
        assert abs(lhs - rhs) < 1e-8

    def test_deep_itm_call_delta_near_one(self):
        res = ql.price_european_bs(spot=200, strike=50, r=0.05, q=0.0, vol=0.2, t=0.1)
        assert res.delta > 0.99

    def test_deep_otm_call_delta_near_zero(self):
        res = ql.price_european_bs(spot=50, strike=200, r=0.05, q=0.0, vol=0.2, t=0.1)
        assert res.delta < 0.01

    def test_repr(self):
        res = ql.price_european_bs(100, 100, 0.05, 0.0, 0.2, 1.0)
        assert "AnalyticResults" in repr(res)


# =========================================================================
# Implied volatility
# =========================================================================

class TestImpliedVol:
    def test_round_trip(self):
        vol = 0.20
        res = ql.price_european_bs(100, 100, 0.05, 0.0, vol, 1.0, is_call=True)
        iv = ql.implied_vol(res.npv, 100, 100, 0.05, 0.0, 1.0, is_call=True)
        assert abs(iv - vol) < 1e-6

    def test_put_implied_vol(self):
        vol = 0.30
        res = ql.price_european_bs(100, 100, 0.05, 0.0, vol, 1.0, is_call=False)
        iv = ql.implied_vol(res.npv, 100, 100, 0.05, 0.0, 1.0, is_call=False)
        assert abs(iv - vol) < 1e-6


# =========================================================================
# Monte Carlo European
# =========================================================================

class TestMCEuropean:
    def test_mc_call_near_bs(self):
        bs = ql.price_european_bs(100, 100, 0.05, 0.0, 0.2, 1.0, is_call=True)
        mc = ql.mc_european_py(100, 100, 0.05, 0.0, 0.2, 1.0, is_call=True,
                               num_paths=500_000, seed=42)
        # MC should be within 3 std errors of BS
        assert abs(mc.npv - bs.npv) < 3 * mc.std_error + 0.1

    def test_mc_put(self):
        mc = ql.mc_european_py(100, 110, 0.05, 0.0, 0.25, 0.5, is_call=False,
                               num_paths=100_000, seed=123)
        assert mc.npv > 0
        assert mc.num_paths == 100_000

    def test_repr(self):
        mc = ql.mc_european_py(100, 100, 0.05, 0.0, 0.2, 1.0, num_paths=1000)
        assert "MCResult" in repr(mc)


# =========================================================================
# Binomial CRR
# =========================================================================

class TestBinomialCRR:
    def test_european_call_near_bs(self):
        bs = ql.price_european_bs(100, 100, 0.05, 0.0, 0.2, 1.0, is_call=True)
        crr = ql.binomial_crr_py(100, 100, 0.05, 0.0, 0.2, 1.0, is_call=True,
                                  is_american=False, num_steps=500)
        assert abs(crr.npv - bs.npv) < 0.1

    def test_american_put_at_least_european(self):
        euro = ql.binomial_crr_py(100, 110, 0.05, 0.0, 0.3, 1.0, is_call=False,
                                   is_american=False, num_steps=200)
        amer = ql.binomial_crr_py(100, 110, 0.05, 0.0, 0.3, 1.0, is_call=False,
                                   is_american=True, num_steps=200)
        assert amer.npv >= euro.npv - 1e-10

    def test_lattice_result_greeks(self):
        res = ql.binomial_crr_py(100, 100, 0.05, 0.0, 0.2, 1.0)
        assert hasattr(res, 'delta')
        assert hasattr(res, 'gamma')
        assert hasattr(res, 'theta')


# =========================================================================
# Barone-Adesi-Whaley
# =========================================================================

class TestBAW:
    def test_american_put(self):
        res = ql.barone_adesi_whaley_py(100, 110, 0.05, 0.0, 0.3, 1.0, is_call=False)
        assert res.npv > 0
        assert res.early_exercise_premium >= 0
        assert res.critical_price > 0

    def test_call_no_dividend_equals_european(self):
        # For zero dividend, American call price equals European
        baw = ql.barone_adesi_whaley_py(100, 100, 0.05, 0.0, 0.2, 1.0, is_call=True)
        bs = ql.price_european_bs(100, 100, 0.05, 0.0, 0.2, 1.0, is_call=True)
        assert abs(baw.npv - bs.npv) < 0.5  # close but approximation

    def test_repr(self):
        res = ql.barone_adesi_whaley_py(100, 100, 0.05, 0.0, 0.2, 1.0)
        assert "AmericanResult" in repr(res)


# =========================================================================
# Bjerksund-Stensland
# =========================================================================

class TestBjerksundStensland:
    def test_american_put(self):
        res = ql.bjerksund_stensland_py(100, 110, 0.05, 0.0, 0.3, 1.0, is_call=False)
        assert res.npv > 0
        assert res.early_exercise_premium >= 0

    def test_agrees_with_baw(self):
        baw = ql.barone_adesi_whaley_py(100, 110, 0.05, 0.02, 0.25, 1.0, is_call=False)
        bjs = ql.bjerksund_stensland_py(100, 110, 0.05, 0.02, 0.25, 1.0, is_call=False)
        # Both are approximations; should be in similar range
        assert abs(baw.npv - bjs.npv) < 1.0


# =========================================================================
# Finite Differences
# =========================================================================

class TestFD:
    def test_european_call_near_bs(self):
        bs = ql.price_european_bs(100, 100, 0.05, 0.0, 0.2, 1.0, is_call=True)
        fd = ql.fd_black_scholes_py(100, 100, 0.05, 0.0, 0.2, 1.0, is_call=True,
                                     is_american=False, num_space=200, num_time=200)
        assert abs(fd.npv - bs.npv) < 0.5

    def test_fd_greeks(self):
        fd = ql.fd_black_scholes_py(100, 100, 0.05, 0.0, 0.2, 1.0, is_call=True)
        assert 0 < fd.delta < 1
        assert fd.gamma > 0

    def test_american_put_premium(self):
        euro = ql.fd_black_scholes_py(100, 110, 0.05, 0.0, 0.3, 1.0, is_call=False,
                                       is_american=False, num_space=200, num_time=200)
        amer = ql.fd_black_scholes_py(100, 110, 0.05, 0.0, 0.3, 1.0, is_call=False,
                                       is_american=True, num_space=200, num_time=200)
        assert amer.npv >= euro.npv - 0.01

    def test_repr(self):
        fd = ql.fd_black_scholes_py(100, 100, 0.05, 0.0, 0.2, 1.0)
        assert "FDResult" in repr(fd)


# =========================================================================
# Heston
# =========================================================================

class TestHeston:
    def test_heston_call(self):
        res = ql.heston_price_py(
            spot=100, strike=100, r=0.05, q=0.0,
            v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
            t=1.0, is_call=True,
        )
        assert res.npv > 0
        assert 0 < res.p1 < 1
        assert 0 < res.p2 < 1

    def test_heston_put(self):
        res = ql.heston_price_py(
            spot=100, strike=100, r=0.05, q=0.0,
            v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
            t=1.0, is_call=False,
        )
        assert res.npv > 0

    def test_repr(self):
        res = ql.heston_price_py(100, 100, 0.05, 0.0, 0.04, 2.0, 0.04, 0.3, -0.7, 1.0)
        assert "HestonResult" in repr(res)


# =========================================================================
# MC Barrier
# =========================================================================

class TestMCBarrier:
    def test_down_and_out_call(self):
        res = ql.mc_barrier_py(
            spot=100, strike=100, barrier=80,
            r=0.05, vol=0.2, t=1.0,
            is_call=True, is_up=False, is_knock_in=False,
            num_paths=50_000, seed=42,
        )
        assert res.npv >= 0
        assert res.num_paths == 50_000

    def test_up_and_in_call(self):
        res = ql.mc_barrier_py(
            spot=100, strike=100, barrier=120,
            r=0.05, vol=0.25, t=1.0,
            is_call=True, is_up=True, is_knock_in=True,
            num_paths=50_000, seed=123,
        )
        assert res.npv >= 0


# =========================================================================
# Kirk spread option
# =========================================================================

class TestKirkSpread:
    def test_call_positive(self):
        price = ql.kirk_spread_call_py(
            s1=100, s2=90, strike=5, r=0.05,
            q1=0.0, q2=0.0, vol1=0.2, vol2=0.25, rho=0.5, t=1.0,
        )
        assert price > 0

    def test_put_positive(self):
        price = ql.kirk_spread_put_py(
            s1=90, s2=100, strike=5, r=0.05,
            q1=0.0, q2=0.0, vol1=0.2, vol2=0.25, rho=0.5, t=1.0,
        )
        assert price > 0

    def test_put_call_parity(self):
        args = dict(s1=100, s2=90, strike=5, r=0.05,
                    q1=0.0, q2=0.0, vol1=0.2, vol2=0.25, rho=0.5, t=1.0)
        call = ql.kirk_spread_call_py(**args)
        put = ql.kirk_spread_put_py(**args)
        # spread put-call parity: C - P = DF * (F1 - F2 - K)
        df = math.exp(-0.05 * 1.0)
        f1 = 100 * math.exp((0.05 - 0.0) * 1.0)
        f2 = 90 * math.exp((0.05 - 0.0) * 1.0)
        expected = df * (f1 - f2 - 5)
        assert abs((call - put) - expected) < 0.5


# =========================================================================
# SABR
# =========================================================================

class TestSABR:
    def test_atm_vol(self):
        vol = ql.sabr_vol_py(
            strike=100, forward=100, expiry=1.0,
            alpha=0.3, beta=1.0, rho=-0.3, nu=0.4,
        )
        assert vol > 0
        assert vol < 2.0  # reasonable bound

    def test_skew(self):
        # Lower strike should have higher vol with negative rho
        vol_low = ql.sabr_vol_py(80, 100, 1.0, 0.3, 1.0, -0.3, 0.4)
        vol_high = ql.sabr_vol_py(120, 100, 1.0, 0.3, 1.0, -0.3, 0.4)
        assert vol_low > vol_high  # negative skew


# =========================================================================
# Smoke test: all functions are importable
# =========================================================================

class TestModuleApi:
    """Verify all expected names exist in the module."""

    @pytest.mark.parametrize("name", [
        "Date", "Period", "Schedule",
        "FlatForward", "NelsonSiegelCurve", "SvenssonCurve", "PiecewiseYieldCurve",
        "VanillaOption",
        "AnalyticResults", "MCResult", "LatticeResult",
        "SwapResults", "BondResults",
        "AmericanResult", "FDResult", "HestonResult",
        "price_european_bs", "implied_vol",
        "mc_european_py", "mc_barrier_py",
        "binomial_crr_py", "fd_black_scholes_py",
        "barone_adesi_whaley_py", "bjerksund_stensland_py",
        "heston_price_py",
        "kirk_spread_call_py", "kirk_spread_put_py",
        "sabr_vol_py",
        "bootstrap_yield_curve",
        "is_business_day", "advance_date", "year_fraction",
        "business_days_between",
    ])
    def test_attribute_exists(self, name):
        assert hasattr(ql, name), f"ql_python should have attribute '{name}'"
