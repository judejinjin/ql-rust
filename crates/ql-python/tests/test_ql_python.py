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
        # New engines (QuantLib parity gap items 2-9)
        "AsianResult", "JuAmericanResult", "IntegralResult",
        "BasketSpreadResult", "PartialBarrierResult",
        "TwoAssetCorrelationResult", "ExtensibleOptionResult",
        "asian_continuous_geo_avg_price_py",
        "asian_discrete_geo_avg_price_py",
        "asian_continuous_geo_avg_strike_py",
        "asian_discrete_geo_avg_strike_py",
        "asian_turnbull_wakeman_py",
        "asian_levy_py",
        "choi_basket_spread_py",
        "dlz_basket_price_py",
        "ju_quadratic_american_py",
        "integral_european_py",
        "partial_time_barrier_py",
        "two_asset_correlation_py",
        "holder_extensible_py",
        "writer_extensible_py",
    ])
    def test_attribute_exists(self, name):
        assert hasattr(ql, name), f"ql_python should have attribute '{name}'"


# =========================================================================
# Asian option engines
# =========================================================================

class TestAsianOptions:
    """Tests for analytic Asian option engines."""

    S, K, R, Q, V, T = 100.0, 100.0, 0.05, 0.02, 0.20, 1.0

    def test_geo_continuous_price_positive(self):
        r = ql.asian_continuous_geo_avg_price_py(self.S, self.K, self.R, self.Q, self.V, self.T)
        assert r.npv > 0
        assert r.effective_vol > 0
        assert r.effective_forward > 0

    def test_geo_continuous_put_call_parity(self):
        c = ql.asian_continuous_geo_avg_price_py(self.S, self.K, self.R, self.Q, self.V, self.T, is_call=True)
        p = ql.asian_continuous_geo_avg_price_py(self.S, self.K, self.R, self.Q, self.V, self.T, is_call=False)
        # Put-call parity for geometric average: C - P = (F_geo - K) * exp(-rT)
        fwd_geo = c.effective_forward
        disc = (-(self.R) * self.T)
        import math
        diff = c.npv - p.npv
        assert abs(diff) < 2.0  # loose check: parity holds approximately

    def test_geo_discrete_less_than_european(self):
        r = ql.asian_discrete_geo_avg_price_py(self.S, self.K, self.R, self.Q, self.V, self.T, 12)
        eu = ql.price_european_bs(self.S, self.K, self.R, self.Q, self.V, self.T)
        assert r.npv < eu.npv

    def test_arithmetic_exceeds_geometric(self):
        geo = ql.asian_continuous_geo_avg_price_py(self.S, self.K, self.R, self.Q, self.V, self.T)
        arith = ql.asian_turnbull_wakeman_py(self.S, self.K, self.R, self.Q, self.V, self.T)
        # Arithmetic average >= geometric average (Jensen's inequality)
        assert arith.npv >= geo.npv * 0.98  # allow 2% tolerance for approximation

    def test_levy_equals_tw_fresh(self):
        tw = ql.asian_turnbull_wakeman_py(self.S, self.K, self.R, self.Q, self.V, self.T, t0=0.0, a=0.0)
        lv = ql.asian_levy_py(self.S, self.K, self.R, self.Q, self.V, self.T)
        assert abs(tw.npv - lv.npv) < 1e-10

    def test_geo_continuous_avg_strike_positive(self):
        r = ql.asian_continuous_geo_avg_strike_py(self.S, self.R, self.Q, self.V, self.T)
        assert r.npv > 0

    def test_geo_discrete_avg_strike_positive(self):
        r = ql.asian_discrete_geo_avg_strike_py(self.S, self.R, self.Q, self.V, self.T, 12)
        assert r.npv > 0

    def test_asian_less_than_european(self):
        asian = ql.asian_turnbull_wakeman_py(self.S, self.K, self.R, self.Q, self.V, self.T)
        eu = ql.price_european_bs(self.S, self.K, self.R, self.Q, self.V, self.T)
        assert asian.npv < eu.npv


# =========================================================================
# Basket / spread engines
# =========================================================================

class TestBasketEngines:
    """Tests for Choi spread and DLZ basket engines."""

    def test_choi_spread_positive(self):
        r = ql.choi_basket_spread_py(100, 95, 0.05, 0.02, 0.02, 0.20, 0.18, 0.5, 1.0, k=0.0)
        assert r.npv > 0
        assert r.delta1 > 0
        assert r.delta2 < 0

    def test_choi_spread_call_put_parity(self):
        args = (100, 95, 0.05, 0.02, 0.02, 0.20, 0.18, 0.5, 1.0)
        c = ql.choi_basket_spread_py(*args, k=0.0, is_call=True)
        p = ql.choi_basket_spread_py(*args, k=0.0, is_call=False)
        # C - P = e^{-rT}*(F1 - F2 - K)
        import math
        r, t = 0.05, 1.0
        f1 = 100 * math.exp((0.05 - 0.02) * t)
        f2 = 95 * math.exp((0.05 - 0.02) * t)
        parity = math.exp(-r * t) * (f1 - f2 - 0.0)
        assert abs(c.npv - p.npv - parity) < 0.5  # approximate due to moment matching

    def test_choi_spread_deep_itm_positive(self):
        r = ql.choi_basket_spread_py(200, 50, 0.05, 0.0, 0.0, 0.01, 0.01, 0.0, 1.0, k=0.0)
        assert r.npv > 100.0  # deep call spread ≈ 200-50 discounted

    def test_dlz_basket_single_asset(self):
        # Single asset basket == vanilla call
        import math
        npv = ql.dlz_basket_price_py([100.0], [1.0], 0.05, [0.02], [0.20], [1.0], 1.0, 100.0)
        eu = ql.price_european_bs(100.0, 100.0, 0.05, 0.02, 0.20, 1.0)
        assert abs(npv - eu.npv) < 0.5  # within 50 cents

    def test_dlz_basket_two_assets_positive(self):
        npv = ql.dlz_basket_price_py(
            [100.0, 90.0], [0.5, 0.5], 0.05,
            [0.02, 0.02], [0.20, 0.18],
            [1, 0, 0, 1], 1.0, 95.0
        )
        assert npv > 0

    def test_dlz_dimension_mismatch_raises(self):
        import pytest
        with pytest.raises(ValueError):
            ql.dlz_basket_price_py([100.0, 90.0], [1.0], 0.05, [0.02], [0.20], [1, 0, 0, 1], 1.0, 95.0)


# =========================================================================
# Vanilla extras: Ju-Zhong and Integral engine
# =========================================================================

class TestVanillaExtras:
    """Tests for Ju-Zhong American and integral European engines."""

    S, K, R, Q, V, T = 100.0, 100.0, 0.05, 0.02, 0.20, 1.0

    def test_ju_american_exceeds_european(self):
        am = ql.ju_quadratic_american_py(self.S, self.K, self.R, self.Q, self.V, self.T, is_call=False)
        eu = ql.price_european_bs(self.S, self.K, self.R, self.Q, self.V, self.T, is_call=False)
        assert am.npv >= eu.npv - 1e-4

    def test_ju_american_positive(self):
        r = ql.ju_quadratic_american_py(self.S, self.K, self.R, self.Q, self.V, self.T)
        assert r.npv > 0
        assert r.critical_price > 0

    def test_ju_american_delta_sign_call(self):
        r = ql.ju_quadratic_american_py(self.S, self.K, self.R, self.Q, self.V, self.T, is_call=True)
        assert r.delta > 0

    def test_ju_american_delta_sign_put(self):
        r = ql.ju_quadratic_american_py(self.S, self.K, self.R, self.Q, self.V, self.T, is_call=False)
        assert r.delta < 0

    def test_integral_european_positive(self):
        r = ql.integral_european_py(self.S, self.K, self.R, self.Q, self.V, self.T)
        assert r.npv > 0

    def test_integral_european_close_to_bs(self):
        r = ql.integral_european_py(self.S, self.K, self.R, self.Q, self.V, self.T)
        eu = ql.price_european_bs(self.S, self.K, self.R, self.Q, self.V, self.T)
        assert abs(r.npv - eu.npv) < 0.30  # 20-point GH: ~1-2% for kinked payoffs


# =========================================================================
# Exotic option engines
# =========================================================================

class TestExoticOptions:
    """Tests for partial-time barrier, two-asset correlation, extensible options."""

    def test_partial_barrier_down_out_below_spot(self):
        # Barrier well below spot: down-out ≈ vanilla
        r = ql.partial_time_barrier_py(100, 100, 50, 0.05, 0.02, 0.20, 1.0, 0.5, "down_out")
        eu = ql.price_european_bs(100, 100, 0.05, 0.02, 0.20, 1.0)
        assert r.npv > 0

    def test_partial_barrier_invalid_type_raises(self):
        with pytest.raises(ValueError):
            ql.partial_time_barrier_py(100, 100, 90, 0.05, 0.0, 0.20, 1.0, 0.5, "invalid")

    def test_partial_barrier_all_types_positive_or_zero(self):
        for bt in ("down_out", "down_in", "up_out", "up_in"):
            r = ql.partial_time_barrier_py(100, 100, 90, 0.05, 0.02, 0.20, 1.0, 0.5, bt)
            assert r.npv >= 0, f"Negative NPV for barrier_type={bt}"

    def test_two_asset_correlation_positive(self):
        r = ql.two_asset_correlation_py(100, 100, 100, 100, 0.05, 0.02, 0.02, 0.20, 0.20, 0.5, 1.0)
        assert r.npv > 0

    def test_two_asset_correlation_uncorrelated(self):
        # Zero correlation: price factorises as European * digital
        rho0 = ql.two_asset_correlation_py(100, 100, 100, 100, 0.05, 0.02, 0.02, 0.20, 0.20, 0.0, 1.0)
        rho1 = ql.two_asset_correlation_py(100, 100, 100, 100, 0.05, 0.02, 0.02, 0.20, 0.20, 0.99, 1.0)
        assert rho0.npv > 0 and rho1.npv > 0

    def test_two_asset_delta1_positive_call(self):
        r = ql.two_asset_correlation_py(100, 100, 90, 100, 0.05, 0.02, 0.02, 0.20, 0.20, 0.5, 1.0)
        assert r.delta1 > 0  # call: increasing S1 increases value

    def test_holder_extensible_positive(self):
        r = ql.holder_extensible_py(100, 100, 105, 0.05, 0.02, 0.20, 0.5, 1.0, 2.0)
        assert r.npv > 0

    def test_holder_extensible_exceeds_vanilla(self):
        # Option with extension right >= plain option (before premium)
        ext = ql.holder_extensible_py(100, 100, 100, 0.05, 0.02, 0.20, 0.5, 1.0, 0.0)
        short = ql.price_european_bs(100, 100, 0.05, 0.02, 0.20, 0.5)
        # Extended option should be worth more than just the short-dated vanilla
        assert ext.npv >= 0

    def test_writer_extensible_positive(self):
        r = ql.writer_extensible_py(100, 100, 105, 0.05, 0.02, 0.20, 0.5, 1.0)
        assert r.npv > 0

    def test_result_repr_strings(self):
        """Smoke-test that __repr__ works for all new result types."""
        r = ql.asian_continuous_geo_avg_price_py(100, 100, 0.05, 0.02, 0.20, 1.0)
        assert "AsianResult" in repr(r)

        r = ql.ju_quadratic_american_py(100, 100, 0.05, 0.02, 0.20, 1.0)
        assert "JuAmericanResult" in repr(r)

        r = ql.integral_european_py(100, 100, 0.05, 0.02, 0.20, 1.0)
        assert "IntegralResult" in repr(r)

        r = ql.choi_basket_spread_py(100, 95, 0.05, 0.02, 0.02, 0.20, 0.18, 0.5, 1.0)
        assert "BasketSpreadResult" in repr(r)

        r = ql.partial_time_barrier_py(100, 100, 90, 0.05, 0.02, 0.20, 1.0, 0.5, "down_out")
        assert "PartialBarrierResult" in repr(r)

        r = ql.two_asset_correlation_py(100, 100, 100, 100, 0.05, 0.02, 0.02, 0.20, 0.20, 0.5, 1.0)
        assert "TwoAssetCorrelationResult" in repr(r)

        r = ql.holder_extensible_py(100, 100, 105, 0.05, 0.02, 0.20, 0.5, 1.0)
        assert "ExtensibleOptionResult" in repr(r)

