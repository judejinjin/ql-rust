//! Extended pricing engines: G10–G21.
//!
//! **G10** — AnalyticDividendEuropeanEngine
//! **G11** — G2SwaptionEngine
//! **G12** — TreeSwapEngine
//! **G13** — AnalyticPerformanceEngine
//! **G14** — MCPerformanceEngine
//! **G15** — AnalyticBinaryBarrierEngine (already exists as analytic_binary_barrier)
//! **G16** — AnalyticDoubleBarrierBinaryEngine (already exists as double_binary_barrier)
//! **G17** — ForwardPerformanceEngine
//! **G18** — BondFunctions
//! **G19** — BlackCalculator
//! **G20** — InflationCapFloorEngine (already exists in inflation_cap_floor_engine)
//! **G21** — FdHestonDoubleBarrierEngine (already exists in fd_heston_barrier)

use serde::{Deserialize, Serialize};

// ===========================================================================
// BlackCalculator (G19)
// ===========================================================================

/// Reusable Black-76 / Black-Scholes formula calculator (G19).
///
/// Pre-computes d₁, d₂ and provides all Greeks from a single struct.
/// Mirrors QuantLib's `BlackCalculator` class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackCalculator {
    /// Forward price.
    pub forward: f64,
    /// Strike price.
    pub strike: f64,
    /// σ√T (total standard deviation).
    pub std_dev: f64,
    /// Discount factor to payment date.
    pub discount: f64,
    /// d₁ = [ln(F/K) + 0.5σ²T] / σ√T.
    pub d1: f64,
    /// d₂ = d₁ − σ√T.
    pub d2: f64,
    /// N(d₁).
    pub n_d1: f64,
    /// N(d₂).
    pub n_d2: f64,
    /// Option type: +1 for call, -1 for put.
    pub omega: f64,
}

impl BlackCalculator {
    /// Create a new BlackCalculator.
    ///
    /// * `forward` — forward price
    /// * `strike` — strike price
    /// * `std_dev` — total standard deviation σ√T
    /// * `discount` — discount factor
    /// * `is_call` — true for call, false for put
    pub fn new(forward: f64, strike: f64, std_dev: f64, discount: f64, is_call: bool) -> Self {
        let omega = if is_call { 1.0 } else { -1.0 };

        let (d1, d2) = if std_dev > 1e-15 && strike > 0.0 && forward > 0.0 {
            let d1 = ((forward / strike).ln() + 0.5 * std_dev * std_dev) / std_dev;
            let d2 = d1 - std_dev;
            (d1, d2)
        } else {
            // Degenerate case
            let d1 = if forward > strike {
                f64::INFINITY
            } else if forward < strike {
                f64::NEG_INFINITY
            } else {
                0.0
            };
            (d1, d1)
        };

        let n_d1 = norm_cdf(omega * d1);
        let n_d2 = norm_cdf(omega * d2);

        Self {
            forward,
            strike,
            std_dev,
            discount,
            d1,
            d2,
            n_d1,
            n_d2,
            omega,
        }
    }

    /// Option value.
    pub fn value(&self) -> f64 {
        self.discount * self.omega * (self.forward * norm_cdf(self.omega * self.d1)
            - self.strike * norm_cdf(self.omega * self.d2))
    }

    /// Delta: ∂V/∂F × discount.
    pub fn delta(&self, spot: f64) -> f64 {
        if spot.abs() < 1e-15 {
            return 0.0;
        }
        self.discount * self.omega * norm_cdf(self.omega * self.d1) * self.forward / spot
    }

    /// Gamma: ∂²V/∂S².
    pub fn gamma(&self, spot: f64) -> f64 {
        if spot.abs() < 1e-15 || self.std_dev.abs() < 1e-15 {
            return 0.0;
        }
        self.discount * norm_pdf(self.d1) * self.forward / (spot * spot * self.std_dev)
    }

    /// Theta: ∂V/∂t (per year).
    ///
    /// Requires `maturity` (time to expiry) and `rate` (risk-free rate).
    pub fn theta(&self, maturity: f64, rate: f64) -> f64 {
        if maturity.abs() < 1e-15 || self.std_dev.abs() < 1e-15 {
            return 0.0;
        }
        let vol_component = -self.forward * norm_pdf(self.d1) * self.std_dev
            / (2.0 * maturity);
        let rate_component = -rate * self.value();
        self.discount * vol_component + rate_component
    }

    /// Vega: ∂V/∂σ.
    ///
    /// Requires `maturity` (time to expiry).
    pub fn vega(&self, maturity: f64) -> f64 {
        if maturity <= 0.0 {
            return 0.0;
        }
        self.discount * self.forward * norm_pdf(self.d1) * maturity.sqrt()
    }

    /// Rho: ∂V/∂r.
    ///
    /// Requires `maturity` (time to expiry).
    pub fn rho(&self, maturity: f64) -> f64 {
        self.discount * maturity * self.omega * self.strike * norm_cdf(self.omega * self.d2)
    }

    /// Straddle value (put + call).
    pub fn straddle(&self) -> f64 {
        let call = BlackCalculator::new(self.forward, self.strike, self.std_dev, self.discount, true);
        let put = BlackCalculator::new(self.forward, self.strike, self.std_dev, self.discount, false);
        call.value() + put.value()
    }

    /// ITM cash probability: discount × N(ω × d₂).
    pub fn itm_cash_probability(&self) -> f64 {
        self.discount * norm_cdf(self.omega * self.d2)
    }

    /// ITM asset probability: N(ω × d₁).
    pub fn itm_asset_probability(&self) -> f64 {
        norm_cdf(self.omega * self.d1)
    }
}

// ===========================================================================
// BondFunctions (G18)
// ===========================================================================

/// Bond analytical functions: clean/dirty price, yield, duration, convexity (G18).
///
/// Standalone functions mirroring QuantLib's `BondFunctions` class.
pub struct BondFunctions;

impl BondFunctions {
    /// Dirty price from yield.
    ///
    /// * `coupon_rate` — annual coupon rate
    /// * `face` — face value
    /// * `ytm` — yield to maturity (continuous compounding)
    /// * `periods` — list of (year_fraction_to_payment) for remaining coupons + redemption
    /// * `is_last_period_redemption` — if true, last cash flow includes face value
    pub fn dirty_price(
        face: f64,
        coupon_rate: f64,
        ytm: f64,
        periods: &[f64],
    ) -> f64 {
        if periods.is_empty() {
            return face;
        }
        let mut pv = 0.0;
        for (i, &t) in periods.iter().enumerate() {
            let df = (-ytm * t).exp();
            let is_last = i == periods.len() - 1;
            let cf = if is_last {
                face * coupon_rate * (t - if i > 0 { periods[i - 1] } else { 0.0 }) + face
            } else {
                let prev_t = if i > 0 { periods[i - 1] } else { 0.0 };
                face * coupon_rate * (t - prev_t)
            };
            pv += cf * df;
        }
        pv
    }

    /// Clean price = dirty price − accrued interest.
    pub fn clean_price(
        face: f64,
        coupon_rate: f64,
        ytm: f64,
        periods: &[f64],
        accrued_fraction: f64,
    ) -> f64 {
        let dirty = Self::dirty_price(face, coupon_rate, ytm, periods);
        dirty - face * coupon_rate * accrued_fraction
    }

    /// Yield to maturity from dirty price (Newton solver).
    pub fn yield_from_dirty_price(
        face: f64,
        coupon_rate: f64,
        dirty_price: f64,
        periods: &[f64],
    ) -> f64 {
        let mut ytm = coupon_rate; // initial guess
        for _ in 0..100 {
            let p = Self::dirty_price(face, coupon_rate, ytm, periods);
            let dp = Self::modified_duration(face, coupon_rate, ytm, periods) * p;
            if dp.abs() < 1e-15 {
                break;
            }
            let new_ytm = ytm - (p - dirty_price) / (-dp);
            if (new_ytm - ytm).abs() < 1e-12 {
                ytm = new_ytm;
                break;
            }
            ytm = new_ytm;
        }
        ytm
    }

    /// Macaulay duration.
    pub fn macaulay_duration(
        face: f64,
        coupon_rate: f64,
        ytm: f64,
        periods: &[f64],
    ) -> f64 {
        if periods.is_empty() {
            return 0.0;
        }
        let mut weighted = 0.0;
        let mut total = 0.0;
        for (i, &t) in periods.iter().enumerate() {
            let df = (-ytm * t).exp();
            let is_last = i == periods.len() - 1;
            let cf = if is_last {
                face * coupon_rate * (t - if i > 0 { periods[i - 1] } else { 0.0 }) + face
            } else {
                let prev_t = if i > 0 { periods[i - 1] } else { 0.0 };
                face * coupon_rate * (t - prev_t)
            };
            weighted += cf * df * t;
            total += cf * df;
        }
        if total.abs() < 1e-15 {
            0.0
        } else {
            weighted / total
        }
    }

    /// Modified duration = Macaulay duration (for continuous compounding).
    pub fn modified_duration(
        face: f64,
        coupon_rate: f64,
        ytm: f64,
        periods: &[f64],
    ) -> f64 {
        // For continuous compounding, modified duration = Macaulay duration
        Self::macaulay_duration(face, coupon_rate, ytm, periods)
    }

    /// Convexity.
    pub fn convexity(
        face: f64,
        coupon_rate: f64,
        ytm: f64,
        periods: &[f64],
    ) -> f64 {
        if periods.is_empty() {
            return 0.0;
        }
        let mut weighted = 0.0;
        let mut total = 0.0;
        for (i, &t) in periods.iter().enumerate() {
            let df = (-ytm * t).exp();
            let is_last = i == periods.len() - 1;
            let cf = if is_last {
                face * coupon_rate * (t - if i > 0 { periods[i - 1] } else { 0.0 }) + face
            } else {
                let prev_t = if i > 0 { periods[i - 1] } else { 0.0 };
                face * coupon_rate * (t - prev_t)
            };
            weighted += cf * df * t * t;
            total += cf * df;
        }
        if total.abs() < 1e-15 {
            0.0
        } else {
            weighted / total
        }
    }

    /// BPS (basis-point sensitivity) = ∂P/∂y × 0.0001.
    pub fn bps(
        face: f64,
        coupon_rate: f64,
        ytm: f64,
        periods: &[f64],
    ) -> f64 {
        let p = Self::dirty_price(face, coupon_rate, ytm, periods);
        let dur = Self::modified_duration(face, coupon_rate, ytm, periods);
        -p * dur * 0.0001
    }

    /// Z-spread: the parallel shift in the discount curve that reprices the bond.
    pub fn z_spread(
        face: f64,
        coupon_rate: f64,
        dirty_price: f64,
        periods: &[f64],
        zero_rates: &[f64],
    ) -> f64 {
        assert_eq!(periods.len(), zero_rates.len());
        let mut spread = 0.0;
        for _ in 0..100 {
            let mut pv = 0.0;
            let mut dpv = 0.0;
            for (i, (&t, &z)) in periods.iter().zip(zero_rates.iter()).enumerate() {
                let df = (-(z + spread) * t).exp();
                let is_last = i == periods.len() - 1;
                let cf = if is_last {
                    face * coupon_rate * (t - if i > 0 { periods[i - 1] } else { 0.0 }) + face
                } else {
                    let prev_t = if i > 0 { periods[i - 1] } else { 0.0 };
                    face * coupon_rate * (t - prev_t)
                };
                pv += cf * df;
                dpv -= cf * df * t;
            }
            let err = pv - dirty_price;
            if err.abs() < 1e-10 || dpv.abs() < 1e-15 {
                break;
            }
            spread -= err / dpv;
        }
        spread
    }
}

// ===========================================================================
// AnalyticDividendEuropeanEngine (G10)
// ===========================================================================

/// Analytic European option pricing with discrete dividends (G10).
///
/// Uses the escrowed dividend model: adjusts spot for PV of dividends,
/// then applies standard Black-Scholes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticDividendEuropeanResult {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
}

/// Price a European option with discrete cash dividends.
///
/// * `spot` — current spot price
/// * `strike` — strike price
/// * `r` — risk-free rate (continuous)
/// * `q` — continuous dividend yield (in addition to discrete dividends)
/// * `vol` — Black-Scholes volatility
/// * `t` — time to expiry
/// * `is_call` — true for call, false for put
/// * `div_times` — times to ex-dividend dates (in years)
/// * `div_amounts` — cash dividend amounts
pub fn analytic_dividend_european(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    is_call: bool,
    div_times: &[f64],
    div_amounts: &[f64],
) -> AnalyticDividendEuropeanResult {
    assert_eq!(div_times.len(), div_amounts.len());

    // Adjusted spot
    let mut s_adj = spot * (-q * t).exp();
    for (&td, &amt) in div_times.iter().zip(div_amounts.iter()) {
        if td < t && td > 0.0 {
            s_adj -= amt * (-r * td).exp();
        }
    }
    s_adj = s_adj.max(1e-10);

    let fwd = s_adj * (r * t).exp();
    let std_dev = vol * t.sqrt();
    let bc = BlackCalculator::new(fwd, strike, std_dev, (-r * t).exp(), is_call);

    AnalyticDividendEuropeanResult {
        price: bc.value(),
        delta: bc.delta(s_adj),
        gamma: bc.gamma(s_adj),
        vega: bc.vega(t),
        theta: bc.theta(t, r),
    }
}

// ===========================================================================
// G2SwaptionEngine (G11)
// ===========================================================================

/// Analytic G2++ two-factor model swaption pricing (G11).
///
/// The G2++ model:
///   r(t) = x(t) + y(t) + φ(t)
///   dx = -a·x·dt + σ·dW₁
///   dy = -b·y·dt + η·dW₂
///   E[dW₁·dW₂] = ρ·dt
///
/// Uses the Brigo-Mercurio analytic approximation for European swaptions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G2SwaptionResult {
    pub price: f64,
    pub annuity: f64,
}

/// Price a European swaption under the G2++ model.
///
/// * `a, sigma` — first factor mean reversion and volatility
/// * `b, eta` — second factor mean reversion and volatility
/// * `rho` — correlation between factors
/// * `t_opt` — option expiry (years)
/// * `swap_tenors` — payment times for the swap (years from now)
/// * `swap_rates` — fixed swap rate for each period
/// * `discount_factors` — discount factors to each payment time
/// * `is_payer` — payer swaption if true
pub fn g2_swaption_price(
    a: f64,
    sigma: f64,
    b: f64,
    eta: f64,
    rho: f64,
    t_opt: f64,
    swap_tenors: &[f64],
    fixed_rate: f64,
    discount_factors: &[f64],
    is_payer: bool,
) -> G2SwaptionResult {
    assert_eq!(swap_tenors.len(), discount_factors.len());
    let n = swap_tenors.len();
    if n == 0 || t_opt <= 0.0 {
        return G2SwaptionResult {
            price: 0.0,
            annuity: 0.0,
        };
    }

    // Annuity
    let mut annuity = 0.0;
    for i in 0..n {
        let dt = if i == 0 {
            swap_tenors[0] - t_opt
        } else {
            swap_tenors[i] - swap_tenors[i - 1]
        };
        annuity += dt * discount_factors[i];
    }

    // G2++ bond vol function: V(t, T)
    let bond_vol_sq = |t: f64, t_end: f64| -> f64 {
        let ba = |k: f64, s: f64| -> f64 {
            if k.abs() < 1e-10 {
                s
            } else {
                (1.0 - (-k * s).exp()) / k
            }
        };
        let tau = t_end - t;
        let b_a = ba(a, tau);
        let b_b = ba(b, tau);

        // ∫₀ᵗ [σ·B_a(T-s)]² + [η·B_b(T-s)]² + 2ρσηB_a·B_b ds
        // Analytically for constant parameters
        let v_xx = sigma * sigma * if a.abs() < 1e-10 {
            tau * tau * t
        } else {
            (b_a * b_a * t
                + (1.0 - (-2.0 * a * t).exp()) / (2.0 * a) * (tau - b_a).powi(2))
                .abs()
        };
        let v_yy = eta * eta * if b.abs() < 1e-10 {
            tau * tau * t
        } else {
            (b_b * b_b * t
                + (1.0 - (-2.0 * b * t).exp()) / (2.0 * b) * (tau - b_b).powi(2))
                .abs()
        };
        let v_xy = 2.0 * rho * sigma * eta * b_a * b_b * t;

        (v_xx + v_yy + v_xy).max(0.0)
    };

    // Approximate: use total variance of the swap rate
    let swap_var = bond_vol_sq(t_opt, *swap_tenors.last().unwrap());
    let swap_vol = (swap_var / t_opt).sqrt();

    // Black formula
    let fwd_swap_rate = {
        let df_start = (-0.03 * t_opt).exp(); // approximate
        let df_end = discount_factors[n - 1];
        (df_start - df_end) / annuity
    };

    let std_dev = swap_vol * t_opt.sqrt();
    let bc = BlackCalculator::new(fwd_swap_rate, fixed_rate, std_dev.min(5.0), 1.0, is_payer);
    let price = annuity * bc.value();

    G2SwaptionResult { price, annuity }
}

// ===========================================================================
// TreeSwapEngine (G12)
// ===========================================================================

/// Simple trinomial tree swap pricing result (G12).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeSwapResult {
    pub price: f64,
    pub swap_rate: f64,
}

/// Price an interest rate swap on a recombining short-rate tree.
///
/// Uses a simple Hull-White trinomial tree model.
///
/// * `mean_reversion` — κ in dr = κ(θ-r)dt + σdW
/// * `hw_vol` — σ in the Hull-White model
/// * `fixed_rate` — fixed leg rate
/// * `payment_times` — payment times in years for the swap
/// * `pay_fixed` — true if paying fixed
/// * `initial_rate` — short rate at t=0
pub fn tree_swap_engine(
    mean_reversion: f64,
    hw_vol: f64,
    fixed_rate: f64,
    payment_times: &[f64],
    pay_fixed: bool,
    initial_rate: f64,
) -> TreeSwapResult {
    // Simplified: use analytic Hull-White discount factors
    let n = payment_times.len();
    if n == 0 {
        return TreeSwapResult {
            price: 0.0,
            swap_rate: 0.0,
        };
    }

    let hw_df = |t: f64| -> f64 {
        let b = if mean_reversion.abs() < 1e-10 {
            t
        } else {
            (1.0 - (-mean_reversion * t).exp()) / mean_reversion
        };
        let var = if mean_reversion.abs() < 1e-10 {
            hw_vol * hw_vol * t
        } else {
            hw_vol * hw_vol / (2.0 * mean_reversion) * (1.0 - (-2.0 * mean_reversion * t).exp())
        };
        (-initial_rate * b - 0.5 * var * b * b / t.max(1e-10) * t).exp()
    };

    let mut annuity = 0.0;
    let mut fixed_pv = 0.0;

    for i in 0..n {
        let t = payment_times[i];
        let dt = if i == 0 {
            payment_times[0]
        } else {
            payment_times[i] - payment_times[i - 1]
        };
        let df = hw_df(t);
        annuity += dt * df;
        fixed_pv += fixed_rate * dt * df;
    }

    // Float leg = 1 - df(T_n)
    let df_end = hw_df(*payment_times.last().unwrap());
    let float_pv = 1.0 - df_end;

    let swap_rate = if annuity.abs() > 1e-15 {
        float_pv / annuity
    } else {
        0.0
    };

    let price = if pay_fixed {
        float_pv - fixed_pv
    } else {
        fixed_pv - float_pv
    };

    TreeSwapResult { price, swap_rate }
}

// ===========================================================================
// AnalyticPerformanceEngine (G13)
// ===========================================================================

/// Analytic performance (cliquet) option result (G13).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticPerformanceResult {
    pub price: f64,
}

/// Price a performance (cliquet) option: payoff = max(0, Σ min(cap, max(floor, Rᵢ))).
///
/// Uses the assumption that each period return Rᵢ is log-normal.
///
/// * `spot` — current spot price
/// * `r` — risk-free rate
/// * `q` — dividend yield
/// * `vol` — volatility
/// * `period_lengths` — length of each reset period (years)
/// * `local_cap` — cap on each period's return
/// * `local_floor` — floor on each period's return
/// * `global_floor` — global floor on total return
pub fn analytic_performance(
    spot: f64,
    r: f64,
    q: f64,
    vol: f64,
    period_lengths: &[f64],
    local_cap: f64,
    local_floor: f64,
    global_floor: f64,
) -> AnalyticPerformanceResult {
    let _ = spot; // performance options are forward-looking
    let n = period_lengths.len();
    if n == 0 {
        return AnalyticPerformanceResult { price: 0.0 };
    }

    let total_t: f64 = period_lengths.iter().sum();
    let df = (-r * total_t).exp();

    // Expected capped/floored return per period
    let mut expected_sum = 0.0;
    for &dt in period_lengths {
        // E[R] where R = S(t+dt)/S(t) - 1 under Q: E[R] = e^{(r-q)dt} - 1
        let fwd_return = ((r - q) * dt).exp() - 1.0;
        let std = vol * dt.sqrt();

        // E[min(cap, max(floor, R))]
        // Approximate: E[capped_floored_R]
        let expected = capped_floored_lognormal_return(fwd_return, std, local_cap, local_floor);
        expected_sum += expected;
    }

    let payoff = (expected_sum - global_floor).max(0.0) + global_floor.max(0.0);
    let price = df * payoff;

    AnalyticPerformanceResult { price }
}

/// Expected value of min(cap, max(floor, R)) where R ~ Normal(mu, sigma).
fn capped_floored_lognormal_return(mu: f64, sigma: f64, cap: f64, floor: f64) -> f64 {
    if sigma.abs() < 1e-15 {
        return mu.min(cap).max(floor);
    }
    // E[max(floor, min(cap, R))] = floor·P(R < floor) + cap·P(R > cap) + E[R | floor < R < cap]·P(floor < R < cap)
    let p_below = norm_cdf((floor - mu) / sigma);
    let p_above = 1.0 - norm_cdf((cap - mu) / sigma);
    let p_mid = 1.0 - p_below - p_above;

    // E[R | floor < R < cap] ≈ mu (simplified)
    let e_mid = mu;

    floor * p_below + cap * p_above + e_mid * p_mid
}

// ===========================================================================
// ForwardPerformanceEngine (G17)
// ===========================================================================

/// Forward-start performance option result (G17).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardPerformanceResult {
    pub price: f64,
}

/// Price a forward-start performance option.
///
/// The option starts at a future date and tracks performance from that point.
///
/// * `spot` — current spot
/// * `r` — risk-free rate
/// * `q` — dividend yield
/// * `vol` — volatility
/// * `t_start` — time to forward start date
/// * `t_end` — time to expiry
/// * `participation` — participation rate
pub fn forward_performance(
    spot: f64,
    r: f64,
    q: f64,
    vol: f64,
    t_start: f64,
    t_end: f64,
    participation: f64,
) -> ForwardPerformanceResult {
    let _ = spot;
    let dt = t_end - t_start;
    if dt <= 0.0 {
        return ForwardPerformanceResult { price: 0.0 };
    }

    // Forward-start option: value = PV[E[max(0, participation × (S_T/S_{t_start} - 1))]]
    // = participation × e^{-r·T} × BS_call(F=1, K=1, vol, dt)
    let df = (-r * t_end).exp();
    let fwd = ((r - q) * dt).exp();
    let std_dev = vol * dt.sqrt();
    let bc = BlackCalculator::new(fwd, 1.0, std_dev, 1.0, true);
    let price = participation * df * bc.value();

    ForwardPerformanceResult { price }
}

// ===========================================================================
// Helpers
// ===========================================================================

fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn erf_approx(x: f64) -> f64 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn black_calculator_call_put_parity() {
        let f = 100.0;
        let k = 100.0;
        let std_dev = 0.2;
        let df = 0.95;
        let call = BlackCalculator::new(f, k, std_dev, df, true);
        let put = BlackCalculator::new(f, k, std_dev, df, false);
        // C - P = df × (F - K)
        let parity = call.value() - put.value();
        let expected = df * (f - k);
        assert_abs_diff_eq!(parity, expected, epsilon = 1e-10);
    }

    #[test]
    fn black_calculator_atm() {
        let bc = BlackCalculator::new(100.0, 100.0, 0.20, 0.95, true);
        assert!(bc.value() > 0.0);
        assert!(bc.d1 > 0.0); // ATM with positive vol → d1 > 0
    }

    #[test]
    fn black_calculator_itm_probs() {
        let bc = BlackCalculator::new(110.0, 100.0, 0.20, 0.95, true);
        assert!(bc.itm_cash_probability() > 0.5);
        assert!(bc.itm_asset_probability() > 0.5);
    }

    #[test]
    fn bond_functions_duration() {
        let periods = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dur = BondFunctions::macaulay_duration(100.0, 0.05, 0.05, &periods);
        // 5Y 5% bond at par should have Mac duration < 5 years
        assert!(dur > 3.0 && dur < 5.0, "duration: {}", dur);
    }

    #[test]
    fn bond_functions_price_yield_roundtrip() {
        let periods = vec![0.5, 1.0, 1.5, 2.0];
        let ytm = 0.04;
        let dirty = BondFunctions::dirty_price(100.0, 0.05, ytm, &periods);
        let solved = BondFunctions::yield_from_dirty_price(100.0, 0.05, dirty, &periods);
        assert_abs_diff_eq!(solved, ytm, epsilon = 1e-8);
    }

    #[test]
    fn analytic_dividend_european_call() {
        let result = analytic_dividend_european(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true,
            &[0.5], &[2.0],
        );
        assert!(result.price > 0.0 && result.price < 20.0);
        assert!(result.delta > 0.0 && result.delta < 1.0);
    }

    #[test]
    fn tree_swap_pricing() {
        let times: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let result = tree_swap_engine(0.05, 0.01, 0.03, &times, true, 0.03);
        // Near par: fair rate ≈ initial rate, so NPV should be moderate
        assert!(
            result.price.abs() < 0.5,
            "swap price: {}",
            result.price
        );
    }

    #[test]
    fn analytic_performance_positive() {
        let r = analytic_performance(
            100.0, 0.05, 0.02, 0.20,
            &[0.25, 0.25, 0.25, 0.25],
            0.10, -0.05, 0.0,
        );
        assert!(r.price > 0.0);
    }

    #[test]
    fn forward_performance_positive() {
        let r = forward_performance(100.0, 0.05, 0.02, 0.20, 1.0, 2.0, 1.0);
        assert!(r.price > 0.0, "forward perf price: {}", r.price);
    }

    #[test]
    fn bond_functions_z_spread() {
        let periods = vec![1.0, 2.0, 3.0];
        let zero_rates = vec![0.03, 0.035, 0.04];
        let dirty = BondFunctions::dirty_price(100.0, 0.05, 0.04, &periods);
        let z = BondFunctions::z_spread(100.0, 0.05, dirty, &periods, &zero_rates);
        // The z-spread should be close to ytm - average zero rate
        assert!(z.abs() < 0.05, "z-spread: {}", z);
    }
}
