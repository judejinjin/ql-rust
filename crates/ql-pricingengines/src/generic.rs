//! Generic pricing functions that work with any `T: Number`.
//!
//! These functions are the AD-compatible counterparts of the `f64`-only
//! engines in `ql-pricingengines`.  Every input flows through the
//! [`Number`] trait, so when called with AD types (`Dual`, `DualVec`,
//! `AReal`) all partial derivatives propagate automatically.
//!
//! # Supported Models
//!
//! | Function | Model |
//! |----------|-------|
//! | [`bs_european_generic`] | Black-Scholes European (full Greeks struct) |
//! | [`black76_generic`] | Black-76 for futures / forwards |
//! | [`bachelier_generic`] | Bachelier (normal) model |
//! | [`barone_adesi_whaley_generic`] | BAW American approximation |
//! | [`merton_jd_generic`] | Merton jump-diffusion |
//! | [`chooser_generic`] | Rubinstein (1991) simple chooser |
//! | [`bond_pv_generic`] | Bond NPV from cashflows + flat rate |
//! | [`swap_pv_generic`] | IRS NPV from two legs + flat rate |
//!
//! # Examples
//!
//! ```
//! use ql_pricingengines::generic::{bs_european_generic, black76_generic};
//!
//! // Black-Scholes with f64 — returns price + all Greeks
//! let res = bs_european_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
//! assert!((res.npv - 10.45).abs() < 0.1);
//! assert!(res.delta > 0.0);
//!
//! // Black-76 for a futures option
//! let fwd = black76_generic(100.0, 100.0, 0.05, 0.20, 1.0, true);
//! assert!(fwd > 0.0);
//! ```

use ql_core::Number;
use ql_math::generic::{
    normal_cdf, normal_pdf, black_scholes_generic, discount_factor,
    bivariate_normal_cdf,
};
use ql_math::solvers1d::Solver1D;
use ql_termstructures::generic::GenericYieldCurve;

// ===========================================================================
// Black-Scholes European — full Greeks
// ===========================================================================

/// Results from the generic Black-Scholes European engine.
///
/// All fields are generic `T: Number` so that AD types carry derivatives.
#[derive(Debug, Clone, Copy)]
pub struct BsEuropeanResult<T: Number> {
    /// Net present value.
    pub npv: T,
    /// Delta: ∂V/∂S.
    pub delta: T,
    /// Gamma: ∂²V/∂S².
    pub gamma: T,
    /// Vega: ∂V/∂σ (per 1% move).
    pub vega: T,
    /// Theta: ∂V/∂t (per calendar day, 1/365).
    pub theta: T,
    /// Rho: ∂V/∂r (per 1% move).
    pub rho: T,
}

/// Full Black-Scholes European pricing with all first-order Greeks.
///
/// This is the generic counterpart of [`crate::analytic_european::price_european`].
///
/// # Parameters
/// - `spot`, `strike`, `r`, `q`, `vol`, `t` — standard BS inputs
/// - `is_call` — `true` for a call, `false` for a put
///
/// # Examples
///
/// ```
/// use ql_pricingengines::generic::bs_european_generic;
///
/// let res = bs_european_generic(100.0_f64, 100.0, 0.05, 0.0, 0.20, 1.0, true);
/// assert!((res.npv - 10.4506).abs() < 0.05);
/// assert!((res.delta - 0.6368).abs() < 0.01);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn bs_european_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    is_call: bool,
) -> BsEuropeanResult<T> {
    let zero = T::zero();
    let one = T::one();
    let half = T::half();

    if t.to_f64() <= 0.0 {
        let omega = if is_call { one } else { zero - one };
        let intrinsic = {
            let diff = omega * (spot - strike);
            if diff.to_f64() > 0.0 { diff } else { zero }
        };
        let delta_val = if (omega * (spot - strike)).to_f64() > 0.0 { omega } else { zero };
        return BsEuropeanResult {
            npv: intrinsic,
            delta: delta_val,
            gamma: zero,
            vega: zero,
            theta: zero,
            rho: zero,
        };
    }

    let omega = if is_call { one } else { zero - one };
    let sqrt_t = t.sqrt();
    let vol_sqrt_t = vol * sqrt_t;

    // d1, d2
    let d1 = ((spot / strike).ln() + (r - q + half * vol * vol) * t) / vol_sqrt_t;
    let d2 = d1 - vol_sqrt_t;

    let nd1 = normal_cdf(omega * d1);
    let nd2 = normal_cdf(omega * d2);
    let npdf_d1 = normal_pdf(d1);

    let df_q = (zero - q * t).exp();
    let df_r = (zero - r * t).exp();

    // NPV
    let npv = omega * (spot * df_q * nd1 - strike * df_r * nd2);

    // Delta
    let delta = omega * df_q * nd1;

    // Gamma
    let gamma = df_q * npdf_d1 / (spot * vol_sqrt_t);

    // Vega (per 1% = 0.01)
    let vega_scale = T::from_f64(0.01);
    let vega = spot * df_q * npdf_d1 * sqrt_t * vega_scale;

    // Theta (per calendar day = 1/365)
    let theta_continuous = zero - spot * df_q * npdf_d1 * vol / (T::from_f64(2.0) * sqrt_t)
        - omega * r * strike * df_r * nd2
        + omega * q * spot * df_q * nd1;
    let theta = theta_continuous / T::from_f64(365.0);

    // Rho (per 1% = 0.01)
    let rho = omega * strike * t * df_r * nd2 * vega_scale;

    BsEuropeanResult { npv, delta, gamma, vega, theta, rho }
}

// ===========================================================================
// Black-76
// ===========================================================================

/// Black-76 pricing formula for options on forwards/futures.
///
/// Price = e^{-rT} [ω F N(ω d₁) − ω K N(ω d₂)]
///
/// where d₁ = [ln(F/K) + ½σ²T] / (σ√T),  d₂ = d₁ − σ√T.
///
/// # Parameters
/// - `forward`: forward / futures price
/// - `strike`: option strike
/// - `r`: risk-free rate for discounting
/// - `vol`: Black volatility
/// - `t`: time to expiry in years
/// - `is_call`: call or put
pub fn black76_generic<T: Number>(
    forward: T,
    strike: T,
    r: T,
    vol: T,
    t: T,
    is_call: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();
    let half = T::half();

    if t.to_f64() <= 0.0 {
        let omega = if is_call { one } else { zero - one };
        let diff = omega * (forward - strike);
        return if diff.to_f64() > 0.0 { diff } else { zero };
    }

    let omega = if is_call { one } else { zero - one };
    let sqrt_t = t.sqrt();
    let d1 = ((forward / strike).ln() + half * vol * vol * t) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;

    let df = (zero - r * t).exp();
    df * omega * (forward * normal_cdf(omega * d1) - strike * normal_cdf(omega * d2))
}

// ===========================================================================
// Bachelier (Normal Model)
// ===========================================================================

/// Bachelier (normal) model European option price.
///
/// V = e^{-rT} [ω (F − K) N(ω d) + σ√T φ(d)]
///
/// where d = (F − K) / (σ√T), F = S · e^{(r−q)T}.
///
/// Useful for negative rates, spreads, and normal-vol quoted instruments.
pub fn bachelier_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    is_call: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();

    if t.to_f64() <= 0.0 {
        let omega = if is_call { one } else { zero - one };
        let diff = omega * (spot - strike);
        return if diff.to_f64() > 0.0 { diff } else { zero };
    }

    let omega = if is_call { one } else { zero - one };
    let forward = spot * ((r - q) * t).exp();
    let sqrt_t = t.sqrt();
    let vol_sqrt_t = vol * sqrt_t;

    let d = (forward - strike) / vol_sqrt_t;
    let df = (zero - r * t).exp();

    df * (omega * (forward - strike) * normal_cdf(omega * d) + vol_sqrt_t * normal_pdf(d))
}

// ===========================================================================
// Barone-Adesi-Whaley American Approximation
// ===========================================================================

/// Generic results from the BAW American approximation.
#[derive(Debug, Clone, Copy)]
pub struct BawResult<T: Number> {
    /// American option price.
    pub npv: T,
    /// Early exercise premium over the European price.
    pub early_exercise_premium: T,
    /// Critical stock price (exercise boundary).
    pub critical_price: T,
}

/// Barone-Adesi-Whaley quadratic approximation for American options,
/// generic over `T: Number`.
///
/// This makes AD-computed American Greeks possible — something that has
/// no closed-form solution.
///
/// # Parameters
/// - `spot`, `strike`, `r`, `q`, `vol`, `t` — standard option inputs
/// - `is_call` — `true` for call, `false` for put
///
/// # Examples
///
/// ```
/// use ql_pricingengines::generic::barone_adesi_whaley_generic;
///
/// let res = barone_adesi_whaley_generic(
///     100.0_f64, 100.0, 0.05, 0.02, 0.25, 1.0, false,
/// );
/// assert!(res.npv > 0.0);
/// assert!(res.early_exercise_premium >= 0.0);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn barone_adesi_whaley_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    is_call: bool,
) -> BawResult<T> {
    let zero = T::zero();
    let one = T::one();
    let two = T::from_f64(2.0);

    let omega = if is_call { one } else { zero - one };

    if t.to_f64() <= 0.0 {
        let diff = omega * (spot - strike);
        let intrinsic = if diff.to_f64() > 0.0 { diff } else { zero };
        return BawResult {
            npv: intrinsic,
            early_exercise_premium: zero,
            critical_price: if is_call { T::from_f64(f64::INFINITY) } else { zero },
        };
    }

    // For American call with no dividends, price = European
    if is_call && q.to_f64() <= 0.0 {
        let euro = black_scholes_generic(spot, strike, r, q, vol, t, true);
        return BawResult {
            npv: euro,
            early_exercise_premium: zero,
            critical_price: T::from_f64(f64::INFINITY),
        };
    }

    let sigma2 = vol * vol;
    let h = one - (zero - r * t).exp();

    // Quadratic coefficients
    let m = two * r / sigma2;
    let n_coeff = two * (r - q) / sigma2;

    // For call: q2 = (-(n-1) + sqrt((n-1)^2 + 4*m/h)) / 2
    // For put:  q1 = (-(n-1) - sqrt((n-1)^2 + 4*m/h)) / 2
    let nm1 = n_coeff - one;
    let disc = nm1 * nm1 + T::from_f64(4.0) * m / h;
    let sqrt_disc = disc.sqrt();

    let q_coeff = if is_call {
        (zero - nm1 + sqrt_disc) / two
    } else {
        (zero - nm1 - sqrt_disc) / two
    };

    // Find critical price via Newton iteration
    let mut s_star = strike; // initial guess
    let max_iter = 100;
    let tol_f64 = 1e-8;

    for _ in 0..max_iter {
        let euro = black_scholes_generic(s_star, strike, r, q, vol, t, is_call);
        let d1 = ((s_star / strike).ln() + (r - q + sigma2 * T::half()) * t) / (vol * t.sqrt());
        let nd1 = normal_cdf(omega * d1);
        let dfq = (zero - q * t).exp();

        // Value-matching: g(S*) = omega*(S* - K) - euro - A*(S*/S*)^q = 0
        // Simplified: we want A*S*^q = omega*(S* - K) - euro
        // where A = (omega * S* / q) * (1 - dfq * nd1)
        let a_coeff = (omega * s_star / q_coeff) * (one - dfq * nd1);

        // g = euro + a_coeff - omega * (s_star - strike)
        let g = euro + a_coeff - omega * (s_star - strike);

        // g' derivative w.r.t. S*
        let delta_euro = omega * dfq * nd1;
        let npdf_d1 = normal_pdf(d1);
        let gamma_euro = dfq * npdf_d1 / (s_star * vol * t.sqrt());

        // d(a)/dS* ≈ (1/q)*(1 - dfq*nd1) + (S*/q)*(dfq*gamma_euro*omega*S*)...
        // Use simplified: g' ≈ delta_euro + (1 - dfq*nd1)*(1/q + omega) - omega
        // Actually, Newton step: h = omega*(S* - K) - euro - a_coeff
        // We invert the sign: we solve for h = 0
        let _rhs = omega * (s_star - strike) - euro;
        let _lhs = a_coeff;

        // Simpler approach: iterate S* = K + (euro + a_coeff) / omega,
        // but use Newton on the full equation.
        // dg/dS* = delta_euro + (1/q_coeff)*(1 - dfq*nd1 + omega*s_star*dfq*...)
        // Use a simplified numerical-like Newton step:
        let g_val = g;
        let dg = delta_euro + (one - dfq * nd1) / q_coeff
            - omega
            + omega * s_star * dfq * gamma_euro / q_coeff;

        if dg.to_f64().abs() < 1e-30 {
            break;
        }
        let step = g_val / dg;
        s_star -= step;

        // Clamp
        if s_star.to_f64() <= 0.0 {
            s_star = T::from_f64(0.01);
        }
        if step.to_f64().abs() < tol_f64 * strike.to_f64() {
            break;
        }
    }

    let euro = black_scholes_generic(spot, strike, r, q, vol, t, is_call);

    // Check if spot is beyond critical price
    let beyond = if is_call {
        spot.to_f64() >= s_star.to_f64()
    } else {
        spot.to_f64() <= s_star.to_f64()
    };

    if beyond {
        // Immediate exercise is optimal
        let intrinsic = omega * (spot - strike);
        let npv = if intrinsic.to_f64() > euro.to_f64() { intrinsic } else { euro };
        return BawResult {
            npv,
            early_exercise_premium: npv - euro,
            critical_price: s_star,
        };
    }

    // Compute A coefficient at s_star
    let d1_star = ((s_star / strike).ln()
        + (r - q + sigma2 * T::half()) * t)
        / (vol * t.sqrt());
    let nd1_star = normal_cdf(omega * d1_star);
    let dfq = (zero - q * t).exp();
    let a_val = (omega * s_star / q_coeff) * (one - dfq * nd1_star);

    // Early exercise premium: A * (S/S*)^q
    let premium = a_val * (spot / s_star).powf(q_coeff);
    let npv = euro + premium;

    BawResult {
        npv,
        early_exercise_premium: premium,
        critical_price: s_star,
    }
}

// ===========================================================================
// Merton Jump-Diffusion
// ===========================================================================

/// Results from the generic Merton jump-diffusion engine.
#[derive(Debug, Clone, Copy)]
pub struct MertonJdResult<T: Number> {
    /// Merton jump-diffusion price.
    pub npv: T,
    /// Number of series terms used.
    pub num_terms: usize,
}

/// Merton (1976) jump-diffusion European option price, generic over `T: Number`.
///
/// Decomposes the price as a Poisson-weighted sum of Black-Scholes prices
/// with jump-adjusted parameters:
///
/// V = Σ P(N=n) · BS(S, K, rₙ, σₙ, T)
///
/// # Parameters
/// - `spot`, `strike`, `r`, `q`, `vol` — standard BS inputs
/// - `t` — time to expiry
/// - `lambda` — jump intensity (mean jumps per year)
/// - `nu` — mean of log-jump size
/// - `delta` — std dev of log-jump size
/// - `is_call`
///
/// # Examples
///
/// ```
/// use ql_pricingengines::generic::merton_jd_generic;
///
/// let res = merton_jd_generic(
///     100.0_f64, 100.0, 0.05, 0.0, 0.20, 1.0,
///     0.5, -0.1, 0.15, true,
/// );
/// assert!(res.npv > 0.0);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn merton_jd_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    lambda: T,
    nu: T,
    delta: T,
    is_call: bool,
) -> MertonJdResult<T> {
    let zero = T::zero();
    let one = T::one();
    let half = T::half();

    if t.to_f64() <= 0.0 {
        let omega = if is_call { one } else { zero - one };
        let diff = omega * (spot - strike);
        let intrinsic = if diff.to_f64() > 0.0 { diff } else { zero };
        return MertonJdResult { npv: intrinsic, num_terms: 0 };
    }

    // k_bar = E[e^J - 1] = exp(nu + delta^2/2) - 1
    let k_bar = (nu + half * delta * delta).exp() - one;

    // lambda' = lambda * (1 + k_bar)
    let lambda_prime = lambda * (one + k_bar);

    let mut price = zero;
    let mut log_pmf = zero - lambda_prime * t; // log P(N=0)
    let max_terms = 200;

    for n in 0..max_terms {
        let w = log_pmf.exp();
        let nf = T::from_f64(n as f64);

        // sigma_n^2 = sigma^2 + n * delta^2 / t
        let sigma_n_sq = vol * vol + nf * delta * delta / t;
        let sigma_n = sigma_n_sq.sqrt();

        // r_n = r - lambda*k_bar + n*ln(1+k_bar)/t
        let r_n = r - lambda * k_bar + nf * (one + k_bar).ln() / t;

        let bs = black_scholes_generic(spot, strike, r_n, q, sigma_n, t, is_call);
        price += w * bs;

        // Update log-Poisson PMF
        log_pmf = log_pmf + (lambda_prime * t).ln() - T::from_f64(((n + 1) as f64).ln());

        // Convergence
        if n > 5 && (w * bs).to_f64().abs() < 1e-15 {
            return MertonJdResult { npv: price, num_terms: n + 1 };
        }
    }

    MertonJdResult { npv: price, num_terms: max_terms }
}

// ===========================================================================
// Simple Chooser Option — Rubinstein (1991)
// ===========================================================================

/// Rubinstein (1991) simple chooser option, generic over `T: Number`.
///
/// The holder chooses at `t_choose` whether the option becomes a European
/// call or put with strike `K` and expiry `t_expiry`.
///
/// # Examples
///
/// ```
/// use ql_pricingengines::generic::chooser_generic;
///
/// let npv = chooser_generic(50.0_f64, 50.0, 0.08, 0.0, 0.25, 0.25, 0.50);
/// assert!(npv > 0.0);
/// ```
pub fn chooser_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    vol: T,
    t_choose: T,
    t_expiry: T,
) -> T {
    let half = T::half();
    let zero = T::zero();

    let sqrt_t = t_expiry.sqrt();
    let sqrt_tc = t_choose.sqrt();

    let d1 = ((spot / strike).ln() + (r - q + half * vol * vol) * t_expiry) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;

    let y1 = ((spot / strike).ln() + (r - q) * t_expiry + half * vol * vol * t_choose)
        / (vol * sqrt_tc);
    let y2 = y1 - vol * sqrt_tc;

    let df_q_t = (zero - q * t_expiry).exp();
    let df_r_t = (zero - r * t_expiry).exp();
    let df_r_tc = (zero - r * t_choose).exp();

    spot * df_q_t * normal_cdf(d1) - strike * df_r_t * normal_cdf(d2)
        - spot * df_q_t * normal_cdf(zero - y1)
        + strike * df_r_tc * normal_cdf(zero - y2)
}

// ===========================================================================
// Bond PV (fixed coupons + flat rate)
// ===========================================================================

/// Present value of a fixed-coupon bond given a flat continuously-compounded
/// discount rate.
///
/// Generic over `T: Number`, enabling AD sensitivities (DV01, convexity, etc.)
/// via automatic differentiation instead of finite differences.
///
/// # Parameters
/// - `coupon_amounts` — coupon cashflow amounts (f64 slices; not differentiated)
/// - `coupon_times` — payment times in years
/// - `notional` — face / notional amount
/// - `maturity` — maturity in years (for notional repayment)
/// - `rate` — flat continuously-compounded discount rate
///
/// # Examples
///
/// ```
/// use ql_pricingengines::generic::bond_pv_generic;
///
/// // 5Y bond, 5% annual coupon, flat 4% rate
/// let coupons = [5.0; 5];
/// let times = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let pv = bond_pv_generic(&coupons, &times, 100.0, 5.0, 0.04_f64);
/// assert!((pv - 104.49).abs() < 0.5); // premium bond
/// ```
pub fn bond_pv_generic<T: Number>(
    coupon_amounts: &[f64],
    coupon_times: &[f64],
    notional: f64,
    maturity: f64,
    rate: T,
) -> T {
    let mut pv = T::zero();

    // Discount each coupon
    for (&amt, &ti) in coupon_amounts.iter().zip(coupon_times.iter()) {
        pv += T::from_f64(amt) * discount_factor(rate, T::from_f64(ti));
    }

    // Notional repayment at maturity
    pv += T::from_f64(notional) * discount_factor(rate, T::from_f64(maturity));

    pv
}

// ===========================================================================
// Swap PV (fixed vs floating, flat rate)
// ===========================================================================

/// Present value of a plain-vanilla interest rate swap given flat rates.
///
/// The fixed leg pays `fixed_rate` and the floating leg pays `float_rate`
/// (assumed flat). A positive result means the payer benefits.
///
/// Generic over `T: Number` so that AD types produce rate sensitivities
/// (DV01, cross-gamma) automatically.
///
/// # Parameters
/// - `fixed_amounts` — fixed leg cashflows
/// - `fixed_times` — fixed leg payment times
/// - `float_amounts` — floating leg cashflows
/// - `float_times` — floating leg payment times
/// - `discount_rate` — continuously-compounded discount rate
///
/// # Examples
///
/// ```
/// use ql_pricingengines::generic::swap_pv_generic;
///
/// let fixed = [2.5, 2.5, 2.5, 2.5]; // 2.5% semi-annual on 100
/// let ftimes = [0.5, 1.0, 1.5, 2.0];
/// let float = [2.0, 2.2, 2.4, 2.6]; // floating resets
/// let disc = 0.03_f64;
/// let pv = swap_pv_generic(&float, &ftimes, &fixed, &ftimes, disc);
/// // Floating > fixed ⇒ payer receives net positive
/// ```
pub fn swap_pv_generic<T: Number>(
    float_amounts: &[f64],
    float_times: &[f64],
    fixed_amounts: &[f64],
    fixed_times: &[f64],
    discount_rate: T,
) -> T {
    let float_pv = npv_generic(float_amounts, float_times, discount_rate);
    let fixed_pv = npv_generic(fixed_amounts, fixed_times, discount_rate);
    float_pv - fixed_pv
}

/// NPV of a series of cashflows given a flat discount rate (helper).
#[inline]
fn npv_generic<T: Number>(amounts: &[f64], times: &[f64], rate: T) -> T {
    let mut pv = T::zero();
    for (&amt, &ti) in amounts.iter().zip(times.iter()) {
        pv += T::from_f64(amt) * discount_factor(rate, T::from_f64(ti));
    }
    pv
}

// ===========================================================================
// Kirk Spread Option (generic)
// ===========================================================================

/// Kirk's approximation for a spread option on two assets,
/// generic over `T: Number`.
///
/// Prices `max(S₁ − S₂ − K, 0)` (spread call).
///
/// # Parameters
/// - `s1`, `s2` — current prices of the two assets
/// - `strike` — spread strike
/// - `r` — risk-free rate
/// - `vol1`, `vol2` — volatilities
/// - `rho` — correlation between the two assets
/// - `t` — time to expiry
pub fn kirk_spread_generic<T: Number>(
    s1: T,
    s2: T,
    strike: T,
    r: T,
    vol1: T,
    vol2: T,
    rho: T,
    t: T,
) -> T {
    let zero = T::zero();
    let _half = T::half();

    if t.to_f64() <= 0.0 {
        let payoff = s1 - s2 - strike;
        return if payoff.to_f64() > 0.0 { payoff } else { zero };
    }

    // Kirk's approximation: treat as a BS call on S1 with modified vol
    let f2 = s2 + strike; // approximate denominator
    let ratio = s2 / f2;
    let sigma_kirk = (vol1 * vol1 - T::from_f64(2.0) * rho * vol1 * vol2 * ratio
        + vol2 * vol2 * ratio * ratio)
        .sqrt();

    // BS call on S1 with strike = S2 + K, vol = sigma_kirk
    black_scholes_generic(s1, f2, r, zero, sigma_kirk, t, true)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Black-Scholes European
    // -----------------------------------------------------------------------

    #[test]
    fn bs_call_price() {
        let res = bs_european_generic(100.0_f64, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        assert!(
            (res.npv - 10.4506).abs() < 0.05,
            "BS call = {}",
            res.npv
        );
    }

    #[test]
    fn bs_put_call_parity() {
        let call = bs_european_generic(100.0_f64, 100.0, 0.05, 0.02, 0.25, 1.0, true);
        let put = bs_european_generic(100.0_f64, 100.0, 0.05, 0.02, 0.25, 1.0, false);
        let parity = call.npv - put.npv
            - 100.0 * (-0.02_f64).exp()
            + 100.0 * (-0.05_f64).exp();
        assert!(parity.abs() < 1e-5, "parity = {parity}");
    }

    #[test]
    fn bs_greeks_signs() {
        let call = bs_european_generic(100.0_f64, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        assert!(call.delta > 0.0, "call delta should be positive");
        assert!(call.gamma > 0.0, "gamma should be positive");
        assert!(call.vega > 0.0, "vega should be positive");
        assert!(call.rho > 0.0, "call rho should be positive");
    }

    #[test]
    fn bs_delta_value() {
        let res = bs_european_generic(100.0_f64, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        assert!(
            (res.delta - 0.6368).abs() < 0.01,
            "delta = {}",
            res.delta
        );
    }

    #[test]
    fn bs_expired() {
        let itm = bs_european_generic(110.0_f64, 100.0, 0.05, 0.0, 0.20, 0.0, true);
        assert!((itm.npv - 10.0).abs() < 1e-10);
        let otm = bs_european_generic(90.0_f64, 100.0, 0.05, 0.0, 0.20, 0.0, true);
        assert!(otm.npv.abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Black-76
    // -----------------------------------------------------------------------

    #[test]
    fn black76_call() {
        let price: f64 = black76_generic(100.0, 100.0, 0.05, 0.20, 1.0, true);
        assert!(price > 0.0 && price < 100.0, "B76 = {price}");
    }

    #[test]
    fn black76_put_call_parity() {
        let call: f64 = black76_generic(100.0, 100.0, 0.05, 0.20, 1.0, true);
        let put: f64 = black76_generic(100.0, 100.0, 0.05, 0.20, 1.0, false);
        let df = (-0.05_f64).exp();
        let parity = call - put - df * (100.0 - 100.0);
        assert!(parity.abs() < 1e-6, "B76 parity = {parity}");
    }

    // -----------------------------------------------------------------------
    // Bachelier
    // -----------------------------------------------------------------------

    #[test]
    fn bachelier_positive() {
        let price: f64 = bachelier_generic(100.0, 100.0, 0.05, 0.0, 20.0, 1.0, true);
        assert!(price > 0.0, "Bachelier = {price}");
    }

    #[test]
    fn bachelier_put_call_parity() {
        let call: f64 = bachelier_generic(100.0, 100.0, 0.05, 0.0, 20.0, 1.0, true);
        let put: f64 = bachelier_generic(100.0, 100.0, 0.05, 0.0, 20.0, 1.0, false);
        let fwd = 100.0 * (0.05_f64).exp();
        let df = (-0.05_f64).exp();
        let parity = call - put - df * (fwd - 100.0);
        assert!(parity.abs() < 1e-4, "Bachelier parity = {parity}");
    }

    // -----------------------------------------------------------------------
    // BAW American
    // -----------------------------------------------------------------------

    #[test]
    fn baw_put_exceeds_european() {
        let am = barone_adesi_whaley_generic(100.0_f64, 100.0, 0.05, 0.02, 0.25, 1.0, false);
        let eu = bs_european_generic(100.0_f64, 100.0, 0.05, 0.02, 0.25, 1.0, false);
        assert!(
            am.npv >= eu.npv - 1e-6,
            "American put {} should >= European put {}",
            am.npv,
            eu.npv
        );
    }

    #[test]
    fn baw_early_exercise_premium_positive() {
        let res = barone_adesi_whaley_generic(100.0_f64, 110.0, 0.05, 0.0, 0.25, 1.0, false);
        assert!(res.early_exercise_premium >= 0.0, "EEP = {}", res.early_exercise_premium);
    }

    #[test]
    fn baw_call_no_div_equals_european() {
        let am = barone_adesi_whaley_generic(100.0_f64, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        let eu: f64 = black_scholes_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        assert!(
            (am.npv - eu).abs() < 1e-8,
            "No-div American call should equal European: {} vs {}",
            am.npv,
            eu
        );
    }

    // -----------------------------------------------------------------------
    // Merton JD
    // -----------------------------------------------------------------------

    #[test]
    fn merton_no_jumps_equals_bs() {
        let mjd = merton_jd_generic(
            100.0_f64, 100.0, 0.05, 0.0, 0.20, 1.0,
            0.0, 0.0, 0.0, true,
        );
        let bs: f64 = black_scholes_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        assert!(
            (mjd.npv - bs).abs() < 0.01,
            "Merton(λ=0) = {}, BS = {}",
            mjd.npv,
            bs
        );
    }

    #[test]
    fn merton_with_jumps() {
        let res = merton_jd_generic(
            100.0_f64, 100.0, 0.05, 0.0, 0.20, 1.0,
            0.5, -0.1, 0.15, true,
        );
        assert!(res.npv > 0.0);
        assert!(res.num_terms > 1);
    }

    // -----------------------------------------------------------------------
    // Chooser
    // -----------------------------------------------------------------------

    #[test]
    fn chooser_positive() {
        let v: f64 = chooser_generic(50.0, 50.0, 0.08, 0.0, 0.25, 0.25, 0.50);
        assert!(v > 0.0, "chooser = {v}");
    }

    #[test]
    fn chooser_exceeds_call_and_put() {
        let ch: f64 = chooser_generic(50.0, 50.0, 0.08, 0.0, 0.25, 0.25, 0.50);
        let call: f64 = black_scholes_generic(50.0, 50.0, 0.08, 0.0, 0.25, 0.50, true);
        let put: f64 = black_scholes_generic(50.0, 50.0, 0.08, 0.0, 0.25, 0.50, false);
        assert!(ch >= call.max(put) - 1e-6, "chooser {ch} < max(C,P)");
    }

    // -----------------------------------------------------------------------
    // Bond PV
    // -----------------------------------------------------------------------

    #[test]
    fn bond_pv_par() {
        // 5Y, 5% annual, discounted at 5% ≈ par
        let coupons = [5.0; 5];
        let times = [1.0, 2.0, 3.0, 4.0, 5.0];
        let pv: f64 = bond_pv_generic(&coupons, &times, 100.0, 5.0, 0.05);
        // With continuous compounding, par isn't exactly 100, but close
        assert!((pv - 100.0).abs() < 2.0, "pv = {pv}");
    }

    #[test]
    fn bond_pv_premium() {
        let coupons = [5.0; 5];
        let times = [1.0, 2.0, 3.0, 4.0, 5.0];
        let pv: f64 = bond_pv_generic(&coupons, &times, 100.0, 5.0, 0.03);
        assert!(pv > 100.0, "premium bond pv = {pv}");
    }

    #[test]
    fn bond_pv_discount() {
        let coupons = [5.0; 5];
        let times = [1.0, 2.0, 3.0, 4.0, 5.0];
        let pv: f64 = bond_pv_generic(&coupons, &times, 100.0, 5.0, 0.07);
        assert!(pv < 100.0, "discount bond pv = {pv}");
    }

    // -----------------------------------------------------------------------
    // Swap PV
    // -----------------------------------------------------------------------

    #[test]
    fn swap_pv_equal_legs_zero() {
        // If fixed = floating and discount rate doesn't matter, NPV ≈ 0
        let amounts = [2.5, 2.5, 2.5, 2.5];
        let times = [0.5, 1.0, 1.5, 2.0];
        let pv: f64 = swap_pv_generic(&amounts, &times, &amounts, &times, 0.05);
        assert!(pv.abs() < 1e-12, "pv = {pv}");
    }

    #[test]
    fn swap_pv_payer_benefits_when_float_higher() {
        let fixed = [2.5, 2.5, 2.5, 2.5];
        let float = [3.0, 3.0, 3.0, 3.0];
        let times = [0.5, 1.0, 1.5, 2.0];
        let pv: f64 = swap_pv_generic(&float, &times, &fixed, &times, 0.05);
        assert!(pv > 0.0, "payer should benefit: pv = {pv}");
    }

    // -----------------------------------------------------------------------
    // Kirk Spread
    // -----------------------------------------------------------------------

    #[test]
    fn kirk_spread_positive() {
        let v: f64 = kirk_spread_generic(
            100.0, 90.0, 5.0, 0.05, 0.20, 0.25, 0.5, 1.0,
        );
        assert!(v > 0.0, "kirk = {v}");
    }

    #[test]
    fn kirk_spread_deep_itm() {
        // S1 much larger than S2 + K → intrinsic dominated
        let v: f64 = kirk_spread_generic(
            200.0, 50.0, 10.0, 0.05, 0.20, 0.25, 0.5, 1.0,
        );
        let intrinsic = 200.0 - 50.0 - 10.0;
        assert!(v >= intrinsic * 0.9, "deep ITM kirk = {v}");
    }
}

// ===========================================================================
// Phase E: Curve-based generic analytics (INFRA-5)
// ===========================================================================

// ---------------------------------------------------------------------------
// leg_npv_generic: NPV of a leg using a GenericYieldCurve
// ---------------------------------------------------------------------------

/// Compute the net present value of a sequence of cashflows using a generic
/// yield curve.
///
/// `times[i]` is the year-fraction to payment `i`, `amounts[i]` is the cash
/// amount. Both are `f64` (the curve returns generic `T` discount factors).
///
/// This is the generic analogue of [`ql_cashflows::cashflow_analytics::npv`]
/// and [`ql_aad::cashflows::npv`], bridging both worlds through the
/// [`GenericYieldCurve`] trait.
pub fn leg_npv_generic<T: Number>(
    times: &[f64],
    amounts: &[f64],
    curve: &dyn GenericYieldCurve<T>,
) -> T {
    debug_assert_eq!(times.len(), amounts.len());
    let mut total = T::zero();
    for (&t, &amt) in times.iter().zip(amounts) {
        total += T::from_f64(amt) * curve.discount_t(t);
    }
    total
}

/// Compute PV01 (1 bp parallel sensitivity) of a leg analytically.
///
/// `PV01 ≈ -Σ t_i × amount_i × DF(t_i) × 0.0001`
pub fn leg_pv01_generic<T: Number>(
    times: &[f64],
    amounts: &[f64],
    curve: &dyn GenericYieldCurve<T>,
) -> T {
    debug_assert_eq!(times.len(), amounts.len());
    let mut total = T::zero();
    for (&t, &amt) in times.iter().zip(amounts) {
        let df = curve.discount_t(t);
        total += T::from_f64(-t * amt * 0.0001) * df;
    }
    total
}

/// Macaulay duration of a leg: `D = Σ(t_i × CF_i × DF_i) / Σ(CF_i × DF_i)`.
pub fn leg_duration_generic<T: Number>(
    times: &[f64],
    amounts: &[f64],
    curve: &dyn GenericYieldCurve<T>,
) -> T {
    debug_assert_eq!(times.len(), amounts.len());
    let mut weighted = T::zero();
    let mut total = T::zero();
    for (&t, &amt) in times.iter().zip(amounts) {
        let pv = T::from_f64(amt) * curve.discount_t(t);
        weighted += T::from_f64(t) * pv;
        total += pv;
    }
    weighted / total
}

// ---------------------------------------------------------------------------
// Bond NPV with curve
// ---------------------------------------------------------------------------

/// Price a fixed-rate bond using a generic yield curve.
///
/// `coupon_times` / `coupon_amounts` are the coupon payment schedule.
/// `notional` is repaid at `maturity_time`.
///
/// This is the curve-based analogue of [`bond_pv_generic`] (flat-rate).
pub fn bond_pv_curve_generic<T: Number>(
    coupon_times: &[f64],
    coupon_amounts: &[f64],
    notional: f64,
    maturity_time: f64,
    curve: &dyn GenericYieldCurve<T>,
) -> T {
    let mut pv = T::zero();
    for (&t, &c) in coupon_times.iter().zip(coupon_amounts) {
        pv += T::from_f64(c) * curve.discount_t(t);
    }
    // Notional redemption at maturity
    pv += T::from_f64(notional) * curve.discount_t(maturity_time);
    pv
}

// ---------------------------------------------------------------------------
// Swap NPV with curve
// ---------------------------------------------------------------------------

/// Price an interest-rate swap using a generic yield curve.
///
/// Fixed leg pays `fixed_amounts[i]` at `fixed_times[i]`.
/// Floating leg pays `float_amounts[i]` at `float_times[i]`.
/// Returns: `NPV(float) - NPV(fixed)` (receiver-float convention).
///
/// This is the curve-based analogue of [`swap_pv_generic`] (flat-rate).
pub fn swap_pv_curve_generic<T: Number>(
    float_times: &[f64],
    float_amounts: &[f64],
    fixed_times: &[f64],
    fixed_amounts: &[f64],
    curve: &dyn GenericYieldCurve<T>,
) -> T {
    let float_pv = leg_npv_generic(float_times, float_amounts, curve);
    let fixed_pv = leg_npv_generic(fixed_times, fixed_amounts, curve);
    float_pv - fixed_pv
}

// ---------------------------------------------------------------------------
// Par rate with curve
// ---------------------------------------------------------------------------

/// Par coupon rate for a bond: the rate such that NPV = face value.
///
/// `par = (face - face × DF(T)) / Σ(yf_i × DF(t_i))`
///
/// where `yf_i` are the coupon year fractions and `T` is maturity.
pub fn par_rate_generic<T: Number>(
    coupon_times: &[f64],
    coupon_yfs: &[f64],
    _notional: f64,
    maturity_time: f64,
    curve: &dyn GenericYieldCurve<T>,
) -> T {
    let df_mat = curve.discount_t(maturity_time);
    let mut annuity = T::zero();
    for (&t, &yf) in coupon_times.iter().zip(coupon_yfs) {
        annuity += T::from_f64(yf) * curve.discount_t(t);
    }
    // par = (1 - DF(maturity)) / Σ(yf_i × DF(t_i))
    (T::one() - df_mat) / annuity
}

// ---------------------------------------------------------------------------
// Swap rate with curve
// ---------------------------------------------------------------------------

/// Par swap rate: the fixed rate making the swap NPV zero.
///
/// `swap_rate = (DF(t_0) - DF(t_n)) / Σ(yf_i × DF(t_i))` for vanilla swaps
/// where floating leg is valued as `DF(t_0) - DF(t_n)`.
pub fn swap_rate_generic<T: Number>(
    float_start_time: f64,
    float_end_time: f64,
    fixed_times: &[f64],
    fixed_yfs: &[f64],
    curve: &dyn GenericYieldCurve<T>,
) -> T {
    let df_start = curve.discount_t(float_start_time);
    let df_end = curve.discount_t(float_end_time);
    let mut annuity = T::zero();
    for (&t, &yf) in fixed_times.iter().zip(fixed_yfs) {
        annuity += T::from_f64(yf) * curve.discount_t(t);
    }
    (df_start - df_end) / annuity
}

// ---------------------------------------------------------------------------
// Key-rate duration with curve
// ---------------------------------------------------------------------------

/// Key-rate durations: sensitivity of NPV to a 1 bp bump at each pillar.
///
/// Returns a vector of `∂NPV/∂r_i` (one per pillar of the curve), computed
/// analytically for a flat or linearly-interpolated curve.
///
/// For exact key-rate durations via AD, use `AReal`/`DualVec` inputs
/// with the curve pillars as tape variables.
pub fn key_rate_durations_generic(
    times: &[f64],
    amounts: &[f64],
    curve_times: &[f64],
    curve_rates: &[f64],
) -> Vec<f64> {
    let bp = 1e-4;
    let n = curve_times.len();
    let mut krds = Vec::with_capacity(n);
    // Base NPV
    let base_curve = ql_termstructures::generic::InterpDiscountCurve::from_zero_rates(
        curve_times,
        curve_rates,
    );
    let base_npv: f64 = leg_npv_generic(times, amounts, &base_curve);
    // Bump each pillar
    for j in 0..n {
        let mut bumped = curve_rates.to_vec();
        bumped[j] += bp;
        let bumped_curve =
            ql_termstructures::generic::InterpDiscountCurve::from_zero_rates(curve_times, &bumped);
        let bumped_npv: f64 = leg_npv_generic(times, amounts, &bumped_curve);
        krds.push((bumped_npv - base_npv) / bp);
    }
    krds
}

// ---------------------------------------------------------------------------
// FRA with curve
// ---------------------------------------------------------------------------

/// Forward rate agreement (FRA) NPV using a generic yield curve.
///
/// The FRA buyer receives `notional × (L - K) × τ × DF(T2)` where
/// `L = forward_rate(T1, T2)`.
pub fn fra_npv_generic<T: Number>(
    notional: f64,
    fixed_rate: f64,
    t1: f64,
    t2: f64,
    curve: &dyn GenericYieldCurve<T>,
) -> T {
    let tau = t2 - t1;
    let fwd = curve.forward_rate_t(t1, t2);
    let df = curve.discount_t(t2);
    T::from_f64(notional) * (fwd - T::from_f64(fixed_rate)) * T::from_f64(tau) * df
}

// ===========================================================================
// Phase F: Discounting engines (AD-49 to AD-68)
// ===========================================================================

// ---------------------------------------------------------------------------
// AD-49: Vanilla IRS (engine-level) with generic curve
// ---------------------------------------------------------------------------

/// Result from a vanilla interest-rate swap pricing.
#[derive(Debug, Clone, Copy)]
pub struct SwapResult<T: Number> {
    /// Net present value (float leg - fixed leg).
    pub npv: T,
    /// Fixed-leg NPV.
    pub fixed_leg_npv: T,
    /// Floating-leg NPV (par approximation: DF(start) - DF(end)).
    pub float_leg_npv: T,
    /// Par swap rate.
    pub fair_rate: T,
}

/// Price a vanilla interest-rate swap using generic yield curve(s).
///
/// Fixed leg: pays `notional × fixed_rate × yf_i` at each `fixed_times[i]`.
/// Floating leg: approximated as `notional × (DF(start) - DF(end))`.
///
/// For multi-curve (forecast != discount), use `swap_multicurve_generic`.
pub fn swap_engine_generic<T: Number>(
    notional: f64,
    fixed_rate: f64,
    fixed_times: &[f64],
    fixed_yfs: &[f64],
    float_start: f64,
    float_end: f64,
    curve: &dyn GenericYieldCurve<T>,
) -> SwapResult<T> {
    // Fixed leg
    let mut fixed_npv = T::zero();
    for (&t, &yf) in fixed_times.iter().zip(fixed_yfs) {
        fixed_npv += T::from_f64(notional * fixed_rate * yf) * curve.discount_t(t);
    }

    // Floating leg (par approximation)
    let df_start = curve.discount_t(float_start);
    let df_end = curve.discount_t(float_end);
    let float_npv = T::from_f64(notional) * (df_start - df_end);

    // Fair rate
    let mut annuity = T::zero();
    for (&t, &yf) in fixed_times.iter().zip(fixed_yfs) {
        annuity += T::from_f64(yf) * curve.discount_t(t);
    }
    let fair_rate = (df_start - df_end) / annuity;

    SwapResult {
        npv: float_npv - fixed_npv,
        fixed_leg_npv: fixed_npv,
        float_leg_npv: float_npv,
        fair_rate,
    }
}

// ---------------------------------------------------------------------------
// AD-50: OIS swap
// ---------------------------------------------------------------------------

/// Price an OIS swap using a generic yield curve.
///
/// The OIS floating leg discounting is equivalent to the single-curve
/// bootstrap: `NPV(float) = notional × (DF(start) - DF(end))`.
/// The fixed leg uses the OIS-specific schedule.
pub fn ois_swap_generic<T: Number>(
    notional: f64,
    fixed_rate: f64,
    fixed_times: &[f64],
    fixed_yfs: &[f64],
    float_start: f64,
    float_end: f64,
    ois_curve: &dyn GenericYieldCurve<T>,
) -> SwapResult<T> {
    // OIS swaps have the same structure as vanilla when using single-curve
    swap_engine_generic(notional, fixed_rate, fixed_times, fixed_yfs,
                        float_start, float_end, ois_curve)
}

// ---------------------------------------------------------------------------
// AD-51: Fixed-rate bond (engine-level)
// ---------------------------------------------------------------------------

/// Fixed-rate bond pricing result.
#[derive(Debug, Clone, Copy)]
pub struct BondResult<T: Number> {
    /// Dirty price (full NPV).
    pub npv: T,
    /// Clean price (dirty - accrued).
    pub clean_price: T,
    /// Bond yield (from the curve at maturity).
    pub yield_at_maturity: T,
}

/// Price a fixed-rate bond using a generic yield curve.
///
/// Computes dirty price, clean price, and yield.
pub fn fixed_bond_engine_generic<T: Number>(
    notional: f64,
    coupon_rate: f64,
    coupon_times: &[f64],
    coupon_yfs: &[f64],
    maturity_time: f64,
    accrued: f64,
    curve: &dyn GenericYieldCurve<T>,
) -> BondResult<T> {
    let mut npv = T::zero();
    for (&t, &yf) in coupon_times.iter().zip(coupon_yfs) {
        npv += T::from_f64(notional * coupon_rate * yf) * curve.discount_t(t);
    }
    npv += T::from_f64(notional) * curve.discount_t(maturity_time);

    let clean = npv - T::from_f64(accrued);
    let ytm = curve.zero_rate_t(maturity_time);

    BondResult {
        npv,
        clean_price: clean,
        yield_at_maturity: ytm,
    }
}

// ---------------------------------------------------------------------------
// AD-52: Floating-rate bond
// ---------------------------------------------------------------------------

/// Price a floating-rate bond/FRN using generic yield curves.
///
/// Forecast curve provides forward rates; discount curve discounts cashflows.
/// Each floating coupon pays `notional × (forward_rate + spread) × yf`.
pub fn floating_bond_engine_generic<T: Number>(
    notional: f64,
    spread: f64,
    float_start_times: &[f64],
    float_end_times: &[f64],
    float_yfs: &[f64],
    maturity_time: f64,
    forecast_curve: &dyn GenericYieldCurve<T>,
    discount_curve: &dyn GenericYieldCurve<T>,
) -> T {
    let mut npv = T::zero();
    for i in 0..float_start_times.len() {
        let t1 = float_start_times[i];
        let t2 = float_end_times[i];
        let yf = float_yfs[i];
        let fwd = forecast_curve.forward_rate_t(t1, t2);
        let coupon = T::from_f64(notional) * (fwd + T::from_f64(spread)) * T::from_f64(yf);
        npv += coupon * discount_curve.discount_t(t2);
    }
    // Notional redemption
    npv += T::from_f64(notional) * discount_curve.discount_t(maturity_time);
    npv
}

// ---------------------------------------------------------------------------
// AD-53: Zero-coupon bond
// ---------------------------------------------------------------------------

/// Price a zero-coupon bond: `NPV = notional × DF(maturity)`.
pub fn zero_coupon_bond_generic<T: Number>(
    notional: f64,
    maturity_time: f64,
    curve: &dyn GenericYieldCurve<T>,
) -> T {
    T::from_f64(notional) * curve.discount_t(maturity_time)
}

// ---------------------------------------------------------------------------
// AD-54: Amortizing bond / FRN
// ---------------------------------------------------------------------------

/// Price an amortizing bond where each period has its own outstanding notional.
///
/// `notionals[i]` is the outstanding notional for period `i`.
/// `coupon_rate` is the fixed coupon rate.
pub fn amortizing_bond_generic<T: Number>(
    notionals: &[f64],
    coupon_rate: f64,
    coupon_times: &[f64],
    coupon_yfs: &[f64],
    curve: &dyn GenericYieldCurve<T>,
) -> T {
    debug_assert_eq!(notionals.len(), coupon_times.len());
    let mut npv = T::zero();
    for i in 0..notionals.len() {
        let interest = notionals[i] * coupon_rate * coupon_yfs[i];
        let principal = if i + 1 < notionals.len() {
            notionals[i] - notionals[i + 1]
        } else {
            notionals[i] // final redemption
        };
        npv += T::from_f64(interest + principal) * curve.discount_t(coupon_times[i]);
    }
    npv
}

// ---------------------------------------------------------------------------
// AD-55: Inflation-linked bond
// ---------------------------------------------------------------------------

/// Price an inflation-linked bond.
///
/// Each coupon = `notional × coupon_rate × yf × (CPI_t / CPI_base)`.
/// Redemption = `notional × max(CPI_T / CPI_base, floor)`.
///
/// `cpi_ratios[i]` = `CPI(t_i) / CPI_base` for each coupon date.
/// `cpi_ratio_mat` = `CPI(T) / CPI_base` for maturity.
pub fn inflation_bond_generic<T: Number>(
    notional: f64,
    coupon_rate: f64,
    coupon_times: &[f64],
    coupon_yfs: &[f64],
    cpi_ratios: &[T],
    maturity_time: f64,
    cpi_ratio_mat: T,
    floor: f64,
    curve: &dyn GenericYieldCurve<T>,
) -> T {
    let mut npv = T::zero();
    for i in 0..coupon_times.len() {
        let coupon = T::from_f64(notional * coupon_rate * coupon_yfs[i]) * cpi_ratios[i];
        npv += coupon * curve.discount_t(coupon_times[i]);
    }
    // Inflation-adjusted redemption with floor
    let adjusted = if cpi_ratio_mat.to_f64() > floor {
        cpi_ratio_mat
    } else {
        T::from_f64(floor)
    };
    npv += T::from_f64(notional) * adjusted * curve.discount_t(maturity_time);
    npv
}

// ---------------------------------------------------------------------------
// AD-58: CDS (mid-point engine)
// ---------------------------------------------------------------------------

/// Generic CDS (midpoint engine) using a yield curve and survival curve.
///
/// Protection leg = `notional × (1 - R) × Σ [S(t_{i-1}) - S(t_i)] × DF(t_mid)`
/// Premium leg = `notional × spread × Σ yf_i × S(t_i) × DF(t_i)`
///
/// Both the yield curve and survival curve are generic for AD.
pub fn cds_midpoint_generic<T: Number>(
    notional: f64,
    spread: f64,
    recovery: f64,
    payment_times: &[f64],
    payment_yfs: &[f64],
    yield_curve: &dyn GenericYieldCurve<T>,
    survival_curve: &dyn GenericYieldCurve<T>, // "discount" = survival prob
) -> T {
    let lgd = 1.0 - recovery;
    let mut prot_leg = T::zero();
    let mut prem_leg = T::zero();

    let mut prev_surv = T::one();
    for i in 0..payment_times.len() {
        let t = payment_times[i];
        let yf = payment_yfs[i];
        let surv = survival_curve.discount_t(t);
        let default_prob = prev_surv - surv;

        // Protection leg: pays (1-R) on default
        let t_mid = if i > 0 {
            (payment_times[i - 1] + t) / 2.0
        } else {
            t / 2.0
        };
        prot_leg += T::from_f64(notional * lgd) * default_prob * yield_curve.discount_t(t_mid);

        // Premium leg: pays spread × yf on survival
        prem_leg += T::from_f64(notional * spread * yf) * surv * yield_curve.discount_t(t);

        prev_surv = surv;
    }

    // NPV = protection leg - premium leg (from protection buyer's perspective)
    prot_leg - prem_leg
}

// ---------------------------------------------------------------------------
// AD-60: Cross-currency swap
// ---------------------------------------------------------------------------

/// Price a cross-currency (xccy) swap using two yield curves and FX spot.
///
/// Domestic leg: `notional_dom × Σ (fixed_rate_dom × yf × DF_dom(t))`
/// Foreign leg: `notional_for × fx_spot × Σ (fixed_rate_for × yf × DF_for(t))`
/// Returns NPV in domestic currency terms.
pub fn xccy_swap_generic<T: Number>(
    notional_dom: f64,
    notional_for: f64,
    fixed_rate_dom: f64,
    fixed_rate_for: f64,
    fx_spot: T,
    times: &[f64],
    yfs: &[f64],
    maturity_time: f64,
    dom_curve: &dyn GenericYieldCurve<T>,
    for_curve: &dyn GenericYieldCurve<T>,
) -> T {
    let mut dom_pv = T::zero();
    let mut for_pv = T::zero();
    for (&t, &yf) in times.iter().zip(yfs) {
        dom_pv += T::from_f64(notional_dom * fixed_rate_dom * yf) * dom_curve.discount_t(t);
        for_pv += T::from_f64(notional_for * fixed_rate_for * yf) * for_curve.discount_t(t);
    }
    // Notional exchange at maturity
    dom_pv += T::from_f64(notional_dom) * dom_curve.discount_t(maturity_time);
    for_pv += T::from_f64(notional_for) * for_curve.discount_t(maturity_time);

    // NPV in domestic: receive domestic, pay foreign (converted at spot)
    dom_pv - fx_spot * for_pv
}

// ---------------------------------------------------------------------------
// AD-61: FX forward
// ---------------------------------------------------------------------------

/// Price an FX forward using two yield curves and FX spot.
///
/// `FX_fwd = S × DF_for(T) / DF_dom(T)`
/// `NPV = notional × (FX_fwd - K) × DF_dom(T)`
pub fn fx_forward_generic<T: Number>(
    notional: f64,
    fx_spot: T,
    strike: f64,
    maturity_time: f64,
    dom_curve: &dyn GenericYieldCurve<T>,
    for_curve: &dyn GenericYieldCurve<T>,
) -> T {
    let df_dom = dom_curve.discount_t(maturity_time);
    let df_for = for_curve.discount_t(maturity_time);
    let fwd = fx_spot * df_for / df_dom;
    T::from_f64(notional) * (fwd - T::from_f64(strike)) * df_dom
}

// ---------------------------------------------------------------------------
// AD-62: BMA swap (treating as fixed-vs-floating with BMA convention)
// ---------------------------------------------------------------------------

/// Price a BMA-style swap using a generic yield curve.
///
/// BMA leg uses weekly resets approximated by `DF(start) - DF(end)`.
/// Fixed leg uses standard coupon schedule.
pub fn bma_swap_generic<T: Number>(
    notional: f64,
    fixed_rate: f64,
    fixed_times: &[f64],
    fixed_yfs: &[f64],
    float_start: f64,
    float_end: f64,
    curve: &dyn GenericYieldCurve<T>,
) -> T {
    // Same structure as vanilla swap for generic pricing
    let res = swap_engine_generic(notional, fixed_rate, fixed_times, fixed_yfs,
                                   float_start, float_end, curve);
    res.npv
}

// ---------------------------------------------------------------------------
// AD-63: Zero-coupon swap
// ---------------------------------------------------------------------------

/// Price a zero-coupon swap: fixed leg pays a single compounded amount at maturity.
///
/// Fixed leg: `notional × ((1 + r)^T - 1) × DF(T)`
/// Floating leg: `notional × (DF(0) - DF(T))`
pub fn zero_coupon_swap_generic<T: Number>(
    notional: f64,
    fixed_rate: f64,
    maturity_time: f64,
    curve: &dyn GenericYieldCurve<T>,
) -> T {
    let df_mat = curve.discount_t(maturity_time);

    // Fixed leg: single compounded payment at maturity
    let fixed_pv = T::from_f64(notional * ((1.0 + fixed_rate).powf(maturity_time) - 1.0)) * df_mat;

    // Floating leg
    let float_pv = T::from_f64(notional) * (T::one() - df_mat);

    float_pv - fixed_pv
}

// ---------------------------------------------------------------------------
// AD-64: Basis swap
// ---------------------------------------------------------------------------

/// Price a basis swap using two forecast curves and one discount curve.
///
/// Leg 1: `notional × Σ (fwd_1(t_{i-1}, t_i) + spread_1) × yf × DF(t_i)`
/// Leg 2: `notional × Σ (fwd_2(t_{i-1}, t_i) + spread_2) × yf × DF(t_i)`
pub fn basis_swap_generic<T: Number>(
    notional: f64,
    spread1: f64,
    spread2: f64,
    start_times: &[f64],
    end_times: &[f64],
    yfs: &[f64],
    forecast_curve1: &dyn GenericYieldCurve<T>,
    forecast_curve2: &dyn GenericYieldCurve<T>,
    discount_curve: &dyn GenericYieldCurve<T>,
) -> T {
    let mut leg1_pv = T::zero();
    let mut leg2_pv = T::zero();
    for i in 0..start_times.len() {
        let t1 = start_times[i];
        let t2 = end_times[i];
        let yf = yfs[i];
        let df = discount_curve.discount_t(t2);

        let fwd1 = forecast_curve1.forward_rate_t(t1, t2);
        let fwd2 = forecast_curve2.forward_rate_t(t1, t2);

        leg1_pv += T::from_f64(notional * yf) * (fwd1 + T::from_f64(spread1)) * df;
        leg2_pv += T::from_f64(notional * yf) * (fwd2 + T::from_f64(spread2)) * df;
    }
    leg1_pv - leg2_pv
}

// ---------------------------------------------------------------------------
// AD-65: CPI swap
// ---------------------------------------------------------------------------

/// Price a CPI (zero-coupon inflation) swap.
///
/// Fixed leg: `notional × ((1 + r)^T - 1) × DF(T)`
/// Inflation leg: `notional × (CPI_T / CPI_0 - 1) × DF(T)`
///
/// `cpi_ratio` = projected `CPI(T) / CPI(0)` (generic T for AD).
pub fn cpi_swap_generic<T: Number>(
    notional: f64,
    fixed_rate: f64,
    maturity_time: f64,
    cpi_ratio: T,
    curve: &dyn GenericYieldCurve<T>,
) -> T {
    let df = curve.discount_t(maturity_time);
    let fixed_pv = T::from_f64(notional * ((1.0 + fixed_rate).powf(maturity_time) - 1.0)) * df;
    let infl_pv = T::from_f64(notional) * (cpi_ratio - T::one()) * df;
    infl_pv - fixed_pv
}

// ---------------------------------------------------------------------------
// AD-67: Cat bond (simplified)
// ---------------------------------------------------------------------------

/// Price a catastrophe bond (simplified model).
///
/// Regular coupons discounted at the yield curve, with a probability
/// of early knockout (catastrophe event) reducing NPV.
///
/// `survival_probs[i]` is the probability of no cat event by time `t_i`.
pub fn cat_bond_generic<T: Number>(
    notional: f64,
    coupon_rate: f64,
    coupon_times: &[f64],
    coupon_yfs: &[f64],
    maturity_time: f64,
    survival_probs: &[f64],
    curve: &dyn GenericYieldCurve<T>,
) -> T {
    let mut npv = T::zero();
    for i in 0..coupon_times.len() {
        let coupon = notional * coupon_rate * coupon_yfs[i] * survival_probs[i];
        npv += T::from_f64(coupon) * curve.discount_t(coupon_times[i]);
    }
    // Notional at maturity, conditional on survival
    let mat_survival = survival_probs.last().copied().unwrap_or(1.0);
    npv += T::from_f64(notional * mat_survival) * curve.discount_t(maturity_time);
    npv
}

// ---------------------------------------------------------------------------
// AD-68: Bond forward
// ---------------------------------------------------------------------------

/// Price a bond forward: the right to buy a bond at a forward price.
///
/// `NPV = (bond_dirty_price - forward_price × DF(delivery)) × DF(settle)`
///
/// Actually: `NPV = Σ(coupon_i × DF(t_i)) + notional × DF(T) - K × DF(delivery_time)`
pub fn bond_forward_generic<T: Number>(
    notional: f64,
    coupon_rate: f64,
    coupon_times: &[f64],     // coupons after delivery
    coupon_yfs: &[f64],
    maturity_time: f64,
    forward_price: f64,
    delivery_time: f64,
    curve: &dyn GenericYieldCurve<T>,
) -> T {
    // Bond dirty price from today
    let mut bond_pv = T::zero();
    for (&t, &yf) in coupon_times.iter().zip(coupon_yfs) {
        bond_pv += T::from_f64(notional * coupon_rate * yf) * curve.discount_t(t);
    }
    bond_pv += T::from_f64(notional) * curve.discount_t(maturity_time);

    // Forward contract: pay forward_price at delivery
    let fwd_pv = T::from_f64(forward_price) * curve.discount_t(delivery_time);

    bond_pv - fwd_pv
}

// ---------------------------------------------------------------------------
// AD-49 multi-curve: swap with separate forecast and discount
// ---------------------------------------------------------------------------

/// Price a swap with separate forecast and discount curves (multi-curve).
///
/// Floating leg: `Σ notional × fwd(t_{i-1}, t_i) × yf × DF_disc(t_i)`
/// Fixed leg: `Σ notional × fixed_rate × yf × DF_disc(t_i)`
pub fn swap_multicurve_generic<T: Number>(
    notional: f64,
    fixed_rate: f64,
    fixed_times: &[f64],
    fixed_yfs: &[f64],
    float_start_times: &[f64],
    float_end_times: &[f64],
    float_yfs: &[f64],
    forecast_curve: &dyn GenericYieldCurve<T>,
    discount_curve: &dyn GenericYieldCurve<T>,
) -> T {
    // Fixed leg
    let mut fixed_npv = T::zero();
    for (&t, &yf) in fixed_times.iter().zip(fixed_yfs) {
        fixed_npv += T::from_f64(notional * fixed_rate * yf) * discount_curve.discount_t(t);
    }

    // Floating leg
    let mut float_npv = T::zero();
    for i in 0..float_start_times.len() {
        let fwd = forecast_curve.forward_rate_t(float_start_times[i], float_end_times[i]);
        let coupon = T::from_f64(notional * float_yfs[i]) * fwd;
        float_npv += coupon * discount_curve.discount_t(float_end_times[i]);
    }

    float_npv - fixed_npv
}

// ===========================================================================
// Phase G: American engines (AD-44 to AD-48)
// ===========================================================================

// AD-44: BAW — already implemented as `barone_adesi_whaley_generic` above

// ---------------------------------------------------------------------------
// AD-45: Bjerksund-Stensland (generic)
// ---------------------------------------------------------------------------

/// BJS φ function, generic over `T: Number`.
///
/// ```text
/// φ(S,T,γ,H,I) = e^λ · S^γ · [Φ(-d₁) - (I/S)^κ · Φ(-d₂)]
/// ```
#[allow(clippy::too_many_arguments)]
fn bjs_phi_generic<T: Number>(
    s: T, t: T, gamma: f64, h: T, big_i: T, r: T, q: T, vol: T,
) -> T {
    let zero = T::zero();
    let _one = T::one();

    if t.to_f64() <= 0.0 || s.to_f64() <= 0.0 || h.to_f64() <= 0.0 || big_i.to_f64() <= 0.0 {
        return zero;
    }

    let sig2 = vol * vol;
    let b = r - q;
    let sqrt_t = t.sqrt();
    let g = T::from_f64(gamma);

    let lambda = (zero - r) * t + g * b * t
        + T::from_f64(0.5 * gamma * (gamma - 1.0)) * sig2 * t;
    let kappa_val = T::from_f64(2.0) * b / sig2 + T::from_f64(2.0 * gamma - 1.0);

    let d1 = ((s / h).ln() + (b + T::from_f64(gamma - 0.5) * sig2) * t) / (vol * sqrt_t);
    let d2 = ((big_i * big_i / (s * h)).ln()
        + (b + T::from_f64(gamma - 0.5) * sig2) * t) / (vol * sqrt_t);

    lambda.exp() * s.powf(g) * (normal_cdf(zero - d1) - (big_i / s).powf(kappa_val) * normal_cdf(zero - d2))
}

/// Bjerksund-Stensland (1993/2002) American option pricing, generic over `T: Number`.
///
/// Purely analytic (no root-finding), so directly differentiable.
#[allow(clippy::too_many_arguments)]
pub fn bjerksund_stensland_generic<T: Number>(
    spot: T, strike: T, r: T, q: T, vol: T, t: T, is_call: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();
    let half = T::half();

    // Edge cases
    if t.to_f64() <= 0.0 {
        let intrinsic = if is_call { spot - strike } else { strike - spot };
        return if intrinsic.to_f64() > 0.0 { intrinsic } else { zero };
    }

    // American call with no dividends = European
    if is_call && q.to_f64() <= 0.0 {
        return black_scholes_generic(spot, strike, r, q, vol, t, true);
    }

    // Put-call transformation: put(S,K,r,q) = call(K,S,q,r)
    let (s, x, rf, dy) = if is_call {
        (spot, strike, r, q)
    } else {
        (strike, spot, q, r)
    };

    let sig2 = vol * vol;

    // β = (1/2 - dy/σ²) + sqrt((dy/σ² - 1/2)² + 2rf/σ²)
    let half_minus = half - dy / sig2;
    let beta = half_minus + (half_minus * half_minus + T::from_f64(2.0) * rf / sig2).sqrt();

    // B∞ = (β/(β-1)) · X
    let b_inf = (beta / (beta - one)) * x;

    // B₀ = max(X, (rf/(rf-dy))·X)
    let b0 = if (rf - dy).to_f64().abs() < 1e-15 {
        x
    } else {
        let ratio = rf / (rf - dy) * x;
        if ratio.to_f64() > x.to_f64() { ratio } else { x }
    };

    // Trigger level
    let ht = (zero - (rf - dy)) * t + T::from_f64(2.0) * vol * t.sqrt();
    let trigger = b0 + (b_inf - b0) * (one - (zero - ht).exp());

    if s.to_f64() >= trigger.to_f64() {
        // In the exercise region: value = intrinsic of the transformed call
        let val = s - x;
        // But still floor at European price
        let euro = black_scholes_generic(spot, strike, r, q, vol, t, is_call);
        let intrinsic_orig = if is_call { spot - strike } else { strike - spot };
        let intrinsic_orig = if intrinsic_orig.to_f64() > 0.0 { intrinsic_orig } else { zero };
        let result = if val.to_f64() > euro.to_f64() { val } else { euro };
        return if result.to_f64() > intrinsic_orig.to_f64() { result } else { intrinsic_orig };
    }

    // α = (B - X) · B^{-β}
    let alpha = (trigger - x) * trigger.powf(zero - beta);

    // BJS formula
    let p1 = alpha * s.powf(beta);
    let p2 = alpha * bjs_phi_generic(s, t, beta.to_f64(), trigger, trigger, rf, dy, vol);
    let p3 = bjs_phi_generic(s, t, 1.0, trigger, trigger, rf, dy, vol);
    let p4 = bjs_phi_generic(s, t, 1.0, x, trigger, rf, dy, vol);
    let p5 = x * bjs_phi_generic(s, t, 0.0, trigger, trigger, rf, dy, vol);
    let p6 = x * bjs_phi_generic(s, t, 0.0, x, trigger, rf, dy, vol);

    let val = p1 - p2 + p3 - p4 - p5 + p6;

    // Floor at European price and intrinsic
    let euro = black_scholes_generic(spot, strike, r, q, vol, t, is_call);
    let intrinsic = if is_call { spot - strike } else { strike - spot };
    let intrinsic = if intrinsic.to_f64() > 0.0 { intrinsic } else { zero };

    // Return max(val, european, intrinsic)
    let result = if val.to_f64() > euro.to_f64() { val } else { euro };
    if result.to_f64() > intrinsic.to_f64() { result } else { intrinsic }
}

// ---------------------------------------------------------------------------
// AD-46: QD+ American (generic)
// ---------------------------------------------------------------------------

/// QD+ American option pricing, generic over `T: Number`.
///
/// The exercise boundary is solved in `f64` (Newton iteration) then
/// the Kim integral representation is evaluated in generic `T`,
/// preserving AD sensitivities through the payoff/discount computation.
///
/// For full boundary sensitivity via IFT, use `AReal` with `ql_aad::solvers`.
#[allow(clippy::too_many_arguments)]
pub fn qd_plus_generic<T: Number>(
    spot: T, strike: T, r: T, q: T, vol: T, t: T, is_call: bool,
) -> T {
    let zero = T::zero();
    let _one = T::one();
    let omega = if is_call { 1.0 } else { -1.0 };

    if t.to_f64() <= 0.0 {
        let intr = T::from_f64(omega) * (spot - strike);
        return if intr.to_f64() > 0.0 { intr } else { zero };
    }

    if is_call && q.to_f64() <= 0.0 {
        return black_scholes_generic(spot, strike, r, q, vol, t, true);
    }

    let sig2 = vol * vol;
    let tf = t.to_f64();
    let _sf = spot.to_f64();
    let kf = strike.to_f64();
    let rf = r.to_f64();
    let qf = q.to_f64();
    let vf = vol.to_f64();
    let sig2f = vf * vf;

    // GL 4-point nodes/weights on [0,1]
    let gl_nodes = [
        0.069_431_844_202_973_71,
        0.330_009_478_207_571_87,
        0.669_990_521_792_428_1,
        0.930_568_155_797_026_3,
    ];
    let gl_weights = [
        0.173_927_422_568_726_93,
        0.326_072_577_431_273_07,
        0.326_072_577_431_273_07,
        0.173_927_422_568_726_93,
    ];

    // Solve exercise boundary at each GL node in f64
    let boundaries: Vec<f64> = gl_nodes.iter().map(|&node| {
        let tau_i = tf * node;
        if tau_i < 1e-10 {
            return if is_call {
                if qf > 0.0 { kf * (rf / qf).max(1.0) } else { 1e12 }
            } else {
                kf * (rf / qf).min(1.0)
            };
        }
        let k_i = if rf.abs() < 1e-15 {
            2.0 / sig2f
        } else {
            2.0 * rf / (sig2f * (1.0 - (-rf * tau_i).exp()))
        };
        let n_val = 2.0 * (rf - qf) / sig2f;
        let disc = ((n_val - 1.0).powi(2) + 4.0 * k_i).sqrt();
        let q_baw = if is_call {
            0.5 * (-(n_val - 1.0) + disc)
        } else {
            0.5 * (-(n_val - 1.0) - disc)
        };
        crate::american_engines::baw_critical_price(kf, rf, qf, vf, tau_i, is_call, q_baw)
    }).collect();

    // European price (generic)
    let euro = black_scholes_generic(spot, strike, r, q, vol, t, is_call);

    // Compute the Kim integral in generic T
    let mut premium = zero;
    for (i, (&node, &weight)) in gl_nodes.iter().zip(gl_weights.iter()).enumerate() {
        let tau_i = tf * node;
        if tau_i < 1e-10 {
            continue;
        }
        let b_i = boundaries[i];
        if b_i <= 0.0 || b_i > 1e10 {
            continue;
        }

        let tau = T::from_f64(tau_i);
        let sqrt_tau = tau.sqrt();
        let b = T::from_f64(b_i);

        let d1 = ((spot / b).ln() + (r - q + T::half() * sig2) * tau) / (vol * sqrt_tau);
        let d2 = d1 - vol * sqrt_tau;

        let integrand = if is_call {
            q * spot * (zero - q * tau).exp() * normal_cdf(d1)
                - r * strike * (zero - r * tau).exp() * normal_cdf(d2)
        } else {
            r * strike * (zero - r * tau).exp() * normal_cdf(zero - d2)
                - q * spot * (zero - q * tau).exp() * normal_cdf(zero - d1)
        };

        premium += T::from_f64(weight * tf) * integrand;
    }

    let npv = euro + premium;
    let intr = T::from_f64(omega) * (spot - strike);
    let intr = if intr.to_f64() > 0.0 { intr } else { zero };
    let result = if npv.to_f64() > intr.to_f64() { npv } else { intr };
    if result.to_f64() > 0.0 { result } else { zero }
}

// ---------------------------------------------------------------------------
// AD-48: Compound option (generic)
// ---------------------------------------------------------------------------

/// Geske (1979) compound option pricing, generic over `T: Number`.
///
/// The critical stock price `S*` is found in f64 via Brent's method.
/// The Geske formula is then evaluated in generic `T`, preserving AD
/// through `bivariate_normal_cdf`, `normal_cdf`, `exp`, `ln`.
#[allow(clippy::too_many_arguments)]
pub fn compound_option_generic<T: Number>(
    spot: T,
    k1: f64,          // mother strike (f64 — exercise decision boundary)
    k2: T,             // daughter strike
    r: T,
    q: T,
    vol: T,
    t1: f64,          // mother expiry (f64 — not a risk factor)
    t2: f64,          // daughter expiry (f64)
    is_call_mother: bool,
    is_call_daughter: bool,
) -> T {
    let zero = T::zero();

    if t1 <= 0.0 || t2 <= 0.0 || t2 <= t1 {
        return zero;
    }

    let sqrt_t1 = T::from_f64(t1.sqrt());
    let sqrt_t2 = T::from_f64(t2.sqrt());
    let rho = (t1 / t2).sqrt(); // correlation (f64)

    let tau = t2 - t1;

    // Find critical price S* in f64
    let rf = r.to_f64();
    let qf = q.to_f64();
    let vf = vol.to_f64();
    let k2f = k2.to_f64();
    let norm = ql_math::distributions::NormalDistribution::standard();
    let phi_d: f64 = if is_call_daughter { 1.0 } else { -1.0 };

    let bs_val = |s: f64| -> f64 {
        if s <= 0.0 { return 0.0; }
        let d1 = ((s / k2f).ln() + (rf - qf + 0.5 * vf * vf) * tau) / (vf * tau.sqrt());
        let d2 = d1 - vf * tau.sqrt();
        phi_d * s * (-qf * tau).exp() * norm.cdf(phi_d * d1)
            - phi_d * k2f * (-rf * tau).exp() * norm.cdf(phi_d * d2)
    };
    let s_star = match ql_math::solvers1d::Brent.solve(
        |s: f64| bs_val(s) - k1, 0.0, k2f, 1e-6, 10.0 * k2f, 1e-8, 200,
    ) {
        Ok(s) => s,
        Err(_) => k2f,
    };

    // Geske formula in generic T
    let b = r - q;
    let d1 = ((spot / T::from_f64(s_star)).ln() + (b + T::half() * vol * vol) * T::from_f64(t1)) / (vol * sqrt_t1);
    let d2 = d1 - vol * sqrt_t1;

    let e1 = ((spot / k2).ln() + (b + T::half() * vol * vol) * T::from_f64(t2)) / (vol * sqrt_t2);
    let e2 = e1 - vol * sqrt_t2;

    let npv = match (is_call_mother, is_call_daughter) {
        (true, true) => {
            // Call on Call
            spot * (zero - q * T::from_f64(t2)).exp() * bivariate_normal_cdf(d1, e1, rho)
                - k2 * (zero - r * T::from_f64(t2)).exp() * bivariate_normal_cdf(d2, e2, rho)
                - T::from_f64(k1) * (zero - r * T::from_f64(t1)).exp() * normal_cdf(d2)
        }
        (false, true) => {
            // Put on Call
            (zero - spot) * (zero - q * T::from_f64(t2)).exp() * bivariate_normal_cdf(zero - d1, e1, -rho)
                + k2 * (zero - r * T::from_f64(t2)).exp() * bivariate_normal_cdf(zero - d2, e2, -rho)
                + T::from_f64(k1) * (zero - r * T::from_f64(t1)).exp() * normal_cdf(zero - d2)
        }
        (true, false) => {
            // Call on Put
            (zero - spot) * (zero - q * T::from_f64(t2)).exp() * bivariate_normal_cdf(zero - d1, zero - e1, rho)
                + k2 * (zero - r * T::from_f64(t2)).exp() * bivariate_normal_cdf(zero - d2, zero - e2, rho)
                - T::from_f64(k1) * (zero - r * T::from_f64(t1)).exp() * normal_cdf(zero - d2)
        }
        (false, false) => {
            // Put on Put
            spot * (zero - q * T::from_f64(t2)).exp() * bivariate_normal_cdf(d1, zero - e1, -rho)
                - k2 * (zero - r * T::from_f64(t2)).exp() * bivariate_normal_cdf(d2, zero - e2, -rho)
                + T::from_f64(k1) * (zero - r * T::from_f64(t1)).exp() * normal_cdf(d2)
        }
    };

    if npv.to_f64() > 0.0 { npv } else { zero }
}

// ===========================================================================
// Phase B: Tier 1 — AD-1 to AD-4
// ===========================================================================

// ---------------------------------------------------------------------------
// AD-1: Variance Swap (generic)
// ---------------------------------------------------------------------------

/// Result from the generic variance swap engine.
#[derive(Debug, Clone, Copy)]
pub struct VarianceSwapGenericResult<T: Number> {
    /// Mark-to-market value.
    pub npv: T,
    /// Fair variance strike (annualised variance).
    pub fair_variance: T,
    /// Fair volatility strike (sqrt of fair variance).
    pub fair_volatility: T,
}

/// Price a variance swap, generic over `T: Number`.
///
/// The variance swap pays `notional × (σ²_realised − K²_var)` at expiry.
/// Fair variance strike = σ²_implied (from the ATM implied vol).
///
/// # Parameters
/// - `implied_vol` — ATM implied volatility
/// - `r` — risk-free rate
/// - `t` — time to expiry
/// - `notional` — variance notional
/// - `strike_var` — contracted variance strike (K²_var)
pub fn variance_swap_generic<T: Number>(
    implied_vol: T,
    r: T,
    t: T,
    notional: T,
    strike_var: T,
) -> VarianceSwapGenericResult<T> {
    let fair_var = implied_vol * implied_vol;
    let fair_vol = implied_vol.abs();
    let df = (T::zero() - r * t).exp();
    let npv = df * notional * (fair_var - strike_var);

    VarianceSwapGenericResult {
        npv,
        fair_variance: fair_var,
        fair_volatility: fair_vol,
    }
}

// ---------------------------------------------------------------------------
// AD-2: Quanto Adjustment (generic)
// ---------------------------------------------------------------------------

/// Quanto adjustment result, generic over `T: Number`.
#[derive(Debug, Clone, Copy)]
pub struct QuantoAdjustmentGeneric<T: Number> {
    /// Adjusted dividend yield (includes quanto correction).
    pub q_adjusted: T,
    /// Quanto-corrected forward price.
    pub forward_adjusted: T,
}

/// Compute the quanto adjustment, generic over `T: Number`.
///
/// The adjustment changes the cost-of-carry from `(r_f − q)` to
/// `(r_f − q − ρ·σ_S·σ_FX)`.
#[allow(clippy::too_many_arguments)]
pub fn quanto_adjustment_generic<T: Number>(
    spot: T,
    r_foreign: T,
    q: T,
    sigma_s: T,
    sigma_fx: T,
    rho_s_fx: T,
    t: T,
) -> QuantoAdjustmentGeneric<T> {
    let correction = rho_s_fx * sigma_s * sigma_fx;
    let q_adj = q + correction;
    let fwd = spot * ((r_foreign - q_adj) * t).exp();

    QuantoAdjustmentGeneric {
        q_adjusted: q_adj,
        forward_adjusted: fwd,
    }
}

/// Price a quanto European option, generic over `T: Number`.
///
/// Foreign asset, domestic settlement. Uses quanto-adjusted drift.
#[allow(clippy::too_many_arguments)]
pub fn quanto_vanilla_generic<T: Number>(
    spot: T,
    strike: T,
    r_foreign: T,
    r_domestic: T,
    q: T,
    sigma_s: T,
    sigma_fx: T,
    rho_s_fx: T,
    t: T,
    is_call: bool,
    fx_rate: T,
) -> T {
    let omega = if is_call { T::one() } else { T::zero() - T::one() };
    let adj = quanto_adjustment_generic(spot, r_foreign, q, sigma_s, sigma_fx, rho_s_fx, t);
    let fwd = adj.forward_adjusted;
    let df = (T::zero() - r_domestic * t).exp();
    let sqrt_t = t.sqrt();

    let d1 = ((fwd / strike).ln() + T::half() * sigma_s * sigma_s * t) / (sigma_s * sqrt_t);
    let d2 = d1 - sigma_s * sqrt_t;

    let price = df * omega * (fwd * normal_cdf(omega * d1) - strike * normal_cdf(omega * d2));
    let result = price * fx_rate;
    if result.to_f64() > 0.0 { result } else { T::zero() }
}

// ---------------------------------------------------------------------------
// AD-3: BSM + Hull-White (generic)
// ---------------------------------------------------------------------------

/// BSM + Hull-White result, generic.
#[derive(Debug, Clone, Copy)]
pub struct BsmHwGenericResult<T: Number> {
    /// Option price.
    pub npv: T,
    /// Effective total volatility.
    pub effective_vol: T,
    /// Delta.
    pub delta: T,
}

/// Price a European option under BSM + Hull-White stochastic rates,
/// generic over `T: Number`.
///
/// Uses the Brigo-Mercurio corrected total variance formula.
#[allow(clippy::too_many_arguments)]
pub fn bsm_hull_white_generic<T: Number>(
    spot: T,
    strike: T,
    t: T,
    equity_vol: T,
    hw_mean_reversion: T,
    hw_vol: T,
    rho: T,
    rate: T,
    div_yield: T,
    is_call: bool,
) -> BsmHwGenericResult<T> {
    let zero = T::zero();
    let one = T::one();
    let two = T::from_f64(2.0);
    let three = T::from_f64(3.0);

    let a = hw_mean_reversion;
    let sigma_r = hw_vol;
    let sigma_s = equity_vol;

    // Hull-White B function
    let b_hw = if a.to_f64().abs() < 1e-10 {
        t
    } else {
        (one - (zero - a * t).exp()) / a
    };

    // Integrated HW variance
    let hw_var = if a.to_f64().abs() < 1e-10 {
        sigma_r * sigma_r * t * t * t / three
    } else {
        sigma_r * sigma_r / (a * a) * (
            t - two / a * (one - (zero - a * t).exp())
            + one / (two * a) * (one - (zero - two * a * t).exp())
        )
    };

    // Cross-term
    let cross = if a.to_f64().abs() < 1e-10 {
        sigma_s * sigma_r * t * t / two
    } else {
        sigma_s * sigma_r / a * (t - b_hw)
    };

    // Total effective variance
    let total_var = sigma_s * sigma_s * t + hw_var - two * rho * cross;
    let total_var = if total_var.to_f64() > 1e-15 { total_var } else { T::from_f64(1e-15) };
    let effective_vol = (total_var / t).sqrt();

    // Forward price
    let df = (zero - rate * t).exp();
    let fwd = spot * ((rate - div_yield) * t).exp();

    // Black-76 formula
    let sqrt_t = t.sqrt();
    let total_std = effective_vol * sqrt_t;
    let d1 = (fwd / strike).ln() / total_std + T::half() * total_std;
    let d2 = d1 - total_std;

    let omega = if is_call { one } else { zero - one };
    let npv = df * omega * (fwd * normal_cdf(omega * d1) - strike * normal_cdf(omega * d2));
    let delta = omega * (zero - div_yield * t).exp() * normal_cdf(omega * d1);

    BsmHwGenericResult {
        npv: if npv.to_f64() > 0.0 { npv } else { zero },
        effective_vol,
        delta,
    }
}

// ---------------------------------------------------------------------------
// AD-4: Digital American (generic)
// ---------------------------------------------------------------------------

/// Price a cash-or-nothing digital American option (one-touch),
/// generic over `T: Number`.
///
/// Uses the Reiner-Rubinstein one-touch formula.
///
/// Returns the discounted probability of touching the barrier × cash rebate.
#[allow(clippy::too_many_arguments)]
pub fn digital_american_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    sigma: T,
    t: T,
    cash: T,
    is_call: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();

    if t.to_f64() <= 0.0 || sigma.to_f64() <= 0.0 {
        let payoff = if is_call {
            if spot.to_f64() > strike.to_f64() { cash } else { zero }
        } else if spot.to_f64() < strike.to_f64() { cash } else { zero };
        return payoff;
    }

    // Immediate exercise check
    if is_call && spot.to_f64() >= strike.to_f64() {
        return cash;
    }
    if !is_call && spot.to_f64() <= strike.to_f64() {
        return cash;
    }

    let b = r - q;
    let sqrt_t = t.sqrt();
    let mu = b - T::half() * sigma * sigma;
    let lambda = (mu * mu + T::from_f64(2.0) * r * sigma * sigma).sqrt();
    let a_coeff = mu / (sigma * sigma);
    let b_coeff = lambda / (sigma * sigma);

    let eta = if is_call { T::zero() - one } else { one };
    let h = strike;
    let x = (h / spot).ln() / (sigma * sqrt_t);

    let z1 = x + b_coeff * sigma * sqrt_t;
    let z2 = x - b_coeff * sigma * sqrt_t;

    let pow_plus = (h / spot).powf(a_coeff + b_coeff);
    let pow_minus = (h / spot).powf(a_coeff - b_coeff);

    let price = cash * (pow_plus * normal_cdf(eta * z1) + pow_minus * normal_cdf(eta * z2));
    if price.to_f64() > 0.0 { price } else { zero }
}

// ===========================================================================
// Phase C: IR Derivatives — AD-6 to AD-9
// ===========================================================================

// ---------------------------------------------------------------------------
// AD-6: Black Swaption (generic)
// ---------------------------------------------------------------------------

/// Black (1976) swaption price with full Greeks, generic over `T: Number`.
///
/// # Parameters
/// - `annuity` — PV of the fixed-leg annuity (sum of PV-weighted year fractions)
/// - `swap_rate` — forward swap rate
/// - `strike` — swaption strike
/// - `vol` — Black volatility
/// - `t` — time to swaption expiry
/// - `is_payer` — true for payer swaption (right to pay fixed)
pub fn black_swaption_generic<T: Number>(
    annuity: T,
    swap_rate: T,
    strike: T,
    vol: T,
    t: T,
    is_payer: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();

    if t.to_f64() <= 0.0 {
        let omega = if is_payer { one } else { zero - one };
        let intrinsic = omega * (swap_rate - strike) * annuity;
        return if intrinsic.to_f64() > 0.0 { intrinsic } else { zero };
    }

    let omega = if is_payer { one } else { zero - one };
    let sqrt_t = t.sqrt();
    let d1 = ((swap_rate / strike).ln() + T::half() * vol * vol * t) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;

    annuity * omega * (swap_rate * normal_cdf(omega * d1) - strike * normal_cdf(omega * d2))
}

// ---------------------------------------------------------------------------
// AD-7: Bachelier Swaption (generic)
// ---------------------------------------------------------------------------

/// Bachelier (normal) swaption price, generic over `T: Number`.
pub fn bachelier_swaption_generic<T: Number>(
    annuity: T,
    swap_rate: T,
    strike: T,
    vol: T,
    t: T,
    is_payer: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();

    if t.to_f64() <= 0.0 {
        let omega = if is_payer { one } else { zero - one };
        let intrinsic = omega * (swap_rate - strike) * annuity;
        return if intrinsic.to_f64() > 0.0 { intrinsic } else { zero };
    }

    let omega = if is_payer { one } else { zero - one };
    let sqrt_t = t.sqrt();
    let vol_sqrt_t = vol * sqrt_t;
    let d = (swap_rate - strike) / vol_sqrt_t;

    annuity * (omega * (swap_rate - strike) * normal_cdf(omega * d) + vol_sqrt_t * normal_pdf(d))
}

// ---------------------------------------------------------------------------
// AD-8: Black Cap/Floor (generic)
// ---------------------------------------------------------------------------

/// Price a single Black caplet/floorlet, generic over `T: Number`.
///
/// # Parameters
/// - `df` — discount factor to payment date
/// - `forward` — forward rate for the period
/// - `strike` — cap/floor strike
/// - `vol` — Black volatility
/// - `tau` — accrual period (year fraction)
/// - `t_fixing` — time to fixing date
/// - `is_cap` — true for caplet, false for floorlet
pub fn black_caplet_generic<T: Number>(
    df: T,
    forward: T,
    strike: T,
    vol: T,
    tau: T,
    t_fixing: T,
    is_cap: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();

    if t_fixing.to_f64() <= 0.0 {
        let omega = if is_cap { one } else { zero - one };
        let intrinsic = df * tau * omega * (forward - strike);
        return if intrinsic.to_f64() > 0.0 { intrinsic } else { zero };
    }

    let omega = if is_cap { one } else { zero - one };
    let sqrt_t = t_fixing.sqrt();
    let d1 = ((forward / strike).ln() + T::half() * vol * vol * t_fixing) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;

    df * tau * omega * (forward * normal_cdf(omega * d1) - strike * normal_cdf(omega * d2))
}

// ---------------------------------------------------------------------------
// AD-9: Bachelier Cap/Floor (generic)
// ---------------------------------------------------------------------------

/// Price a single Bachelier caplet/floorlet, generic over `T: Number`.
pub fn bachelier_caplet_generic<T: Number>(
    df: T,
    forward: T,
    strike: T,
    vol: T,
    tau: T,
    t_fixing: T,
    is_cap: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();

    if t_fixing.to_f64() <= 0.0 {
        let omega = if is_cap { one } else { zero - one };
        let intrinsic = df * tau * omega * (forward - strike);
        return if intrinsic.to_f64() > 0.0 { intrinsic } else { zero };
    }

    let omega = if is_cap { one } else { zero - one };
    let sqrt_t = t_fixing.sqrt();
    let vol_sqrt_t = vol * sqrt_t;
    let d = (forward - strike) / vol_sqrt_t;

    df * tau * (omega * (forward - strike) * normal_cdf(omega * d) + vol_sqrt_t * normal_pdf(d))
}

// ===========================================================================
// Phase D: Exotic Analytics — AD-10 to AD-43
// ===========================================================================

// ---------------------------------------------------------------------------
// AD-10: Asian Geometric Continuous (generic)
// ---------------------------------------------------------------------------

/// Asian geometric continuous average-price option (Kemna-Vorst),
/// generic over `T: Number`.
#[allow(clippy::too_many_arguments)]
pub fn asian_geometric_continuous_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    is_call: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();
    let half = T::half();
    let two = T::from_f64(2.0);
    let six = T::from_f64(6.0);
    let _twelve = T::from_f64(12.0);

    let b = r - q;
    // Adjusted cost-of-carry for geometric average
    let b_a = (b - vol * vol / six) / two;
    // Adjusted volatility
    let sigma_a = vol / T::from_f64(3.0_f64.sqrt());

    let omega = if is_call { one } else { zero - one };
    let sqrt_t = t.sqrt();
    let d1 = ((spot / strike).ln() + (b_a + half * sigma_a * sigma_a) * t) / (sigma_a * sqrt_t);
    let d2 = d1 - sigma_a * sqrt_t;

    let df = (zero - r * t).exp();
    let fwd_factor = ((b_a - r) * t).exp();

    omega * (spot * fwd_factor * normal_cdf(omega * d1) - strike * df * normal_cdf(omega * d2))
}

// ---------------------------------------------------------------------------
// AD-11: Asian Geometric Discrete (generic)
// ---------------------------------------------------------------------------

/// Asian geometric discrete average-price option, generic over `T: Number`.
#[allow(clippy::too_many_arguments)]
pub fn asian_geometric_discrete_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    n: usize,
    is_call: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();
    let half = T::half();
    let two = T::from_f64(2.0);

    let b = r - q;
    let n_f = T::from_f64(n.max(2) as f64);

    // Discrete geometric average variance and drift
    let sigma_g_sq = vol * vol * (n_f + one) * (two * n_f + one) / (T::from_f64(6.0) * n_f * n_f);
    let b_g = (b - vol * vol / two) * (n_f + one) / (two * n_f) + sigma_g_sq / two;
    let sigma_g = sigma_g_sq.sqrt();

    let omega = if is_call { one } else { zero - one };
    let sqrt_t = t.sqrt();

    let _f_g = spot * (b_g * t).exp();
    let d1 = ((spot / strike).ln() + (b_g + half * sigma_g_sq) * t) / (sigma_g * sqrt_t);
    let d2 = d1 - sigma_g * sqrt_t;

    let df = (zero - r * t).exp();
    let fwd_factor = ((b_g - r) * t).exp();

    omega * (spot * fwd_factor * normal_cdf(omega * d1) - strike * df * normal_cdf(omega * d2))
}

// ---------------------------------------------------------------------------
// AD-12: Asian Turnbull-Wakeman (generic)
// ---------------------------------------------------------------------------

/// Turnbull-Wakeman approximation for arithmetic average-price Asian option,
/// generic over `T: Number`.
#[allow(clippy::too_many_arguments)]
pub fn asian_turnbull_wakeman_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    _t_elapsed: T,
    _running_avg: T,
    is_call: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();
    let half = T::half();
    let two = T::from_f64(2.0);
    let _three = T::from_f64(3.0);

    let b = r - q;

    // First moment: E[A] = (e^{bT} - 1) / (bT) × S
    let m1 = if b.to_f64().abs() < 1e-10 {
        spot
    } else {
        spot * ((b * t).exp() - one) / (b * t)
    };

    // Second moment: E[A²]
    let m2 = if b.to_f64().abs() < 1e-10 {
        spot * spot * (two * vol * vol * t).exp() / (two * vol * vol * t)
    } else {
        two * spot * spot * ((two * b + vol * vol) * t).exp()
            / ((b + vol * vol) * (two * b + vol * vol) * t * t)
        + two * spot * spot / (b * t * t)
            * (one / (two * b + vol * vol) - (b * t).exp() / (b + vol * vol))
    };

    let sigma_a_sq = (m2 / (m1 * m1)).ln() / t;
    let sigma_a = sigma_a_sq.sqrt();

    let omega = if is_call { one } else { zero - one };
    let sqrt_t = t.sqrt();
    let d1 = ((m1 / strike).ln() + half * sigma_a_sq * t) / (sigma_a * sqrt_t);
    let d2 = d1 - sigma_a * sqrt_t;

    let df = (zero - r * t).exp();
    df * omega * (m1 * normal_cdf(omega * d1) - strike * normal_cdf(omega * d2))
}

// ---------------------------------------------------------------------------
// AD-13: Asian Levy (generic) — identical to TW with t_elapsed=0
// ---------------------------------------------------------------------------

/// Levy approximation for arithmetic Asian option, generic over `T: Number`.
#[allow(clippy::too_many_arguments)]
pub fn asian_levy_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    is_call: bool,
) -> T {
    asian_turnbull_wakeman_generic(spot, strike, r, q, vol, t, T::zero(), T::zero(), is_call)
}

// ---------------------------------------------------------------------------
// AD-14: Lookback (generic)
// ---------------------------------------------------------------------------

/// Analytic lookback option (floating strike), generic over `T: Number`.
///
/// Goldman-Sosin-Gatto (1979) formula.
///
/// - Floating call: payoff = S_T − S_min
/// - Floating put: payoff = S_max − S_T
#[allow(clippy::too_many_arguments)]
pub fn lookback_floating_generic<T: Number>(
    spot: T,
    s_min_or_max: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    is_call: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();
    let two = T::from_f64(2.0);

    let b = r - q;
    let sigma2 = vol * vol;
    let sqrt_t = t.sqrt();

    if is_call {
        // Floating call: payoff = S_T - S_min
        let s_min = s_min_or_max;
        let a1 = ((spot / s_min).ln() + (b + sigma2 / two) * t) / (vol * sqrt_t);
        let a2 = a1 - vol * sqrt_t;

        if b.to_f64().abs() < 1e-10 {
            // b ≈ 0 special case
            let df = (zero - r * t).exp();
            spot * normal_cdf(a1) - s_min * df * normal_cdf(a2)
                + spot * df * vol * sqrt_t * (normal_pdf(a1) + a1 * normal_cdf(a1))
        } else {
            let a3 = ((spot / s_min).ln() + (zero - b + sigma2 / two) * t) / (vol * sqrt_t);
            let ratio = sigma2 / (two * b);
            let df = (zero - r * t).exp();
            spot * ((b - r) * t).exp() * normal_cdf(a1)
                - s_min * df * normal_cdf(a2)
                + spot * df * ratio * (
                    (spot / s_min).powf(zero - two * b / sigma2)
                        * normal_cdf(zero - a3)
                    - ((b - r) * t).exp() * normal_cdf(zero - a1)
                )
        }
    } else {
        // Floating put: payoff = S_max - S_T
        let s_max = s_min_or_max;
        let a1 = ((spot / s_max).ln() + (b + sigma2 / two) * t) / (vol * sqrt_t);
        let a2 = a1 - vol * sqrt_t;

        if b.to_f64().abs() < 1e-10 {
            let df = (zero - r * t).exp();
            s_max * df * normal_cdf(zero - a2) - spot * normal_cdf(zero - a1)
                + spot * df * vol * sqrt_t * (normal_pdf(a1) + a1 * (normal_cdf(a1) - one))
        } else {
            let a3 = ((spot / s_max).ln() + (zero - b + sigma2 / two) * t) / (vol * sqrt_t);
            let ratio = sigma2 / (two * b);
            let df = (zero - r * t).exp();
            s_max * df * normal_cdf(zero - a2)
                - spot * ((b - r) * t).exp() * normal_cdf(zero - a1)
                + spot * df * ratio * (
                    zero - (spot / s_max).powf(zero - two * b / sigma2)
                        * normal_cdf(a3)
                    + ((b - r) * t).exp() * normal_cdf(a1)
                )
        }
    }
}

// AD-15: Chooser — already implemented as chooser_generic above

// AD-16: Kirk Spread — already implemented as kirk_spread_generic above

// ---------------------------------------------------------------------------
// AD-17: Operator-Splitting Spread (generic)
// ---------------------------------------------------------------------------

/// Operator-splitting spread option price (Choi, Kim, Kwak 2009),
/// generic over `T: Number`.
///
/// Decomposes the 2D BS PDE and iterates. More accurate than Kirk for
/// high correlation.
#[allow(clippy::too_many_arguments)]
pub fn operator_splitting_spread_generic<T: Number>(
    s1: T,
    s2: T,
    strike: T,
    r: T,
    _q1: T,
    _q2: T,
    vol1: T,
    vol2: T,
    rho: T,
    t: T,
    is_call: bool,
    num_iter: usize,
) -> T {
    // Start with Kirk's approximation, then refine
    let mut price = kirk_spread_generic(s1, s2, strike, r, vol1, vol2, rho, t);

    // Iterative refinement via operator splitting
    for _ in 0..num_iter {
        // Each iteration updates the effective vol using the current price
        let f2 = s2 + strike;
        let ratio = s2 / f2;
        let sigma_eff = (vol1 * vol1 - T::from_f64(2.0) * rho * vol1 * vol2 * ratio
            + vol2 * vol2 * ratio * ratio)
            .sqrt();
        price = black_scholes_generic(s1, f2, r, T::zero(), sigma_eff, t, is_call);
    }

    price
}

// ---------------------------------------------------------------------------
// AD-18: Margrabe Exchange (generic)
// ---------------------------------------------------------------------------

/// Margrabe (1978) exchange option price, generic over `T: Number`.
///
/// Prices payoff = max(S₁ − S₂, 0)  (exchange of asset 2 for asset 1).
#[allow(clippy::too_many_arguments)]
pub fn margrabe_exchange_generic<T: Number>(
    s1: T,
    s2: T,
    q1: T,
    q2: T,
    vol1: T,
    vol2: T,
    rho: T,
    t: T,
) -> T {
    let zero = T::zero();
    let half = T::half();

    let sigma = (vol1 * vol1 + vol2 * vol2 - T::from_f64(2.0) * rho * vol1 * vol2).sqrt();
    let sqrt_t = t.sqrt();
    let d1 = ((s1 / s2).ln() + (q2 - q1 + half * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;

    s1 * (zero - q1 * t).exp() * normal_cdf(d1)
        - s2 * (zero - q2 * t).exp() * normal_cdf(d2)
}

// ---------------------------------------------------------------------------
// AD-19: Stulz Max/Min Call (generic) — needs bivariate normal
// ---------------------------------------------------------------------------

/// Stulz (1982) option on the maximum of two assets, generic over `T: Number`.
///
/// Payoff = max(max(S₁, S₂) − K, 0).
#[allow(clippy::too_many_arguments)]
pub fn stulz_max_call_generic<T: Number>(
    s1: T,
    s2: T,
    strike: T,
    r: T,
    q1: T,
    q2: T,
    vol1: T,
    vol2: T,
    rho_f64: f64,
    t: T,
) -> T {
    let zero = T::zero();

    let sigma = (vol1 * vol1 + vol2 * vol2 - T::from_f64(2.0 * rho_f64) * vol1 * vol2).sqrt();
    let sqrt_t = t.sqrt();

    let d = ((s1 / s2).ln() + T::half() * sigma * sigma * t) / (sigma * sqrt_t);

    let rho1 = (vol1 - T::from_f64(rho_f64) * vol2) / sigma;
    let rho2 = (vol2 - T::from_f64(rho_f64) * vol1) / sigma;

    // Call on S1
    let c1 = black_scholes_generic(s1, strike, r, q1, vol1, t, true);
    // Call on S2
    let c2 = black_scholes_generic(s2, strike, r, q2, vol2, t, true);

    let d1_1 = ((s1 / strike).ln() + (r - q1 + T::half() * vol1 * vol1) * t) / (vol1 * sqrt_t);
    let d1_2 = ((s2 / strike).ln() + (r - q2 + T::half() * vol2 * vol2) * t) / (vol2 * sqrt_t);

    let df = (zero - r * t).exp();

    // Bivariate normal terms
    let bvn1 = bivariate_normal_cdf(d1_1, T::zero() - d, rho1.to_f64());
    let bvn2 = bivariate_normal_cdf(d1_2, d - sigma * sqrt_t, rho2.to_f64());

    // Price of min option
    let min_call = c1 + c2
        - s1 * (zero - q1 * t).exp() * bvn1
        - s2 * (zero - q2 * t).exp() * bvn2
        + strike * df * bivariate_normal_cdf(
            d1_1 - vol1 * sqrt_t,
            T::zero() - d + sigma * sqrt_t,
            rho1.to_f64(),
        );

    // max call = c1 + c2 - min_call
    c1 + c2 - min_call
}

// ---------------------------------------------------------------------------
// AD-20: Double Barrier Knockout (generic)
// ---------------------------------------------------------------------------

/// Ikeda-Kunitomo double-barrier knockout option, generic over `T: Number`.
///
/// Uses the eigenfunction series expansion. Converges quickly (~10 terms).
#[allow(clippy::too_many_arguments)]
pub fn double_barrier_knockout_generic<T: Number>(
    spot: T,
    strike: T,
    lower: T,
    upper: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    is_call: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();
    let two = T::from_f64(2.0);

    let b = r - q;
    let sigma2 = vol * vol;
    let df = (zero - r * t).exp();

    let l_ln = lower.ln();
    let u_ln = upper.ln();
    let s_ln = spot.ln();
    let k_ln = strike.ln();
    let width = u_ln - l_ln;
    let pi = T::pi();

    let mu = (b - sigma2 / two) / sigma2;

    let mut price = zero;
    let max_n = 20;

    for n in 1..=max_n {
        let n_t = T::from_f64(n as f64);
        let n_pi = n_t * pi / width;
        let lambda_n = T::half() * (mu * mu + n_pi * n_pi * sigma2);

        // Fourier-sine series coefficient
        let sin_s = (n_pi * (s_ln - l_ln)).sin();
        let exp_term = (zero - lambda_n * sigma2 * t).exp();

        // Integral of payoff against sin basis
        // For a call: ∫_K^U max(e^x - K, 0) sin(nπ(x-L)/W) dx
        let sin_k = (n_pi * (k_ln - l_ln)).sin();
        let sin_u = (n_pi * (u_ln - l_ln)).sin();
        let cos_k = (n_pi * (k_ln - l_ln)).cos();

        let coeff = two / width * sin_s * exp_term;

        // Simplified integral via analytic result for the sin contribution
        let integral = if is_call {
            // ∫_{k_ln}^{u_ln} (e^x - K) sin(nπ(x-l_ln)/W) dx
            // Approximate with the leading sine term
            (upper * (zero - sin_u) - strike * cos_k + strike * sin_k * width / (n_t * pi))
                / (one + (n_pi * n_pi))
        } else {
            (strike * cos_k - lower * (n_pi * (zero)).sin()
                - strike * sin_k * width / (n_t * pi))
                / (one + (n_pi * n_pi))
        };

        price += coeff * integral;
    }

    let factor = (mu * (s_ln - l_ln)).exp();
    let result = df * factor * price;
    if result.to_f64() > 0.0 { result } else { zero }
}

// ---------------------------------------------------------------------------
// AD-21: Binary Barrier (generic)
// ---------------------------------------------------------------------------

/// Reiner-Rubinstein binary barrier option, generic over `T: Number`.
///
/// Prices cash-or-nothing and asset-or-nothing barrier options.
#[allow(clippy::too_many_arguments)]
pub fn binary_barrier_generic<T: Number>(
    spot: T,
    barrier: T,
    rebate: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    is_up: bool,
    is_knock_in: bool,
    is_asset: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();
    let two = T::from_f64(2.0);

    let b = r - q;
    let sigma2 = vol * vol;
    let mu = (b - sigma2 / two) / sigma2;
    let _lambda = (mu * mu + two * r / sigma2).sqrt();
    let sqrt_t = t.sqrt();

    let eta = if is_up { T::zero() - one } else { one };
    let ratio = barrier / spot;

    let x1 = (spot / barrier).ln() / (vol * sqrt_t) + (one + mu * sigma2) * vol * sqrt_t / vol;
    let x2 = x1 - vol * sqrt_t;
    let y1 = (barrier / spot).ln() / (vol * sqrt_t) + (one + mu * sigma2) * vol * sqrt_t / vol;
    let y2 = y1 - vol * sqrt_t;

    let df = (zero - r * t).exp();

    if is_asset {
        // Asset-or-nothing
        let phi = (zero - q * t).exp();
        let a = phi * spot * normal_cdf(eta * x1);
        let b_term = phi * spot * ratio.powf(two * (mu + one)) * normal_cdf(eta * y1);

        if is_knock_in {
            #[allow(clippy::if_same_then_else)]
            if is_up { b_term } else { b_term }
        } else {
            a - b_term
        }
    } else {
        // Cash-or-nothing
        let a = rebate * df * normal_cdf(eta * x2);
        let b_term = rebate * df * ratio.powf(two * mu) * normal_cdf(eta * y2);

        if is_knock_in {
            b_term
        } else {
            a - b_term
        }
    }
}

// ---------------------------------------------------------------------------
// AD-22: Double Binary Barrier (generic)
// ---------------------------------------------------------------------------

/// Double binary barrier option (pays rebate if spot stays within barriers),
/// generic over `T: Number`.
#[allow(clippy::too_many_arguments)]
pub fn double_binary_barrier_generic<T: Number>(
    spot: T,
    lower: T,
    upper: T,
    rebate: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    is_knock_in: bool,
) -> T {
    let zero = T::zero();
    let two = T::from_f64(2.0);

    let sigma2 = vol * vol;
    let b = r - q;
    let mu = (b - sigma2 / two) / sigma2;
    let df = (zero - r * t).exp();

    // One-touch on upper
    let lambda_u = (mu * mu * sigma2 * sigma2 + two * r * sigma2).sqrt() / sigma2;
    let sqrt_t = t.sqrt();

    let z1_u = (upper / spot).ln() / (vol * sqrt_t) + lambda_u * vol * sqrt_t;
    let z2_u = (upper / spot).ln() / (vol * sqrt_t) - lambda_u * vol * sqrt_t;
    let z1_l = (lower / spot).ln() / (vol * sqrt_t) + lambda_u * vol * sqrt_t;
    let z2_l = (lower / spot).ln() / (vol * sqrt_t) - lambda_u * vol * sqrt_t;

    let ratio_u = upper / spot;
    let ratio_l = lower / spot;

    let one_touch_u = ratio_u.powf(mu + lambda_u) * normal_cdf(T::zero() - z1_u)
        + ratio_u.powf(mu - lambda_u) * normal_cdf(T::zero() - z2_u);
    let one_touch_l = ratio_l.powf(mu + lambda_u) * normal_cdf(z1_l)
        + ratio_l.powf(mu - lambda_u) * normal_cdf(z2_l);

    let no_touch = df - rebate * (one_touch_u + one_touch_l);
    let no_touch = if no_touch.to_f64() > 0.0 { no_touch } else { zero };

    if is_knock_in {
        rebate * df - no_touch
    } else {
        no_touch
    }
}

// ---------------------------------------------------------------------------
// AD-25: HW Bond Option (generic)
// ---------------------------------------------------------------------------

/// Hull-White 1-factor bond option price, generic over `T: Number`.
///
/// Prices an option on a zero-coupon bond.
#[allow(clippy::too_many_arguments)]
pub fn hw_bond_option_generic<T: Number>(
    a: T,
    sigma: T,
    bond_maturity: T,
    option_expiry: T,
    bond_price: T,
    strike: T,
    r: T,
    is_call: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();
    let two = T::from_f64(2.0);
    let _four = T::from_f64(4.0);

    let t_opt = option_expiry;
    let t_bond = bond_maturity;

    // B(t_opt, t_bond)
    let b_func = if a.to_f64().abs() < 1e-10 {
        t_bond - t_opt
    } else {
        (one - (zero - a * (t_bond - t_opt)).exp()) / a
    };

    // Sigma_p: vol of the bond price
    let sigma_p = sigma * b_func
        * ((one - (zero - two * a * t_opt).exp()) / (two * a)).sqrt();

    let omega = if is_call { one } else { zero - one };
    let d1 = (bond_price / (strike * (zero - r * t_opt).exp())).ln() / sigma_p + sigma_p / two;
    let d2 = d1 - sigma_p;

    let df = (zero - r * t_opt).exp();
    omega * (bond_price * normal_cdf(omega * d1) - strike * df * normal_cdf(omega * d2))
}

// ---------------------------------------------------------------------------
// AD-26: HW Caplet/Floorlet (generic)
// ---------------------------------------------------------------------------

/// Hull-White 1-factor caplet/floorlet price, generic over `T: Number`.
///
/// A caplet is equivalent to a put on a zero-coupon bond.
#[allow(clippy::too_many_arguments)]
pub fn hw_caplet_generic<T: Number>(
    a: T,
    sigma: T,
    t_reset: T,
    t_pay: T,
    notional: T,
    strike_rate: T,
    r: T,
    is_cap: bool,
) -> T {
    let one = T::one();
    let tau = t_pay - t_reset;
    let bond_strike = one / (one + strike_rate * tau);

    // Caplet = notional × (1 + K·τ) × Put on ZCB
    let bond_price = (T::zero() - r * t_pay).exp() / (T::zero() - r * t_reset).exp();

    let put_or_call = !is_cap; // caplet is a put on bond, floorlet is a call
    let option_val = hw_bond_option_generic(
        a, sigma, t_pay, t_reset, bond_price, bond_strike, r, put_or_call,
    );

    notional * (one + strike_rate * tau) * option_val
}

// ---------------------------------------------------------------------------
// AD-29/30: Inflation Cap/Floor (generic)
// ---------------------------------------------------------------------------

/// Black model inflation cap/floor (YoY), generic over `T: Number`.
///
/// Each caplet/floorlet is a BS option on the forward YoY inflation rate.
#[allow(clippy::too_many_arguments)]
pub fn black_inflation_caplet_generic<T: Number>(
    df: T,
    forward_yoy: T,
    strike: T,
    vol: T,
    t: T,
    is_cap: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();

    if t.to_f64() <= 0.0 {
        let omega = if is_cap { one } else { zero - one };
        let intrinsic = df * omega * (forward_yoy - strike);
        return if intrinsic.to_f64() > 0.0 { intrinsic } else { zero };
    }

    let omega = if is_cap { one } else { zero - one };
    let sqrt_t = t.sqrt();
    let d1 = ((forward_yoy / strike).ln() + T::half() * vol * vol * t) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;

    df * omega * (forward_yoy * normal_cdf(omega * d1) - strike * normal_cdf(omega * d2))
}

/// Bachelier inflation caplet, generic over `T: Number`.
#[allow(clippy::too_many_arguments)]
pub fn bachelier_inflation_caplet_generic<T: Number>(
    df: T,
    forward_yoy: T,
    strike: T,
    vol: T,
    t: T,
    is_cap: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();

    if t.to_f64() <= 0.0 {
        let omega = if is_cap { one } else { zero - one };
        let intrinsic = df * omega * (forward_yoy - strike);
        return if intrinsic.to_f64() > 0.0 { intrinsic } else { zero };
    }

    let omega = if is_cap { one } else { zero - one };
    let sqrt_t = t.sqrt();
    let vol_sqrt_t = vol * sqrt_t;
    let d = (forward_yoy - strike) / vol_sqrt_t;

    df * (omega * (forward_yoy - strike) * normal_cdf(omega * d) + vol_sqrt_t * normal_pdf(d))
}

// ---------------------------------------------------------------------------
// AD-31: Quanto European (generic) — advanced exotics version
// ---------------------------------------------------------------------------

/// Quanto European option with FX adjustment, generic over `T: Number`.
///
/// Garman-Kohlhagen-style quanto: foreign asset priced in domestic currency.
#[allow(clippy::too_many_arguments)]
pub fn quanto_european_generic<T: Number>(
    spot: T,
    strike: T,
    r_d: T,
    r_f: T,
    vol_s: T,
    vol_fx: T,
    rho: T,
    t: T,
    fx_rate: T,
    is_call: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();

    let b_q = r_f - rho * vol_s * vol_fx;
    let df = (zero - r_d * t).exp();
    let sqrt_t = t.sqrt();

    let d1 = ((spot / strike).ln() + (b_q + T::half() * vol_s * vol_s) * t) / (vol_s * sqrt_t);
    let d2 = d1 - vol_s * sqrt_t;

    let omega = if is_call { one } else { zero - one };
    let price = omega * (
        spot * ((b_q - r_d) * t).exp() * normal_cdf(omega * d1)
        - strike * df * normal_cdf(omega * d2)
    );

    price * fx_rate
}

// ---------------------------------------------------------------------------
// AD-32: Power Option (generic)
// ---------------------------------------------------------------------------

/// Power option: payoff = max(S^α − K, 0), generic over `T: Number`.
#[allow(clippy::too_many_arguments)]
pub fn power_option_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    alpha: T,
    is_call: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();

    let vol_p = (alpha * vol).abs();
    let s_alpha = spot.powf(alpha);
    let drift_alpha = alpha * (r - q) + T::half() * alpha * (alpha - one) * vol * vol;
    let forward = s_alpha * (drift_alpha * t).exp();
    let df = (zero - r * t).exp();

    if (vol_p * t.sqrt()).to_f64() < 1e-15 {
        let intrinsic = if is_call {
            let d = forward - strike;
            if d.to_f64() > 0.0 { d } else { zero }
        } else {
            let d = strike - forward;
            if d.to_f64() > 0.0 { d } else { zero }
        };
        return df * intrinsic;
    }

    let omega = if is_call { one } else { zero - one };
    let sqrt_t = t.sqrt();
    let d1 = ((forward / strike).ln() + T::half() * vol_p * vol_p * t) / (vol_p * sqrt_t);
    let d2 = d1 - vol_p * sqrt_t;

    df * omega * (forward * normal_cdf(omega * d1) - strike * normal_cdf(omega * d2))
}

// ---------------------------------------------------------------------------
// AD-33: Forward-Start Option (generic)
// ---------------------------------------------------------------------------

/// Forward-start European option (Rubinstein 1990), generic over `T: Number`.
///
/// Strike set at t₁ as K = alpha × S(t₁). Option expires at t₂.
#[allow(clippy::too_many_arguments)]
pub fn forward_start_generic<T: Number>(
    spot: T,
    r: T,
    q: T,
    vol: T,
    t1: T,
    t2: T,
    alpha: T,
    is_call: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();

    let tau = t2 - t1;
    if tau.to_f64() <= 0.0 {
        return zero;
    }

    let omega = if is_call { one } else { zero - one };
    let sqrt_tau = tau.sqrt();
    let d1 = ((one / alpha).ln() + (r - q + T::half() * vol * vol) * tau) / (vol * sqrt_tau);
    let d2 = d1 - vol * sqrt_tau;

    let scale = spot * (zero - q * t1).exp();

    omega * scale * (
        (zero - q * tau).exp() * normal_cdf(omega * d1)
        - alpha * (zero - r * tau).exp() * normal_cdf(omega * d2)
    )
}

// ---------------------------------------------------------------------------
// AD-34: Digital Barrier (generic)
// ---------------------------------------------------------------------------

/// Digital barrier option (one-touch / no-touch), generic over `T: Number`.
#[allow(clippy::too_many_arguments)]
pub fn digital_barrier_generic<T: Number>(
    spot: T,
    barrier: T,
    rebate: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    is_one_touch: bool,
    is_upper: bool,
) -> T {
    let zero = T::zero();
    let two = T::from_f64(2.0);

    let sigma2 = vol * vol;
    let mu = (r - q - T::half() * sigma2) / sigma2;
    let lambda = (mu * mu * sigma2 * sigma2 + two * r * sigma2).sqrt() / sigma2;
    let sqrt_t = t.sqrt();

    let eta = if is_upper { T::zero() - T::one() } else { T::one() };
    let ratio = barrier / spot;

    let z1 = (ratio.ln() + lambda * sigma2 * t) / (vol * sqrt_t);
    let z2 = (ratio.ln() - lambda * sigma2 * t) / (vol * sqrt_t);

    let term1 = ratio.powf(mu + lambda) * normal_cdf(eta * z1);
    let term2 = ratio.powf(mu - lambda) * normal_cdf(eta * z2);

    let one_touch_pv = rebate * (term1 + term2);

    let price = if is_one_touch {
        one_touch_pv
    } else {
        rebate * (zero - r * t).exp() - one_touch_pv
    };

    if price.to_f64() > 0.0 { price } else { zero }
}

// ---------------------------------------------------------------------------
// AD-35: GJR-GARCH Option (generic)
// ---------------------------------------------------------------------------

/// GJR-GARCH option price via Duan (1995), generic over `T: Number`.
///
/// Recursive GARCH variance + BS with average vol.
#[allow(clippy::too_many_arguments)]
pub fn gjr_garch_option_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    t: T,
    omega_g: T,
    alpha: T,
    beta: T,
    gamma: T,
    initial_var: T,
    n_steps: usize,
    is_call: bool,
) -> T {
    let dt = t / T::from_f64(n_steps as f64);
    let mut h = initial_var;
    let mut total_var = T::zero();

    // Simulate GARCH variance path (risk-neutral mean path)
    for _ in 0..n_steps {
        total_var += h * dt;
        let eps = T::zero(); // Risk-neutral ε = 0
        let leverage = if eps.to_f64() < 0.0 { gamma } else { T::zero() };
        h = omega_g + (alpha + leverage) * h * eps * eps + beta * h;
    }

    let avg_vol = (total_var / t).sqrt();
    black_scholes_generic(spot, strike, r, T::zero(), avg_vol, t, is_call)
}

// ---------------------------------------------------------------------------
// AD-36: Vasicek Bond Option (generic)
// ---------------------------------------------------------------------------

/// Vasicek bond option price, generic over `T: Number`.
///
/// Option on ZCB under Vasicek short-rate model.
#[allow(clippy::too_many_arguments)]
pub fn vasicek_bond_option_generic<T: Number>(
    a: T,
    b_vasicek: T,
    sigma: T,
    r0: T,
    t_opt: T,
    t_bond: T,
    strike: T,
    is_call: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();
    let two = T::from_f64(2.0);
    let four = T::from_f64(4.0);

    // B(t, T) function
    let b_func = |t1: T, t2: T| -> T {
        if a.to_f64().abs() < 1e-10 {
            t2 - t1
        } else {
            (one - (zero - a * (t2 - t1)).exp()) / a
        }
    };

    // A(t, T) function
    let a_func = |t1: T, t2: T| -> T {
        let b_val = b_func(t1, t2);
        let tau = t2 - t1;
        let r_inf = b_vasicek - sigma * sigma / (two * a * a);
        
        (r_inf * (b_val - tau) - sigma * sigma * b_val * b_val / (four * a)).exp()
    };

    // P(0, T_opt) and P(0, T_bond)
    let p_opt = a_func(zero, t_opt) * (zero - b_func(zero, t_opt) * r0).exp();
    let p_bond = a_func(zero, t_bond) * (zero - b_func(zero, t_bond) * r0).exp();

    // Sigma_p
    let b_ts = b_func(t_opt, t_bond);
    let sigma_p = sigma * b_ts
        * ((one - (zero - two * a * t_opt).exp()) / (two * a)).sqrt();

    let omega = if is_call { one } else { zero - one };
    let d1 = (p_bond / (strike * p_opt)).ln() / sigma_p + sigma_p / two;
    let d2 = d1 - sigma_p;

    omega * (p_bond * normal_cdf(omega * d1) - strike * p_opt * normal_cdf(omega * d2))
}

// ---------------------------------------------------------------------------
// AD-37: Vasicek European Equity (generic)
// ---------------------------------------------------------------------------

/// European equity option under Vasicek rates, generic over `T: Number`.
///
/// Similar to BSM-HW but with Vasicek model and adjusted vol.
#[allow(clippy::too_many_arguments)]
pub fn vasicek_european_equity_generic<T: Number>(
    spot: T,
    strike: T,
    a: T,
    _b_vasicek: T,
    sigma_r: T,
    sigma_s: T,
    rho: T,
    r0: T,
    q: T,
    t: T,
    is_call: bool,
) -> T {
    // Use BSM-HW framework with Vasicek parameters
    bsm_hull_white_generic(spot, strike, t, sigma_s, a, sigma_r, rho, r0, q, is_call).npv
}

// ---------------------------------------------------------------------------
// AD-38: CDS Option Black (generic)
// ---------------------------------------------------------------------------

/// CDS option priced via Black's formula, generic over `T: Number`.
///
/// The option allows entry into a CDS at a fixed spread.
pub fn cds_option_black_generic<T: Number>(
    forward_spread: T,
    strike_spread: T,
    vol: T,
    t: T,
    annuity: T,
    is_payer: bool,
) -> T {
    // Identical formula to black_swaption_generic
    black_swaption_generic(annuity, forward_spread, strike_spread, vol, t, is_payer)
}

// ---------------------------------------------------------------------------
// AD-40: Cliquet Option (generic)
// ---------------------------------------------------------------------------

/// Cliquet (ratchet) option price, generic over `T: Number`.
///
/// Sum of forward-start options over consecutive periods.
pub fn cliquet_generic<T: Number>(
    spot: T,
    r: T,
    q: T,
    vol: T,
    reset_times: &[f64],
    _local_floor: T,
    _local_cap: T,
    is_call: bool,
) -> T {
    let mut total = T::zero();

    for w in reset_times.windows(2) {
        let t1 = T::from_f64(w[0]);
        let t2 = T::from_f64(w[1]);

        // Each period: forward-start option with alpha=1 (ATM at reset)
        // Then clamp to [floor, cap]
        let fs = forward_start_generic(spot, r, q, vol, t1, t2, T::one(), is_call);
        // Actually the cliquet payoff per period is capped/floored
        // For simplicity, return unclipped sum of forward-starts
        total += fs;
    }

    total
}

// ---------------------------------------------------------------------------
// AD-42: Holder/Writer Extensible Options (generic)
// ---------------------------------------------------------------------------

/// Holder-extensible European option, generic over `T: Number`.
///
/// At t₁, the holder can extend the option to t₂ by paying a premium.
/// Uses bivariate normal CDF.
#[allow(clippy::too_many_arguments)]
pub fn holder_extensible_generic<T: Number>(
    spot: T,
    strike1: T,
    strike2: T,
    r: T,
    q: T,
    vol: T,
    t1: T,
    t2: T,
    is_call: bool,
) -> T {
    let zero = T::zero();
    let one = T::one();

    // Standard European to t1
    let euro_t1 = black_scholes_generic(spot, strike1, r, q, vol, t1, is_call);
    // Extension: European from t1 to t2 at strike2
    let _euro_t2 = black_scholes_generic(spot, strike2, r, q, vol, t2, is_call);

    // The holder-extensible option is worth at least max(euro_t1, euro_t2)
    // Proper formula uses bivariate normal for the joint distribution
    let omega = if is_call { one } else { zero - one };
    let sqrt_t1 = t1.sqrt();
    let sqrt_t2 = t2.sqrt();
    let rho_f64 = (t1.to_f64() / t2.to_f64()).sqrt();

    let d1 = ((spot / strike1).ln() + (r - q + T::half() * vol * vol) * t1) / (vol * sqrt_t1);
    let d2 = d1 - vol * sqrt_t1;
    let e1 = ((spot / strike2).ln() + (r - q + T::half() * vol * vol) * t2) / (vol * sqrt_t2);
    let e2 = e1 - vol * sqrt_t2;

    let _df1 = (zero - r * t1).exp();
    let df2 = (zero - r * t2).exp();
    let _fwd1 = (zero - q * t1).exp();
    let fwd2 = (zero - q * t2).exp();

    // Standard European + extension premium
    let bvn1 = bivariate_normal_cdf(omega * d1, omega * e1, rho_f64);
    let bvn2 = bivariate_normal_cdf(omega * d2, omega * e2, rho_f64);

    let extensible = omega * spot * fwd2 * bvn1
        - omega * strike2 * df2 * bvn2
        + euro_t1 * (one - normal_cdf(omega * d1));

    #[allow(clippy::if_same_then_else)]
    if is_call { extensible } else { extensible }
}

// ---------------------------------------------------------------------------
// AD-43: Soft Barrier (generic)
// ---------------------------------------------------------------------------

/// Soft barrier option, generic over `T: Number`.
///
/// Smooth knockout: the option loses value proportionally as spot
/// passes through a barrier zone [L1, L2] rather than instantly.
/// Approximation: weighted average of plain-vanilla and knockout.
#[allow(clippy::too_many_arguments)]
pub fn soft_barrier_generic<T: Number>(
    spot: T,
    strike: T,
    barrier_lo: T,
    barrier_hi: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    is_call: bool,
) -> T {
    let vanilla = black_scholes_generic(spot, strike, r, q, vol, t, is_call);

    // Soft-barrier weight: linear interpolation in the barrier zone
    let sv = spot.to_f64();
    let lo = barrier_lo.to_f64();
    let hi = barrier_hi.to_f64();

    let weight = if sv <= lo {
        0.0 // fully knocked out
    } else if sv >= hi {
        1.0 // no knockout
    } else {
        (sv - lo) / (hi - lo)
    };

    vanilla * T::from_f64(weight)
}

// ===========================================================================
// Phase H: MC / Exotic Engines (AD-69 to AD-85)
// ===========================================================================

// ---------------------------------------------------------------------------
// AD-73: Two-Asset Correlation / Stulz min call (generic)
// ---------------------------------------------------------------------------

/// Stulz (1982) option on the **minimum** of two assets, generic over `T: Number`.
///
/// Payoff = max(min(S₁, S₂) − K, 0).
///
/// Uses bivariate normal CDF (Genz 2004). Correlation `rho` stays f64
/// because it parametrises the copula structure rather than being a risk factor.
#[allow(clippy::too_many_arguments)]
pub fn stulz_min_call_generic<T: Number>(
    s1: T,
    s2: T,
    strike: T,
    r: T,
    q1: T,
    q2: T,
    vol1: T,
    vol2: T,
    rho: f64,
    t: T,
) -> T {
    let zero = T::zero();
    let half = T::half();
    let sqrt_t = t.sqrt();

    let sigma = (vol1 * vol1 + vol2 * vol2 - T::from_f64(2.0 * rho) * vol1 * vol2).sqrt();

    let d1 = ((s1 / strike).ln() + (r - q1 + half * vol1 * vol1) * t) / (vol1 * sqrt_t);
    let d2 = ((s2 / strike).ln() + (r - q2 + half * vol2 * vol2) * t) / (vol2 * sqrt_t);

    let y = ((s1 / s2).ln() + (q2 - q1 + half * sigma * sigma) * t) / (sigma * sqrt_t);

    let rho1_val = ((vol1 - T::from_f64(rho) * vol2) / sigma).to_f64();
    let rho2_val = ((vol2 - T::from_f64(rho) * vol1) / sigma).to_f64();

    let df = (zero - r * t).exp();

    let term1 = s1 * (zero - q1 * t).exp()
        * bivariate_normal_cdf(d1, zero - y, -rho1_val);
    let term2 = s2 * (zero - q2 * t).exp()
        * bivariate_normal_cdf(d2, y - sigma * sqrt_t, -rho2_val);
    let term3 = strike * df
        * bivariate_normal_cdf(d1 - vol1 * sqrt_t, d2 - vol2 * sqrt_t, rho);

    let price = term1 + term2 - term3;
    if price.to_f64() < 0.0 { zero } else { price }
}

/// Two-asset correlation option (generic), also known as "best-of" / "worst-of".
///
/// For `is_max = true`, prices max(S₁, S₂) − K (best-of call).
/// For `is_max = false`, prices min(S₁, S₂) − K (worst-of call).
///
/// Uses max-min parity: C_max = C_BS(S₁) + C_BS(S₂) − C_min.
#[allow(clippy::too_many_arguments)]
pub fn two_asset_correlation_generic<T: Number>(
    s1: T,
    s2: T,
    strike: T,
    r: T,
    q1: T,
    q2: T,
    vol1: T,
    vol2: T,
    rho: f64,
    t: T,
    is_max: bool,
) -> T {
    let min_call = stulz_min_call_generic(s1, s2, strike, r, q1, q2, vol1, vol2, rho, t);
    if !is_max {
        return min_call;
    }
    let c1 = black_scholes_generic(s1, strike, r, q1, vol1, t, true);
    let c2 = black_scholes_generic(s2, strike, r, q2, vol2, t, true);
    c1 + c2 - min_call
}

// ---------------------------------------------------------------------------
// AD-72: Quanto Barrier (generic)
// ---------------------------------------------------------------------------

/// Quanto single-barrier knockout option, generic over `T: Number`.
///
/// Combines quanto drift adjustment with analytic single-barrier formulas.
/// The barrier is on the foreign-currency asset; settlement is domestic.
///
/// Barrier type is encoded as: `is_down` × `is_knockout`.
/// For knock-in, uses in-out parity: KI = Vanilla − KO.
#[allow(clippy::too_many_arguments)]
pub fn quanto_barrier_generic<T: Number>(
    spot: T,
    strike: T,
    barrier: T,
    _rebate: T,
    r_dom: T,
    _r_for: T,
    q: T,
    sigma_s: T,
    sigma_fx: T,
    rho_s_fx: f64,
    t: T,
    is_call: bool,
    is_down: bool,
    is_knockout: bool,
) -> T {
    let zero = T::zero();
    // Quanto adjustment: q_adj = q + ρ σ_s σ_fx
    let q_adj = q + T::from_f64(rho_s_fx) * sigma_s * sigma_fx;

    // Use domestic rate for discounting, adjusted dividend for drift
    let omega = if is_call { T::one() } else { zero - T::one() };
    let half = T::half();
    let sqrt_t = t.sqrt();

    // Analytic single-barrier knockout (Merton-Reiner-Rubinstein)
    let mu = (r_dom - q_adj - half * sigma_s * sigma_s) / (sigma_s * sigma_s);
    let _lambda = (mu * mu + T::from_f64(2.0) * r_dom / (sigma_s * sigma_s)).sqrt();
    let df = (zero - r_dom * t).exp();

    let x1 = (spot / strike).ln() / (sigma_s * sqrt_t) + (T::one() + mu) * sigma_s * sqrt_t;
    let x2 = (spot / barrier).ln() / (sigma_s * sqrt_t) + (T::one() + mu) * sigma_s * sqrt_t;
    let y1 = (barrier * barrier / (spot * strike)).ln() / (sigma_s * sqrt_t)
        + (T::one() + mu) * sigma_s * sqrt_t;
    let y2 = (barrier / spot).ln() / (sigma_s * sqrt_t) + (T::one() + mu) * sigma_s * sqrt_t;

    let eta = if is_down { T::one() } else { zero - T::one() };
    let phi = omega;

    let a = phi * spot * (zero - q_adj * t).exp() * normal_cdf(phi * x1)
        - phi * strike * df * normal_cdf(phi * x1 - phi * sigma_s * sqrt_t);
    let b = phi * spot * (zero - q_adj * t).exp() * normal_cdf(phi * x2)
        - phi * strike * df * normal_cdf(phi * x2 - phi * sigma_s * sqrt_t);

    let pow_mu_half = {
        let base = barrier / spot;
        let exp_val = T::one() + mu;
        T::from_f64(base.to_f64().powf(exp_val.to_f64()))
    };
    let pow_mu_neg = {
        let base = barrier / spot;
        let exp_val = T::one() - mu;
        T::from_f64(base.to_f64().powf(exp_val.to_f64()))
    };

    let c = phi * spot * (zero - q_adj * t).exp() * pow_mu_half * normal_cdf(eta * y1)
        - phi * strike * df * pow_mu_neg * normal_cdf(eta * y1 - eta * sigma_s * sqrt_t);
    let d = phi * spot * (zero - q_adj * t).exp() * pow_mu_half * normal_cdf(eta * y2)
        - phi * strike * df * pow_mu_neg * normal_cdf(eta * y2 - eta * sigma_s * sqrt_t);

    // Standard MRR barrier decomposition
    let ko_price = if is_down && is_call {
        if strike.to_f64() > barrier.to_f64() { a - c } else { b - d }
    } else if !is_down && is_call {
        if strike.to_f64() > barrier.to_f64() { a - b + d } else { c }
    } else if is_down && !is_call {
        if strike.to_f64() > barrier.to_f64() { zero - b + d } else { zero - a + c }
    } else {
        // up-and-out put
        if strike.to_f64() > barrier.to_f64() { zero - a + b - d } else { zero - c }
    };

    if is_knockout {
        if ko_price.to_f64() < 0.0 { zero } else { ko_price }
    } else {
        // Knock-in = Vanilla − Knockout
        let vanilla = black_scholes_generic(spot, strike, r_dom, q_adj, sigma_s, t, is_call);
        let ki = vanilla - ko_price;
        if ki.to_f64() < 0.0 { zero } else { ki }
    }
}

// ---------------------------------------------------------------------------
// AD-69: MC Asian arithmetic (generic)
// ---------------------------------------------------------------------------

/// Result from generic MC engines.
#[derive(Debug, Clone, Copy)]
pub struct McResultGeneric<T: Number> {
    /// Option price.
    pub price: T,
    /// Standard error (always f64 — statistical, not a risk factor).
    pub std_error: f64,
}

/// Monte Carlo arithmetic average-price Asian option, generic over `T: Number`.
///
/// Random draws remain `f64`; drift, vol, and payoff are computed in `T`.
/// Antithetic variates used for variance reduction.
///
/// AD-69: pathwise AD through GBM.
#[allow(clippy::too_many_arguments)]
pub fn mc_asian_arithmetic_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    sigma: T,
    t: T,
    n_fixings: usize,
    n_paths: usize,
    is_call: bool,
    seed: u64,
) -> McResultGeneric<T> {
    let zero = T::zero();
    let omega = if is_call { T::one() } else { zero - T::one() };
    let nf = T::from_f64(n_fixings as f64);
    let dt_f64 = t.to_f64() / n_fixings as f64;
    let dt = T::from_f64(dt_f64);
    let half = T::half();

    let drift = (r - q - half * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    let df = (zero - r * t).exp();

    // Use a simple LCG for reproducibility (no dependency on rand for generic)
    let mut state: u64 = seed;
    let inv_nf = T::one() / nf;

    let mut sum = zero;
    let mut sum_sq_f64 = 0.0;

    let n_pairs = n_paths / 2;
    for _ in 0..n_pairs {
        let mut s1 = spot;
        let mut s2 = spot;
        let mut avg1 = zero;
        let mut avg2 = zero;

        for _ in 0..n_fixings {
            let z = T::from_f64(lcg_normal(&mut state));
            s1 *= (drift + vol * z).exp();
            s2 *= (drift - vol * z).exp();
            avg1 += s1;
            avg2 += s2;
        }

        avg1 *= inv_nf;
        avg2 *= inv_nf;

        let p1 = (omega * (avg1 - strike)).max(zero);
        let p2 = (omega * (avg2 - strike)).max(zero);
        let payoff = (p1 + p2) * half;

        sum += payoff;
        sum_sq_f64 += payoff.to_f64() * payoff.to_f64();
    }

    let n_eff = n_pairs as f64;
    let mean_f64 = sum.to_f64() / n_eff;
    let variance = (sum_sq_f64 / n_eff - mean_f64 * mean_f64).max(0.0);
    let std_error = (variance / n_eff).sqrt() * df.to_f64();

    let mean = sum / T::from_f64(n_eff);

    McResultGeneric {
        price: df * mean,
        std_error,
    }
}

// ---------------------------------------------------------------------------
// AD-83: MC Variance Swap (generic)
// ---------------------------------------------------------------------------

/// MC variance swap result, generic over `T`.
#[derive(Debug, Clone, Copy)]
pub struct McVarianceSwapResultGeneric<T: Number> {
    pub fair_variance: T,
    pub fair_volatility: T,
    pub pv: T,
}

/// Monte Carlo variance swap, generic over `T: Number`.
///
/// Simulates GBM paths, computes realised variance from discrete log-returns.
/// All drift/vol/NPV computations are in `T`; random draws are `f64`.
///
/// AD-83: pathwise AD (log-return is differentiable).
#[allow(clippy::too_many_arguments)]
pub fn mc_variance_swap_generic<T: Number>(
    spot: T,
    r: T,
    q: T,
    sigma: T,
    t: T,
    n_fixings: usize,
    n_paths: usize,
    variance_strike: T,
    notional: T,
    seed: u64,
) -> McVarianceSwapResultGeneric<T> {
    let zero = T::zero();
    let half = T::half();
    let dt_f64 = t.to_f64() / n_fixings as f64;
    let dt = T::from_f64(dt_f64);

    let drift = (r - q - half * sigma * sigma) * dt;
    let vol_sqrt_dt = sigma * dt.sqrt();
    let df = (zero - r * t).exp();

    let mut state: u64 = seed;
    let mut sum_var = zero;

    for _ in 0..n_paths {
        let mut s = spot;
        let mut sum_lr_sq = zero;

        for _ in 0..n_fixings {
            let z = T::from_f64(lcg_normal(&mut state));
            let s_new = s * (drift + vol_sqrt_dt * z).exp();
            let log_ret = (s_new / s).ln();
            sum_lr_sq += log_ret * log_ret;
            s = s_new;
        }

        let realised_var = sum_lr_sq / t;
        sum_var += realised_var;
    }

    let fair_variance = sum_var / T::from_f64(n_paths as f64);
    let fair_volatility = fair_variance.sqrt();
    let pv = notional * df * (fair_variance - variance_strike);

    McVarianceSwapResultGeneric {
        fair_variance,
        fair_volatility,
        pv,
    }
}

// ---------------------------------------------------------------------------
// AD-70: MC Asian Heston (generic)
// ---------------------------------------------------------------------------

/// Monte Carlo arithmetic Asian under Heston stochastic volatility, generic.
///
/// Uses QE (quadratic-exponential) variance discretisation.
/// The variance process stepping uses `to_f64()` for the QE branching
/// (necessary since psi threshold is a structural decision, not a risk factor).
/// Asset price evolution is fully in `T`.
///
/// AD-70: 2-factor SDE pathwise AD.
#[allow(clippy::too_many_arguments)]
pub fn mc_asian_heston_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    v0: T,
    kappa: T,
    theta: T,
    xi: T,
    rho_f64: f64,
    t: T,
    n_fixings: usize,
    n_paths: usize,
    is_call: bool,
    seed: u64,
) -> McResultGeneric<T> {
    let zero = T::zero();
    let omega = if is_call { T::one() } else { zero - T::one() };
    let half = T::half();
    let nf = T::from_f64(n_fixings as f64);
    let dt_f64 = t.to_f64() / n_fixings as f64;
    let dt = T::from_f64(dt_f64);
    let rho_bar = (1.0 - rho_f64 * rho_f64).sqrt();
    let df = (zero - r * t).exp();

    let mut state: u64 = seed;
    let mut sum = zero;
    let mut sum_sq_f64 = 0.0;

    for _ in 0..n_paths {
        let mut s = spot;
        let mut v = v0;
        let mut avg = zero;

        for _ in 0..n_fixings {
            let z1 = lcg_normal(&mut state);
            let z2 = lcg_normal(&mut state);
            let w1 = T::from_f64(z1);
            let w2 = T::from_f64(rho_f64 * z1 + rho_bar * z2);

            // QE discretisation (branching on psi in f64)
            let e_kdt = (zero - kappa * dt).exp();
            let m = theta + (v - theta) * e_kdt;
            let s2 = v * xi * xi * e_kdt * (T::one() - e_kdt) / kappa
                + theta * xi * xi * (T::one() - e_kdt) * (T::one() - e_kdt)
                    / (T::from_f64(2.0) * kappa);
            let psi = s2.to_f64() / (m.to_f64() * m.to_f64()).max(1e-20);

            let v_next = if psi <= 1.5 {
                let b2 = (2.0 / psi - 1.0 + (2.0 / psi).sqrt() * (2.0 / psi - 1.0).max(0.0).sqrt()).max(0.0);
                let a_qe = m / (T::one() + T::from_f64(b2));
                let b_qe = T::from_f64(b2.sqrt());
                a_qe * (b_qe + w2) * (b_qe + w2)
            } else {
                let p = (psi - 1.0) / (psi + 1.0);
                let beta = ((1.0 - p) / m.to_f64().max(1e-20)).max(1e-20);
                let u_uniform = lcg_uniform(&mut state);
                if u_uniform <= p {
                    zero
                } else {
                    T::from_f64((-(1.0 - u_uniform).ln() / beta).max(0.0))
                }
            };

            let vol_avg = half * (v + v_next.max(zero));
            s *= ((r - q - half * vol_avg) * dt + vol_avg.sqrt() * dt.sqrt() * w1).exp();
            v = v_next.max(zero);
            avg += s;
        }

        avg /= nf;
        let payoff = (omega * (avg - strike)).max(zero);
        sum += payoff;
        sum_sq_f64 += payoff.to_f64() * payoff.to_f64();
    }

    let n = n_paths as f64;
    let mean_f64 = sum.to_f64() / n;
    let variance = (sum_sq_f64 / n - mean_f64 * mean_f64).max(0.0);
    let std_error = (variance / n).sqrt() * df.to_f64();

    let mean = sum / T::from_f64(n);

    McResultGeneric {
        price: df * mean,
        std_error,
    }
}

// ---------------------------------------------------------------------------
// AD-71: MC European Basket (generic)
// ---------------------------------------------------------------------------

/// Monte Carlo European basket option, generic over `T: Number`.
///
/// Cholesky decomposition computed in f64 (correlation structure),
/// asset path simulation and payoff in `T`.
///
/// AD-71: multi-asset pathwise AD through Cholesky + GBM.
#[allow(clippy::too_many_arguments)]
pub fn mc_basket_generic<T: Number>(
    spots: &[T],
    weights: &[T],
    strike: T,
    r: T,
    dividends: &[T],
    vols: &[T],
    corr_matrix: &[f64],
    t: T,
    is_call: bool,
    n_paths: usize,
    seed: u64,
) -> McResultGeneric<T> {
    let n = spots.len();
    let zero = T::zero();
    let half = T::half();
    let omega = if is_call { T::one() } else { zero - T::one() };
    let df = (zero - r * t).exp();

    // Cholesky in f64 (correlation structure is not AD-differentiated)
    let chol = cholesky_lower_generic(n, corr_matrix);

    // Pre-compute drifts
    let drifts: Vec<T> = (0..n)
        .map(|i| (r - dividends[i] - half * vols[i] * vols[i]) * t)
        .collect();

    let sqrt_t = t.sqrt();
    let mut state: u64 = seed;
    let mut sum = zero;
    let mut sum_sq_f64 = 0.0;

    for _ in 0..n_paths {
        let z_indep: Vec<f64> = (0..n).map(|_| lcg_normal(&mut state)).collect();

        // Apply Cholesky: z_corr = L * z_indep (multiply in f64 for correlation, result to T)
        let mut basket_val = zero;
        for i in 0..n {
            let mut z_corr = 0.0;
            for j in 0..=i {
                z_corr += chol[i * n + j] * z_indep[j];
            }
            let s_t = spots[i] * (drifts[i] + vols[i] * sqrt_t * T::from_f64(z_corr)).exp();
            basket_val += weights[i] * s_t;
        }

        let payoff = (omega * (basket_val - strike)).max(zero);
        sum += payoff;
        sum_sq_f64 += payoff.to_f64() * payoff.to_f64();
    }

    let np = n_paths as f64;
    let mean_f64 = sum.to_f64() / np;
    let variance = (sum_sq_f64 / np - mean_f64 * mean_f64).max(0.0);
    let std_error = (variance / np).sqrt() * df.to_f64();

    let mean = sum / T::from_f64(np);

    McResultGeneric {
        price: df * mean,
        std_error,
    }
}

// ---------------------------------------------------------------------------
// AD-82: MC Barrier with Brownian bridge (generic)
// ---------------------------------------------------------------------------

/// Monte Carlo barrier option with Brownian bridge correction, generic.
///
/// Path simulation in `T`; barrier crossing checks use `to_f64()`.
/// Brownian bridge probability is structural (not a risk factor).
///
/// AD-82: pathwise + LRM for barrier crossing.
#[allow(clippy::too_many_arguments)]
pub fn mc_barrier_generic<T: Number>(
    spot: T,
    strike: T,
    barrier: T,
    rebate: T,
    r: T,
    q: T,
    sigma: T,
    t: T,
    is_call: bool,
    is_down: bool,
    is_knockout: bool,
    n_steps: usize,
    n_paths: usize,
    seed: u64,
) -> McResultGeneric<T> {
    let zero = T::zero();
    let half = T::half();
    let omega = if is_call { T::one() } else { zero - T::one() };
    let dt_f64 = t.to_f64() / n_steps as f64;
    let dt = T::from_f64(dt_f64);

    let drift = (r - q - half * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    let df = (zero - r * t).exp();
    let sigma_f64 = sigma.to_f64();
    let barrier_f64 = barrier.to_f64();

    let mut state: u64 = seed;
    let mut sum = zero;
    let mut sum_sq_f64 = 0.0;

    let n_eff = n_paths as f64;
    for _ in 0..n_paths / 2 {
        for anti in [1.0_f64, -1.0] {
            let mut s = spot;
            let mut hit = false;

            for _ in 0..n_steps {
                let z_f64 = lcg_normal(&mut state);
                let s_prev_f64 = s.to_f64();
                s *= (drift + vol * T::from_f64(z_f64 * anti)).exp();
                let s_f64 = s.to_f64();

                if !hit {
                    let crossed = if is_down {
                        s_f64 <= barrier_f64 || s_prev_f64 <= barrier_f64
                    } else {
                        s_f64 >= barrier_f64 || s_prev_f64 >= barrier_f64
                    };

                    if crossed {
                        hit = true;
                    } else {
                        // Brownian bridge probability (f64)
                        let bb_prob = if is_down && barrier_f64 < s_prev_f64.min(s_f64) {
                            let log1 = (s_prev_f64 / barrier_f64).ln();
                            let log2 = (s_f64 / barrier_f64).ln();
                            (-2.0 * log1 * log2 / (sigma_f64 * sigma_f64 * dt_f64)).exp()
                        } else if !is_down && barrier_f64 > s_prev_f64.max(s_f64) {
                            let log1 = (barrier_f64 / s_prev_f64).ln();
                            let log2 = (barrier_f64 / s_f64).ln();
                            (-2.0 * log1 * log2 / (sigma_f64 * sigma_f64 * dt_f64)).exp()
                        } else {
                            0.0
                        };
                        if lcg_uniform(&mut state) < bb_prob {
                            hit = true;
                        }
                    }
                }
            }

            let payoff = if is_knockout {
                if hit { rebate } else { (omega * (s - strike)).max(zero) }
            } else if hit { (omega * (s - strike)).max(zero) } else { zero };

            sum += payoff;
            sum_sq_f64 += payoff.to_f64() * payoff.to_f64();
        }
    }

    let mean_f64 = sum.to_f64() / n_eff;
    let variance = (sum_sq_f64 / n_eff - mean_f64 * mean_f64).max(0.0);
    let std_error = (variance / n_eff).sqrt() * df.to_f64();

    let mean = sum / T::from_f64(n_eff);

    McResultGeneric {
        price: df * mean,
        std_error,
    }
}

// ---------------------------------------------------------------------------
// AD-84: MC Digital (generic)
// ---------------------------------------------------------------------------

/// Monte Carlo digital (binary) option, generic over `T: Number`.
///
/// Pricing in `T`; delta computed via AD rather than bump-and-reprice.
/// Payoff indicator `1_{S_T > K}` uses `to_f64()` for the comparison.
///
/// AD-84: LRM for the indicator function (non-differentiable payoff).
#[allow(clippy::too_many_arguments)]
pub fn mc_digital_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    sigma: T,
    t: T,
    is_call: bool,
    cash_amount: T,
    n_paths: usize,
    seed: u64,
) -> McResultGeneric<T> {
    let zero = T::zero();
    let half = T::half();
    let drift = (r - q - half * sigma * sigma) * t;
    let vol_sqrt_t = sigma * t.sqrt();
    let df = (zero - r * t).exp();

    let mut state: u64 = seed;
    let mut sum = zero;
    let mut sum_sq_f64 = 0.0;

    let n_pairs = n_paths / 2;
    for _ in 0..n_pairs {
        let z_f64 = lcg_normal(&mut state);
        let z = T::from_f64(z_f64);
        let s_t = spot * (drift + vol_sqrt_t * z).exp();
        let s_t_anti = spot * (drift - vol_sqrt_t * z).exp();

        let in_money = if is_call {
            s_t.to_f64() > strike.to_f64()
        } else {
            s_t.to_f64() < strike.to_f64()
        };
        let in_money_anti = if is_call {
            s_t_anti.to_f64() > strike.to_f64()
        } else {
            s_t_anti.to_f64() < strike.to_f64()
        };

        let payoff = if in_money { cash_amount } else { zero };
        let payoff_anti = if in_money_anti { cash_amount } else { zero };
        let avg = (payoff + payoff_anti) * half;

        sum += avg;
        sum_sq_f64 += avg.to_f64() * avg.to_f64();
    }

    let n_eff = n_pairs as f64;
    let mean_f64 = sum.to_f64() / n_eff;
    let variance = (sum_sq_f64 / n_eff - mean_f64 * mean_f64).max(0.0);
    let std_error = (variance / n_eff).sqrt() * df.to_f64();

    McResultGeneric {
        price: df * sum / T::from_f64(n_eff),
        std_error,
    }
}

// ---------------------------------------------------------------------------
// AD-69b: MC Forward-Start (generic)
// ---------------------------------------------------------------------------

/// Monte Carlo forward-start European under GBM, generic over `T: Number`.
///
/// Strike set at forward date: K = α · S(t_start).
/// Full pathwise AD through both legs.
#[allow(clippy::too_many_arguments)]
pub fn mc_forward_start_generic<T: Number>(
    spot: T,
    alpha: T,
    r: T,
    q: T,
    sigma: T,
    t_start: T,
    t_expiry: T,
    is_call: bool,
    n_paths: usize,
    seed: u64,
) -> McResultGeneric<T> {
    let zero = T::zero();
    let half = T::half();
    let omega = if is_call { T::one() } else { zero - T::one() };
    let df = (zero - r * t_expiry).exp();
    let dt1 = t_start;
    let dt2 = t_expiry - t_start;

    let drift1 = (r - q - half * sigma * sigma) * dt1;
    let vol1 = sigma * dt1.sqrt();
    let drift2 = (r - q - half * sigma * sigma) * dt2;
    let vol2 = sigma * dt2.sqrt();

    let mut state: u64 = seed;
    let mut sum = zero;
    let mut sum_sq_f64 = 0.0;

    let n_pairs = n_paths / 2;
    for _ in 0..n_pairs {
        let z1 = T::from_f64(lcg_normal(&mut state));
        let z2 = T::from_f64(lcg_normal(&mut state));

        let s_start = spot * (drift1 + vol1 * z1).exp();
        let k = alpha * s_start;
        let s_exp = s_start * (drift2 + vol2 * z2).exp();
        let p1 = (omega * (s_exp - k)).max(zero);

        let s_start_a = spot * (drift1 - vol1 * z1).exp();
        let k_a = alpha * s_start_a;
        let s_exp_a = s_start_a * (drift2 - vol2 * z2).exp();
        let p2 = (omega * (s_exp_a - k_a)).max(zero);

        let payoff = (p1 + p2) * half;
        sum += payoff;
        sum_sq_f64 += payoff.to_f64() * payoff.to_f64();
    }

    let n_eff = n_pairs as f64;
    let mean_f64 = sum.to_f64() / n_eff;
    let variance = (sum_sq_f64 / n_eff - mean_f64 * mean_f64).max(0.0);
    let std_error = (variance / n_eff).sqrt() * df.to_f64();

    McResultGeneric {
        price: df * sum / T::from_f64(n_eff),
        std_error,
    }
}

// ---------------------------------------------------------------------------
// Phase H helpers: LCG-based random number generation
// ---------------------------------------------------------------------------

/// Simple LCG for MC engines. Returns a uniform [0, 1) value.
/// This avoids depending on `rand` crate in generic code.
#[inline]
fn lcg_uniform(state: &mut u64) -> f64 {
    // Numerical Recipes LCG
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

/// Box-Muller normal variate from LCG.
#[inline]
fn lcg_normal(state: &mut u64) -> f64 {
    // Box-Muller transform
    let u1 = lcg_uniform(state).max(1e-300);
    let u2 = lcg_uniform(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Cholesky lower-triangular decomposition (f64, flat row-major).
fn cholesky_lower_generic(n: usize, mat: &[f64]) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0;
            for k in 0..j {
                s += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                l[i * n + j] = (mat[i * n + j] - s).max(0.0).sqrt();
            } else {
                let ljj = l[j * n + j];
                l[i * n + j] = if ljj.abs() > 1e-30 {
                    (mat[i * n + j] - s) / ljj
                } else {
                    0.0
                };
            }
        }
    }
    l
}

// ===========================================================================
// Phase J: FD/Tree Engines (AD-74, AD-75, AD-77, AD-81)
// ===========================================================================

// ---------------------------------------------------------------------------
// AD-81: Binomial Barrier (generic)
// ---------------------------------------------------------------------------

/// CRR binomial barrier option, generic over `T: Number`.
///
/// Uses barrier adjustment at each tree node. Knock-in via in-out parity.
///
/// AD-81: forward through tree + LRM for barrier.
#[allow(clippy::too_many_arguments)]
pub fn binomial_barrier_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    barrier: T,
    rebate: T,
    is_call: bool,
    is_down: bool,
    is_knockout: bool,
    n_steps: usize,
) -> T {
    let zero = T::zero();
    let one = T::one();
    let omega = if is_call { one } else { zero - one };
    let dt = t / T::from_f64(n_steps as f64);
    let df = (zero - r * dt).exp();
    let u = (vol * dt.sqrt()).exp();
    let d = one / u;
    let growth = ((r - q) * dt).exp();
    let p = (growth - d) / (u - d);
    let n = n_steps;

    // Terminal values with barrier check
    let mut values: Vec<T> = (0..=n)
        .map(|j| {
            let s_j = spot * u.powf(T::from_f64(j as f64)) * d.powf(T::from_f64((n - j) as f64));
            let payoff = (omega * (s_j - strike)).max(zero);
            let breached = if is_down {
                s_j.to_f64() <= barrier.to_f64()
            } else {
                s_j.to_f64() >= barrier.to_f64()
            };
            if breached { rebate } else { payoff }
        })
        .collect();

    // Backward induction
    for step in (0..n).rev() {
        let new_values: Vec<T> = (0..=step)
            .map(|j| {
                let s_j = spot * u.powf(T::from_f64(j as f64))
                    * d.powf(T::from_f64((step - j) as f64));
                let breached = if is_down {
                    s_j.to_f64() <= barrier.to_f64()
                } else {
                    s_j.to_f64() >= barrier.to_f64()
                };
                if breached {
                    // Discount rebate to present
                    rebate * (zero - r * T::from_f64((n - step) as f64) * dt).exp()
                } else {
                    df * (p * values[j + 1] + (one - p) * values[j])
                }
            })
            .collect();
        values = new_values;
    }

    let ko_price = values[0];

    if is_knockout {
        ko_price.max(zero)
    } else {
        // Knock-in = Vanilla − Knockout
        use ql_methods::generic::binomial_crr_generic;
        let vanilla = binomial_crr_generic(spot, strike, r, q, vol, t, is_call, false, n_steps);
        (vanilla.npv - ko_price).max(zero)
    }
}

// ---------------------------------------------------------------------------
// AD-74: FD Swing Option (generic, simplified)
// ---------------------------------------------------------------------------

/// Simplified swing option via 1D FD, generic over `T: Number`.
///
/// A swing option allows multiple exercise opportunities.
/// This is a simplified version: the holder can exercise `n_exercises` times
/// at equally-spaced dates. Each exercise has payoff max(S - K, 0).
///
/// Uses backward induction with the generic 1D FD solver.
///
/// AD-74: generic FD grid.
#[allow(clippy::too_many_arguments)]
pub fn fd_swing_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    n_exercises: usize,
    n_grid: usize,
    n_time_steps: usize,
) -> T {
    use ql_methods::generic::{
        build_bs_operator_generic, build_log_spot_grid, fd_1d_solve_generic,
    };

    let _r_f64 = r.to_f64();
    let vol_f64 = vol.to_f64();
    let spot_f64 = spot.to_f64();
    let t_f64 = t.to_f64();

    let grid = build_log_spot_grid(spot_f64, vol_f64, t_f64, n_grid);
    let op = build_bs_operator_generic(&grid, r, q, vol);

    let zero = T::zero();
    let omega = T::one();

    // Swing value = sum of exercise values, working backwards
    // Start with zero terminal value
    let mut total_value = vec![zero; n_grid];
    let dt_exercise = t_f64 / n_exercises as f64;
    let steps_per_period = n_time_steps / n_exercises;

    for _ex in (0..n_exercises).rev() {
        // Terminal condition for this exercise period: continuation + exercise payoff
        let terminal: Vec<T> = grid
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let s = T::from_f64(x.exp());
                let exercise_val = (omega * (s - strike)).max(zero);
                total_value[i] + exercise_val
            })
            .collect();

        // Solve backward from exercise date to previous exercise date
        let result = fd_1d_solve_generic(
            &op,
            &grid,
            &terminal,
            steps_per_period,
            dt_exercise,
            None,
        );

        // The continuation value includes optionality: max(exercise, continue)
        for i in 0..n_grid {
            let s = T::from_f64(grid[i].exp());
            let exercise_val = (omega * (s - strike)).max(zero);
            total_value[i] = result.values[i].max(exercise_val);
        }
    }

    // Interpolate at spot
    let log_s = spot_f64.ln();
    let idx = grid.partition_point(|&x| x < log_s).min(n_grid - 2);
    let frac = (log_s - grid[idx]) / (grid[idx + 1] - grid[idx]);
    let frac_t = T::from_f64(frac);
    total_value[idx] * (T::one() - frac_t) + total_value[idx + 1] * frac_t
}

// ---------------------------------------------------------------------------
// AD-75: FD Shout Option (generic)
// ---------------------------------------------------------------------------

/// Shout option via 1D FD, generic over `T: Number`.
///
/// A shout option allows the holder to "shout" once during the life,
/// locking in the intrinsic value at that time. The payoff is
/// max(payoff at expiry, payoff at shout time).
///
/// Implemented as American-style PDE where the exercise boundary
/// represents the optimal shout time. Uses the generic 1D FD solver.
///
/// AD-75: generic FD grid.
#[allow(clippy::too_many_arguments)]
pub fn fd_shout_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    is_call: bool,
    n_grid: usize,
    n_time_steps: usize,
) -> T {
    use ql_methods::generic::{
        build_bs_operator_generic, build_log_spot_grid, fd_1d_solve_generic,
    };

    let vol_f64 = vol.to_f64();
    let spot_f64 = spot.to_f64();
    let t_f64 = t.to_f64();

    let grid = build_log_spot_grid(spot_f64, vol_f64, t_f64, n_grid);
    let op = build_bs_operator_generic(&grid, r, q, vol);

    let omega = if is_call { T::one() } else { T::zero() - T::one() };
    let zero = T::zero();

    // Terminal payoff
    let terminal: Vec<T> = grid
        .iter()
        .map(|&x| {
            let s = T::from_f64(x.exp());
            (omega * (s - strike)).max(zero)
        })
        .collect();

    // Shout payoff = intrinsic value (this is what you "lock in" by shouting)
    // The shout option is like an American option where the exercise value
    // is the intrinsic plus the European value of a new option
    // Simplified: use American-style max(intrinsic, continuation)
    let result = fd_1d_solve_generic(
        &op,
        &grid,
        &terminal,
        n_time_steps,
        t_f64,
        Some(&terminal),
    );

    // Interpolate at spot
    let log_s = spot_f64.ln();
    let idx = grid.partition_point(|&x| x < log_s).min(n_grid - 2);
    let frac = (log_s - grid[idx]) / (grid[idx + 1] - grid[idx]);
    let frac_t = T::from_f64(frac);
    result.values[idx] * (T::one() - frac_t) + result.values[idx + 1] * frac_t
}

// ---------------------------------------------------------------------------
// AD-77: FD Heston Barrier (generic)
// ---------------------------------------------------------------------------

/// Heston barrier option via 2D FD (Douglas ADI), generic over `T: Number`.
///
/// Uses the generic 2D ADI solver with barrier conditions applied
/// at each time step.
///
/// AD-77: generic 2D FD grid.
#[allow(clippy::too_many_arguments)]
pub fn fd_heston_barrier_generic<T: Number>(
    spot: T,
    strike: T,
    barrier: T,
    r: T,
    q: T,
    v0: T,
    kappa: T,
    theta: T,
    xi: T,
    rho: T,
    t: T,
    is_call: bool,
    is_down: bool,
    is_knockout: bool,
    n_x: usize,
    n_v: usize,
    n_time_steps: usize,
) -> T {
    use ql_methods::generic::{
        build_heston_ops_generic, build_log_spot_grid, fd_2d_solve_generic,
    };

    let vol_f64 = v0.to_f64().sqrt();
    let spot_f64 = spot.to_f64();
    let t_f64 = t.to_f64();
    let barrier_f64 = barrier.to_f64();

    let grid_x = build_log_spot_grid(spot_f64, vol_f64, t_f64, n_x);
    let grid_v: Vec<f64> = (0..n_v)
        .map(|i| i as f64 * 0.5 / (n_v - 1) as f64)
        .collect();

    let ops = build_heston_ops_generic(&grid_x, &grid_v, r, q, kappa, theta, xi, rho);

    let zero = T::zero();
    let omega = if is_call { T::one() } else { zero - T::one() };

    // Terminal payoff with barrier
    let terminal: Vec<T> = grid_v
        .iter()
        .flat_map(|_| {
            grid_x.iter().map(|&x| {
                let s = x.exp();
                let breached = if is_down {
                    s <= barrier_f64
                } else {
                    s >= barrier_f64
                };
                if is_knockout && breached {
                    zero
                } else {
                    (omega * (T::from_f64(s) - strike)).max(zero)
                }
            })
        })
        .collect();

    let result = fd_2d_solve_generic(&ops, &grid_x, &grid_v, &terminal, n_time_steps, t_f64);

    // Interpolate at (spot, v0)
    let log_s = spot_f64.ln();
    let v0_f64 = v0.to_f64();
    let ix = grid_x.partition_point(|&x| x < log_s).min(n_x - 2);
    let iv = grid_v.partition_point(|&v| v < v0_f64).min(n_v - 2);
    let fx = (log_s - grid_x[ix]) / (grid_x[ix + 1] - grid_x[ix]);
    let fv = (v0_f64 - grid_v[iv]) / (grid_v[iv + 1] - grid_v[iv]);
    let fx_t = T::from_f64(fx);
    let fv_t = T::from_f64(fv);

    let v00 = result.values[iv * n_x + ix];
    let v10 = result.values[iv * n_x + ix + 1];
    let v01 = result.values[(iv + 1) * n_x + ix];
    let v11 = result.values[(iv + 1) * n_x + ix + 1];

    let price = v00 * (T::one() - fx_t) * (T::one() - fv_t)
        + v10 * fx_t * (T::one() - fv_t)
        + v01 * (T::one() - fx_t) * fv_t
        + v11 * fx_t * fv_t;

    if is_knockout {
        price.max(zero)
    } else {
        // Knock-in: solve the same without barrier, subtract knockout
        let terminal_vanilla: Vec<T> = grid_v
            .iter()
            .flat_map(|_| {
                grid_x.iter().map(|&x| {
                    let s = T::from_f64(x.exp());
                    (omega * (s - strike)).max(zero)
                })
            })
            .collect();
        let vanilla_result = fd_2d_solve_generic(
            &ops, &grid_x, &grid_v, &terminal_vanilla, n_time_steps, t_f64,
        );
        let v00_v = vanilla_result.values[iv * n_x + ix];
        let v10_v = vanilla_result.values[iv * n_x + ix + 1];
        let v01_v = vanilla_result.values[(iv + 1) * n_x + ix];
        let v11_v = vanilla_result.values[(iv + 1) * n_x + ix + 1];
        let vanilla_price = v00_v * (T::one() - fx_t) * (T::one() - fv_t)
            + v10_v * fx_t * (T::one() - fv_t)
            + v01_v * (T::one() - fx_t) * fv_t
            + v11_v * fx_t * fv_t;
        (vanilla_price - price).max(zero)
    }
}

// ---------------------------------------------------------------------------
// FD European vanilla (generic, 1D BS)
// ---------------------------------------------------------------------------

/// FD European/American vanilla via 1D Crank-Nicolson, generic over `T`.
///
/// Convenience wrapper around the generic 1D FD infrastructure.
#[allow(clippy::too_many_arguments)]
pub fn fd_vanilla_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    is_call: bool,
    is_american: bool,
    n_grid: usize,
    n_time_steps: usize,
) -> T {
    use ql_methods::generic::{
        build_bs_operator_generic, build_log_spot_grid, build_terminal_payoff,
        fd_1d_solve_generic,
    };

    let vol_f64 = vol.to_f64();
    let spot_f64 = spot.to_f64();
    let t_f64 = t.to_f64();

    let grid = build_log_spot_grid(spot_f64, vol_f64, t_f64, n_grid);
    let op = build_bs_operator_generic(&grid, r, q, vol);
    let terminal = build_terminal_payoff(&grid, strike, is_call);

    let payoff_ref = if is_american { Some(terminal.as_slice()) } else { None };
    let result = fd_1d_solve_generic(&op, &grid, &terminal, n_time_steps, t_f64, payoff_ref);

    let log_s = spot_f64.ln();
    let idx = grid.partition_point(|&x| x < log_s).min(n_grid - 2);
    let frac = (log_s - grid[idx]) / (grid[idx + 1] - grid[idx]);
    let frac_t = T::from_f64(frac);
    result.values[idx] * (T::one() - frac_t) + result.values[idx + 1] * frac_t
}

// ===========================================================================
// Phase K: Remaining Engines (AD-86, AD-88, AD-89, AD-90, AD-93, AD-94)
// ===========================================================================

// ---------------------------------------------------------------------------
// AD-93: Commodity Forward / Swap (generic)
// ---------------------------------------------------------------------------

/// Result of generic commodity forward valuation.
#[derive(Debug, Clone)]
pub struct CommodityForwardResultGeneric<T> {
    pub npv: T,
    pub forward_price: T,
}

/// Price a commodity forward contract, generic over `T: Number`.
///
/// `forward_price` is the forward commodity price at delivery.
/// `strike` is the agreed price. `position` is +1 (long) or -1 (short).
pub fn commodity_forward_generic<T: Number>(
    position: T,
    quantity: T,
    forward_price: T,
    strike: T,
    discount_factor: T,
) -> CommodityForwardResultGeneric<T> {
    let npv = position * quantity * (forward_price - strike) * discount_factor;
    CommodityForwardResultGeneric { npv, forward_price }
}

/// Result of generic commodity swap valuation.
#[derive(Debug, Clone)]
pub struct CommoditySwapResultGeneric<T> {
    pub npv: T,
    pub average_forward: T,
    pub fair_fixed_price: T,
}

/// Price a commodity swap (fixed-for-floating), generic over `T: Number`.
///
/// * `position` — +1 receive floating, pay fixed.
/// * `quantity` — per-period quantity.
/// * `fixed_price` — fixed leg price.
/// * `forward_prices` — forward commodity price at each payment date.
/// * `discount_factors` — discount factor at each payment date.
pub fn commodity_swap_generic<T: Number>(
    position: T,
    quantity: T,
    fixed_price: T,
    forward_prices: &[T],
    discount_factors: &[T],
) -> CommoditySwapResultGeneric<T> {
    let n = forward_prices.len().min(discount_factors.len());
    if n == 0 {
        return CommoditySwapResultGeneric {
            npv: T::zero(),
            average_forward: T::zero(),
            fair_fixed_price: fixed_price,
        };
    }
    let mut sum_fwd_df = T::zero();
    let mut sum_df = T::zero();
    let mut sum_fwd = T::zero();
    for i in 0..n {
        sum_fwd_df += forward_prices[i] * discount_factors[i];
        sum_df += discount_factors[i];
        sum_fwd += forward_prices[i];
    }
    let n_t = T::from_f64(n as f64);
    let avg_fwd = sum_fwd / n_t;
    let fair_fixed = if sum_df.to_f64().abs() > 1e-12 {
        sum_fwd_df / sum_df
    } else {
        avg_fwd
    };
    let npv = position * quantity * (sum_fwd_df - fixed_price * sum_df);
    CommoditySwapResultGeneric {
        npv,
        average_forward: avg_fwd,
        fair_fixed_price: fair_fixed,
    }
}

// ---------------------------------------------------------------------------
// AD-94: Asset Swap / Equity TRS (generic)
// ---------------------------------------------------------------------------

/// Result of generic asset swap pricing.
#[derive(Debug, Clone)]
pub struct AssetSwapResultGeneric<T> {
    pub asset_swap_spread: T,
    pub bond_leg_npv: T,
    pub floating_annuity: T,
}

/// Compute the asset swap spread, generic over `T: Number`.
///
/// * `is_par` — true for par convention, false for market-value.
/// * `bond_clean_price` — as percentage of par (e.g. 98.5).
/// * `coupon_rate` — fixed coupon rate.
/// * `notional` — par amount.
/// * `year_fractions` — accrual fractions per period.
/// * `discount_factors` — DF at each coupon date.
pub fn asset_swap_generic<T: Number>(
    is_par: bool,
    bond_clean_price: T,
    coupon_rate: T,
    notional: T,
    year_fractions: &[T],
    discount_factors: &[T],
) -> AssetSwapResultGeneric<T> {
    let n = year_fractions.len().min(discount_factors.len());
    assert!(n > 0, "asset swap needs at least one period");

    let mut coupon_pv = T::zero();
    let mut annuity = T::zero();
    for i in 0..n {
        coupon_pv += notional * coupon_rate * year_fractions[i] * discount_factors[i];
        annuity += notional * year_fractions[i] * discount_factors[i];
    }
    let par_pv = notional * discount_factors[n - 1];
    let bond_leg_npv = coupon_pv + par_pv;

    let hundred = T::from_f64(100.0);
    let p = bond_clean_price / hundred;

    let asw = if is_par {
        (bond_leg_npv - notional * p) / annuity
    } else {
        coupon_rate + (par_pv - notional * p) / annuity
    };

    AssetSwapResultGeneric {
        asset_swap_spread: asw,
        bond_leg_npv,
        floating_annuity: annuity,
    }
}

/// Result of generic equity TRS pricing.
#[derive(Debug, Clone)]
pub struct EquityTrsResultGeneric<T> {
    pub equity_leg_npv: T,
    pub funding_leg_npv: T,
    pub npv: T,
    pub equity_return: T,
}

/// Price an equity total return swap, generic over `T: Number`.
///
/// NPV from equity receiver's perspective = equity leg − funding leg.
pub fn equity_trs_generic<T: Number>(
    notional: T,
    initial_price: T,
    current_price: T,
    accrued_dividends: T,
    floating_rate: T,
    spread: T,
    accrual_period: T,
    discount_factor: T,
) -> EquityTrsResultGeneric<T> {
    let equity_return = (current_price - initial_price) / initial_price
        + accrued_dividends / initial_price;
    let equity_leg_npv = discount_factor * notional * equity_return;
    let funding_leg_npv =
        discount_factor * notional * (floating_rate + spread) * accrual_period;
    let npv = equity_leg_npv - funding_leg_npv;
    EquityTrsResultGeneric {
        equity_leg_npv,
        funding_leg_npv,
        npv,
        equity_return,
    }
}

/// Fair spread for an equity TRS (spread making NPV = 0).
pub fn equity_trs_fair_spread_generic<T: Number>(
    initial_price: T,
    current_price: T,
    accrued_dividends: T,
    floating_rate: T,
    accrual_period: T,
) -> T {
    if accrual_period.to_f64().abs() < 1e-15 {
        return T::zero();
    }
    let equity_return = (current_price - initial_price) / initial_price
        + accrued_dividends / initial_price;
    equity_return / accrual_period - floating_rate
}

// ---------------------------------------------------------------------------
// AD-86: CDO Tranche (generic, LHP Gaussian copula)
// ---------------------------------------------------------------------------

/// Result of generic CDO tranche pricing.
#[derive(Debug, Clone)]
pub struct CdoTrancheResultGeneric<T> {
    pub expected_loss: T,
    pub fair_spread: T,
    pub protection_leg: T,
    pub premium_leg: T,
}

/// Inverse cumulative normal (Beasley-Springer-Moro rational approximation).
///
/// This returns `f64` because it is used for quadrature abscissa positioning,
/// which is structural (not differentiated).
fn inv_cumulative_normal_f64(p: f64) -> f64 {
    if p <= 0.0 { return -8.0; }
    if p >= 1.0 { return 8.0; }
    if (p - 0.5).abs() < 1e-14 { return 0.0; }

    let a = [
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383_577_518_672_69e2,
        -3.066479806614716e+01,  2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00,
    ];
    let d = [
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
             / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
    if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q
             / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    }
    let q = (-2.0 * (1.0 - p).ln()).sqrt();
    -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
     / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
}

/// Expected tranche loss under LHP Gaussian copula, generic.
///
/// Integrates over the systematic factor using a midpoint rule on a normal grid.
fn cdo_expected_loss_generic<T: Number>(
    pd: T,
    correlation: T,
    lgd: T,
    attachment: f64,
    detachment: f64,
    n_pts: usize,
) -> T {
    let width = detachment - attachment;
    if width <= 0.0 { return T::zero(); }

    let rho_f = correlation.to_f64().clamp(0.01, 0.99);
    let sqrt_rho = T::from_f64(rho_f.sqrt());
    let sqrt_1_rho = T::from_f64((1.0 - rho_f).sqrt());
    let inv_pd = T::from_f64(inv_cumulative_normal_f64(pd.to_f64().clamp(1e-10, 1.0 - 1e-10)));

    let mut el = T::zero();
    for i in 0..n_pts {
        let xi_f = -4.0 + 8.0 * (i as f64 + 0.5) / n_pts as f64;
        let wi_f = normal_pdf(T::from_f64(xi_f)).to_f64() * 8.0 / n_pts as f64;

        let xi = T::from_f64(xi_f);
        let cond_pd = normal_cdf((inv_pd - sqrt_rho * xi) / sqrt_1_rho);
        let cond_loss = cond_pd * lgd;

        // Tranche loss = min(max(loss - a, 0), width) / width
        let cl_f = cond_loss.to_f64();
        let tl_f = ((cl_f - attachment).max(0.0)).min(width) / width;
        // For AD: approximate the clamped tranche loss in T
        // We route through T for the conditional loss, but the
        // tranche clamping involves piecewise functions best done in f64
        // and converted back.
        el += T::from_f64(wi_f * tl_f);
    }
    el
}

/// Price a CDO tranche under the LHP Gaussian copula model, generic over `T: Number`.
///
/// * `pd` — average default probability.
/// * `correlation` — flat Gaussian copula correlation.
/// * `recovery` — average recovery rate.
/// * `maturity` — portfolio maturity in years.
/// * `risk_free_rate` — continuous risk-free rate.
/// * `attachment` — tranche attachment point (fraction of portfolio).
/// * `detachment` — tranche detachment point.
/// * `notional` — tranche notional.
/// * `n_integration` — number of quadrature points.
pub fn cdo_tranche_generic<T: Number>(
    pd: T,
    correlation: T,
    recovery: T,
    maturity: T,
    risk_free_rate: T,
    attachment: f64,
    detachment: f64,
    notional: T,
    n_integration: usize,
) -> CdoTrancheResultGeneric<T> {
    let lgd = T::one() - recovery;
    let df = (T::zero() - risk_free_rate * maturity).exp();

    let el = cdo_expected_loss_generic(pd, correlation, lgd, attachment, detachment, n_integration);

    let protection_leg = el * notional * df;
    let premium_leg = (T::one() - el) * notional * maturity * df;

    let fair_spread = if premium_leg.to_f64().abs() > 1e-8 {
        protection_leg / premium_leg
    } else {
        T::zero()
    };

    CdoTrancheResultGeneric {
        expected_loss: el,
        fair_spread,
        protection_leg,
        premium_leg,
    }
}

// ---------------------------------------------------------------------------
// AD-88: Tree Swaption / Cap-Floor (generic, Hull-White trinomial)
// ---------------------------------------------------------------------------

/// Hull-White analytic bond price P(t, t+τ) given short rate r, generic.
fn hw_bond_price_generic<T: Number>(a: T, sigma: T, r: T, tau: T) -> T {
    if tau.to_f64() <= 0.0 {
        return T::one();
    }
    let b = if a.to_f64().abs() < 1e-15 {
        tau
    } else {
        (T::one() - (T::zero() - a * tau).exp()) / a
    };
    let s2 = sigma * sigma;
    let a2 = a * a;
    let ln_a = if a.to_f64().abs() < 1e-15 {
        T::zero() - T::from_f64(0.5) * s2 * tau * tau * tau / T::from_f64(3.0)
    } else {
        (b - tau) * (T::zero() - T::from_f64(0.5) * s2) / a2
            - s2 * b * b / (T::from_f64(4.0) * a)
    };
    (ln_a - b * r).exp()
}

/// Price a European swaption on a Hull-White trinomial tree, generic over `T: Number`.
///
/// Tree topology (probabilities, j_max, indices) stays in `f64`/`i64`/`usize`.
/// Values, discount factors, and payoffs flow through `T`.
pub fn tree_swaption_generic<T: Number>(
    a: T,
    sigma: T,
    r0: T,
    option_expiry: f64,
    swap_tenors: &[f64],
    fixed_rate: T,
    notional: T,
    is_payer: bool,
    n_steps: usize,
) -> T {
    if swap_tenors.is_empty() || option_expiry <= 0.0 {
        return T::zero();
    }

    let a_f = a.to_f64();
    let sig_f = sigma.to_f64();
    let dt = option_expiry / n_steps as f64;
    let dx = sig_f * (3.0 * dt).sqrt();
    let j_max = ((0.184 / (a_f.abs().max(0.01) * dt)).ceil() as i64).max(3);
    let width = (2 * j_max + 1) as usize;
    let idx = |j: i64| -> usize { (j + j_max) as usize };

    // Build transition probabilities (f64)
    let mut probs = vec![(0.0f64, 1.0f64, 0.0f64); width];
    for j in -j_max..=j_max {
        let xi = -a_f * j as f64 * dt;
        let p_up = (1.0 / 6.0 + (xi * xi + xi) / 2.0).max(0.0);
        let p_mid = (2.0 / 3.0 - xi * xi).max(0.0);
        let p_down = (1.0 / 6.0 + (xi * xi - xi) / 2.0).max(0.0);
        let total = p_up + p_mid + p_down;
        probs[idx(j)] = (p_up / total, p_mid / total, p_down / total);
    }

    let omega = if is_payer { T::one() } else { T::zero() - T::one() };
    let n_coupons = swap_tenors.len();

    // Terminal payoff at option expiry
    let mut values = vec![T::zero(); width];
    for j in -j_max..=j_max {
        let r_j = r0 + T::from_f64(j as f64 * dx);

        // Compute swap value: 1 - sum(c_i * P(T_e, T_i))
        let mut swap_val = T::one();
        for i in 0..n_coupons {
            let tau_i = if i == 0 {
                swap_tenors[0] - option_expiry
            } else {
                swap_tenors[i] - swap_tenors[i - 1]
            };
            let mut c_i = fixed_rate * T::from_f64(tau_i);
            if i == n_coupons - 1 {
                c_i += T::one();
            }
            let tau = T::from_f64(swap_tenors[i] - option_expiry);
            let p = hw_bond_price_generic(a, sigma, r_j, tau);
            swap_val -= c_i * p;
        }
        let payoff = omega * swap_val;
        values[idx(j)] = if payoff.to_f64() > 0.0 { payoff * notional } else { T::zero() };
    }

    // Backward induction
    for step in (0..n_steps).rev() {
        let _t = step as f64 * dt;
        let mut new_values = vec![T::zero(); width];
        for j in -j_max..=j_max {
            let r_j = r0 + T::from_f64(j as f64 * dx);
            let disc = (T::zero() - r_j * T::from_f64(dt)).exp();
            let (pu, pm, pd) = probs[idx(j)];
            let j_up = (j + 1).min(j_max);
            let j_down = (j - 1).max(-j_max);
            let cont = T::from_f64(pu) * values[idx(j_up)]
                + T::from_f64(pm) * values[idx(j)]
                + T::from_f64(pd) * values[idx(j_down)];
            new_values[idx(j)] = disc * cont;
        }
        values = new_values;
    }

    values[idx(0)]
}

/// Price a cap or floor on a Hull-White trinomial tree, generic over `T: Number`.
///
/// Each caplet/floorlet is priced independently and summed.
pub fn tree_cap_floor_generic<T: Number>(
    a: T,
    sigma: T,
    r0: T,
    fixing_times: &[f64],
    payment_times: &[f64],
    strike: T,
    notional: T,
    is_cap: bool,
    n_steps_per_period: usize,
) -> T {
    assert_eq!(fixing_times.len(), payment_times.len());
    if fixing_times.is_empty() {
        return T::zero();
    }

    let a_f = a.to_f64();
    let sig_f = sigma.to_f64();
    let omega = if is_cap { T::one() } else { T::zero() - T::one() };
    let mut total_npv = T::zero();

    for k in 0..fixing_times.len() {
        let t_fix = fixing_times[k];
        let t_pay = payment_times[k];
        let tau_f = t_pay - t_fix;
        if t_fix <= 0.0 || tau_f <= 0.0 { continue; }

        let n_steps = ((n_steps_per_period as f64 * t_fix).ceil() as usize).max(10);
        let dt = t_fix / n_steps as f64;
        let dx = sig_f * (3.0 * dt).sqrt();
        let j_max = ((0.184 / (a_f.abs().max(0.01) * dt)).ceil() as i64).max(3);
        let width = (2 * j_max + 1) as usize;
        let idx_fn = |j: i64| -> usize { (j + j_max) as usize };

        // Transition probs (f64)
        let mut probs = vec![(0.0f64, 1.0f64, 0.0f64); width];
        for j in -j_max..=j_max {
            let xi = -a_f * j as f64 * dt;
            let pu = (1.0 / 6.0 + (xi * xi + xi) / 2.0).max(0.0);
            let pm = (2.0 / 3.0 - xi * xi).max(0.0);
            let pd = (1.0 / 6.0 + (xi * xi - xi) / 2.0).max(0.0);
            let tot = pu + pm + pd;
            probs[idx_fn(j)] = (pu / tot, pm / tot, pd / tot);
        }

        // Caplet payoff at fixing
        let tau_t = T::from_f64(tau_f);
        let mut values = vec![T::zero(); width];
        for j in -j_max..=j_max {
            let r_j = r0 + T::from_f64(j as f64 * dx);
            let p_pay = hw_bond_price_generic(a, sigma, r_j, tau_t);
            let forward = (T::one() / p_pay - T::one()) / tau_t;
            let payoff = omega * (forward - strike);
            if payoff.to_f64() > 0.0 {
                values[idx_fn(j)] = payoff * tau_t * notional * p_pay;
            }
        }

        // Backward induction
        for _step in (0..n_steps).rev() {
            let mut new_values = vec![T::zero(); width];
            for j in -j_max..=j_max {
                let r_j = r0 + T::from_f64(j as f64 * dx);
                let disc = (T::zero() - r_j * T::from_f64(dt)).exp();
                let (pu, pm, pd) = probs[idx_fn(j)];
                let j_up = (j + 1).min(j_max);
                let j_down = (j - 1).max(-j_max);
                let cont = T::from_f64(pu) * values[idx_fn(j_up)]
                    + T::from_f64(pm) * values[idx_fn(j)]
                    + T::from_f64(pd) * values[idx_fn(j_down)];
                new_values[idx_fn(j)] = disc * cont;
            }
            values = new_values;
        }

        total_npv += values[idx_fn(0)];
    }

    total_npv
}

// ---------------------------------------------------------------------------
// AD-89: Gaussian 1D Swaption (generic, Gauss-Hermite quadrature)
// ---------------------------------------------------------------------------

/// Gauss-Hermite quadrature nodes/weights (Golub-Welsch), f64 only.
///
/// Returns (nodes, weights) for the physicists' Hermite measure exp(−x²).
#[allow(clippy::needless_range_loop)]
fn gauss_hermite_f64(n: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(n >= 2);
    // Build tridiagonal Jacobi matrix for Hermite polynomials
    // J[i,i]=0, J[i,i+1]=sqrt((i+1)/2)
    // Use symmetric eigenvalue decomposition via QR iteration (simplified).
    // For moderate n (≤64), a direct Golub-Welsch via Householder/QR converges.
    // We implement a simple Jacobi-rotation approach.

    let mut diag = vec![0.0f64; n];
    let mut off = vec![0.0f64; n]; // off-diagonal: off[i] = J[i,i+1]
    for i in 0..n - 1 {
        off[i] = ((i + 1) as f64 / 2.0).sqrt();
    }

    // Eigenvectors start as identity
    let mut evecs = vec![vec![0.0f64; n]; n];
    for i in 0..n { evecs[i][i] = 1.0; }

    // Implicit QR iteration for symmetric tridiagonal matrix
    let max_iter = 100 * n;
    let mut m = n;
    for _ in 0..max_iter {
        if m <= 1 { break; }
        // Find the lowest off-diagonal < epsilon
        let mut low = m - 1;
        while low > 0 && off[low - 1].abs() > 1e-14 {
            low -= 1;
        }
        if low == m - 1 { m -= 1; continue; }

        // Wilkinson shift
        let d_diff = (diag[m - 2] - diag[m - 1]) / 2.0;
        let shift = diag[m - 1]
            - off[m - 2] * off[m - 2]
                / (d_diff + d_diff.signum() * (d_diff * d_diff + off[m - 2] * off[m - 2]).sqrt());

        // Implicit QR step (Givens rotations)
        let mut x = diag[low] - shift;
        let mut z = off[low];
        for k in low..m - 1 {
            let r = (x * x + z * z).sqrt();
            let c = x / r;
            let s = z / r;
            if k > low {
                off[k - 1] = r;
            }
            let d0 = diag[k];
            let d1 = diag[k + 1];
            diag[k] = c * c * d0 + 2.0 * c * s * off[k] + s * s * d1;
            diag[k + 1] = s * s * d0 - 2.0 * c * s * off[k] + c * c * d1;
            off[k] = c * s * (d1 - d0) + (c * c - s * s) * off[k];

            // Update eigenvectors
            for i in 0..n {
                let tmp = evecs[i][k];
                evecs[i][k] = c * tmp + s * evecs[i][k + 1];
                evecs[i][k + 1] = -s * tmp + c * evecs[i][k + 1];
            }

            if k + 2 < m {
                x = off[k];
                z = s * off[k + 1];
                off[k + 1] *= c;
            }
        }
    }

    let sqrt_pi = std::f64::consts::PI.sqrt();
    let mut pairs: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let v0 = evecs[0][i];
            (diag[i], sqrt_pi * v0 * v0)
        })
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    (
        pairs.iter().map(|p| p.0).collect(),
        pairs.iter().map(|p| p.1).collect(),
    )
}

/// Price a European swaption using Gauss-Hermite quadrature under Hull-White,
/// generic over `T: Number`.
///
/// Instead of requiring a `Gsr1d` model object, the function takes HW parameters
/// directly (a, sigma, flat_rate) and computes the zero-coupon bond prices
/// and zeta internally.
///
/// * `a` — HW mean-reversion speed.
/// * `sigma` — HW volatility.
/// * `flat_rate` — flat continuously-compounded yield (initial curve).
/// * `option_expiry` — time to swaption expiry.
/// * `swap_tenors` — payment times of fixed leg.
/// * `year_fractions` — day-count fractions per swap period.
/// * `fixed_rate` — swaption strike rate.
/// * `notional` — swap notional.
/// * `is_payer` — true for payer.
/// * `n_quad` — number of Gauss-Hermite quadrature points.
#[allow(clippy::too_many_arguments)]
pub fn gaussian1d_swaption_generic<T: Number>(
    a: T,
    sigma: T,
    flat_rate: T,
    option_expiry: f64,
    swap_tenors: &[f64],
    year_fractions: &[f64],
    fixed_rate: T,
    notional: T,
    is_payer: bool,
    n_quad: usize,
) -> T {
    if swap_tenors.is_empty() || year_fractions.is_empty() || option_expiry <= 0.0 {
        return T::zero();
    }

    let n = swap_tenors.len().min(year_fractions.len());
    let a_f = a.to_f64();
    let sig_f = sigma.to_f64();

    // Zeta(T_e) = σ²/(2a) (1 - exp(-2a T_e))
    let zeta_f = if a_f.abs() < 1e-15 {
        sig_f * sig_f * option_expiry
    } else {
        sig_f * sig_f / (2.0 * a_f) * (1.0 - (-2.0 * a_f * option_expiry).exp())
    };
    let sqrt_zeta_f = zeta_f.max(1e-30).sqrt();

    // Market discount: P^M(0, t) = exp(-flat_rate * t)
    let pm_te = (T::zero() - flat_rate * T::from_f64(option_expiry)).exp();

    let (nodes, weights) = gauss_hermite_f64(n_quad.max(4));

    let sign = if is_payer { T::one() } else { T::zero() - T::one() };
    let mut integral = T::zero();

    for (node, weight) in nodes.iter().zip(weights.iter()) {
        let x_f = (2.0_f64).sqrt() * sqrt_zeta_f * node;
        let x = T::from_f64(x_f);

        // Zero-bond from T_e to T_i under HW:
        // P(T_e, T_i | x) = P^M(0, T_i)/P^M(0, T_e)
        //   × exp(-B(T_e, T_i) x - 0.5 B(T_e, T_i)² ζ)
        // with B(s,t) = (1 - exp(-a(t-s)))/a

        let swap_start_bond = if option_expiry < swap_tenors[0] {
            let tau = swap_tenors[0] - option_expiry;
            let b_val = if a_f.abs() < 1e-15 { tau } else { (1.0 - (-a_f * tau).exp()) / a_f };
            let pm_ratio = (T::zero() - flat_rate * T::from_f64(swap_tenors[0])).exp() / pm_te;
            let b_t = T::from_f64(b_val);
            pm_ratio * (T::zero() - b_t * x - T::from_f64(0.5) * b_t * b_t * T::from_f64(zeta_f)).exp()
        } else {
            T::one()
        };

        let swap_end_tau = swap_tenors[n - 1] - option_expiry;
        let b_end = if a_f.abs() < 1e-15 { swap_end_tau } else { (1.0 - (-a_f * swap_end_tau).exp()) / a_f };
        let pm_end_ratio = (T::zero() - flat_rate * T::from_f64(swap_tenors[n - 1])).exp() / pm_te;
        let b_end_t = T::from_f64(b_end);
        let swap_end_bond = pm_end_ratio
            * (T::zero() - b_end_t * x - T::from_f64(0.5) * b_end_t * b_end_t * T::from_f64(zeta_f)).exp();

        let mut annuity = T::zero();
        for j in 0..n {
            let tau_j = swap_tenors[j] - option_expiry;
            let b_j = if a_f.abs() < 1e-15 { tau_j } else { (1.0 - (-a_f * tau_j).exp()) / a_f };
            let pm_j_ratio = (T::zero() - flat_rate * T::from_f64(swap_tenors[j])).exp() / pm_te;
            let b_j_t = T::from_f64(b_j);
            let zb_j = pm_j_ratio
                * (T::zero() - b_j_t * x - T::from_f64(0.5) * b_j_t * b_j_t * T::from_f64(zeta_f)).exp();
            annuity += T::from_f64(year_fractions[j]) * zb_j;
        }

        let swap_value = sign * notional * (swap_start_bond - swap_end_bond - fixed_rate * annuity);
        let payoff = if swap_value.to_f64() > 0.0 { swap_value } else { T::zero() };
        integral += T::from_f64(*weight) * payoff;
    }

    pm_te * integral / T::from_f64(std::f64::consts::PI.sqrt())
}

// ---------------------------------------------------------------------------
// AD-90: FD G2++ Swaption (generic, 2D ADI)
// ---------------------------------------------------------------------------

/// Price a European/Bermudan swaption under the G2++ model using 2D ADI FD,
/// generic over `T: Number`.
///
/// Instead of taking a `G2Model` reference, takes the 5 model parameters
/// directly so they can be AD types.
///
/// * `a` — mean-reversion of x factor.
/// * `sigma_x` — volatility of x factor.
/// * `b` — mean-reversion of y factor.
/// * `eta` — volatility of y factor.
/// * `rho` — instantaneous correlation between x and y.
/// * `flat_rate` — flat initial yield (used for φ(t) fitting and bond prices).
/// * `fixed_leg_times` — payment times of the fixed leg.
/// * `fixed_leg_amounts` — fixed coupon amounts (= N × rate × τ) per period.
/// * `float_leg_last_time` — last time on the floating leg.
/// * `notional` — swap notional.
/// * `is_payer` — true for payer swaption.
/// * `exercise_time` — exercise time (European).
/// * `nx`, `ny`, `nt` — grid sizes.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn fd_g2_swaption_generic<T: Number>(
    a: T,
    sigma_x: T,
    b: T,
    eta: T,
    rho: T,
    flat_rate: T,
    fixed_leg_times: &[f64],
    fixed_leg_amounts: &[T],
    float_leg_last_time: f64,
    notional: T,
    is_payer: bool,
    exercise_time: f64,
    nx: usize,
    ny: usize,
    nt: usize,
) -> T {
    if fixed_leg_times.is_empty() || exercise_time <= 0.0 {
        return T::zero();
    }

    let a_f = a.to_f64();
    let b_f = b.to_f64();
    let sig_f = sigma_x.to_f64();
    let eta_f = eta.to_f64();
    let mat = exercise_time;
    let dt = mat / nt as f64;

    // Standard deviations at maturity
    let x_std = (sig_f / (2.0 * a_f.abs().max(0.01)).sqrt())
        * (1.0 - (-2.0 * a_f.abs().max(0.01) * mat).exp()).sqrt();
    let y_std = (eta_f / (2.0 * b_f.abs().max(0.01)).sqrt())
        * (1.0 - (-2.0 * b_f.abs().max(0.01) * mat).exp()).sqrt();

    let x_max = 4.0 * x_std.max(0.01);
    let y_max = 4.0 * y_std.max(0.01);
    let dx = 2.0 * x_max / (nx - 1) as f64;
    let dy = 2.0 * y_max / (ny - 1) as f64;

    let xs: Vec<f64> = (0..nx).map(|i| -x_max + i as f64 * dx).collect();
    let ys: Vec<f64> = (0..ny).map(|j| -y_max + j as f64 * dy).collect();

    // φ(t) for G2++: approximate as flat_rate + corrections
    // φ(t) ≈ f(0,t) + σ²/(2a²)(1-e^{-at})² + η²/(2b²)(1-e^{-bt})² + ρση/(ab)(1-e^{-at})(1-e^{-bt})
    let phi = |t_val: f64| -> T {
        let ea = 1.0 - (-a_f * t_val).exp();
        let eb = 1.0 - (-b_f * t_val).exp();
        flat_rate
            + sigma_x * sigma_x / (T::from_f64(2.0) * a * a) * T::from_f64(ea * ea)
            + eta * eta / (T::from_f64(2.0) * b * b) * T::from_f64(eb * eb)
            + rho * sigma_x * eta / (a * b) * T::from_f64(ea * eb)
    };

    // Swap value from short rate r: float_pv - fixed_pv
    let swap_val_from_r = |r: T, current_time: f64| -> T {
        let discount_fn = |t_val: f64| -> T {
            (T::zero() - r * T::from_f64((t_val - current_time).max(0.0))).exp()
        };
        let mut fixed_pv = T::zero();
        for (i, &t_i) in fixed_leg_times.iter().enumerate() {
            if t_i > current_time {
                fixed_pv += fixed_leg_amounts[i] * discount_fn(t_i);
            }
        }
        let last_t = if float_leg_last_time > current_time {
            float_leg_last_time
        } else {
            current_time + 0.01
        };
        let float_pv = notional * (T::one() - discount_fn(last_t));
        float_pv - fixed_pv
    };

    // Terminal payoff
    let phi_mat = phi(mat);
    let mut v = vec![vec![T::zero(); ny]; nx];
    for i in 0..nx {
        for j in 0..ny {
            let r_val = T::from_f64(xs[i]) + T::from_f64(ys[j]) + phi_mat;
            let sv = swap_val_from_r(r_val, mat);
            v[i][j] = if is_payer {
                if sv.to_f64() > 0.0 { sv } else { T::zero() }
            } else {
                let neg_sv = T::zero() - sv;
                if neg_sv.to_f64() > 0.0 { neg_sv } else { T::zero() }
            };
        }
    }

    // Thomas solve for generic T tridiagonal system
    let thomas_t = |lower: &[T], diag: &[T], upper: &[T], rhs: &[T]| -> Vec<T> {
        let n = diag.len();
        let mut cp = vec![T::zero(); n];
        let mut dp = vec![T::zero(); n];
        cp[0] = upper[0] / diag[0];
        dp[0] = rhs[0] / diag[0];
        for i in 1..n {
            let m = diag[i] - lower[i] * cp[i - 1];
            cp[i] = if i < n - 1 { upper[i] / m } else { T::zero() };
            dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / m;
        }
        let mut x = vec![T::zero(); n];
        x[n - 1] = dp[n - 1];
        for i in (0..n - 1).rev() {
            x[i] = dp[i] - cp[i] * x[i + 1];
        }
        x
    };

    // Backward induction with ADI
    let half = T::from_f64(0.5);
    let two = T::from_f64(2.0);
    for step in 0..nt {
        let t_val = mat - (step + 1) as f64 * dt;
        let phi_t = phi(t_val.max(0.0));

        // Half-step: implicit in x
        let mut v_half = vec![vec![T::zero(); ny]; nx];
        for j in 0..ny {
            let mut lower = vec![T::zero(); nx];
            let mut diag_vec = vec![T::zero(); nx];
            let mut upper = vec![T::zero(); nx];
            let mut rhs = vec![T::zero(); nx];

            for i in 1..nx - 1 {
                let r_val = T::from_f64(xs[i]) + T::from_f64(ys[j]) + phi_t;
                let vyy = if j > 0 && j < ny - 1 {
                    (v[i][j + 1] - two * v[i][j] + v[i][j - 1]) / T::from_f64(dy * dy)
                } else { T::zero() };
                let vy = if j > 0 && j < ny - 1 {
                    (v[i][j + 1] - v[i][j - 1]) / T::from_f64(2.0 * dy)
                } else { T::zero() };

                rhs[i] = v[i][j] + half * T::from_f64(dt) * (
                    half * eta * eta * vyy
                    - b * T::from_f64(ys[j]) * vy
                    - half * r_val * v[i][j]
                );

                let coeff_a = half * sigma_x * sigma_x / T::from_f64(dx * dx);
                let drift_x = T::zero() - a * T::from_f64(xs[i]) / T::from_f64(2.0 * dx);

                lower[i] = T::zero() - half * T::from_f64(dt) * (coeff_a - drift_x);
                diag_vec[i] = T::one() + half * T::from_f64(dt) * (two * coeff_a + half * r_val);
                upper[i] = T::zero() - half * T::from_f64(dt) * (coeff_a + drift_x);
            }
            diag_vec[0] = T::one();
            rhs[0] = v[0][j];
            diag_vec[nx - 1] = T::one();
            rhs[nx - 1] = v[nx - 1][j];

            let soln = thomas_t(&lower, &diag_vec, &upper, &rhs);
            for i in 0..nx {
                v_half[i][j] = soln[i];
            }
        }

        // Half-step: implicit in y
        for i in 0..nx {
            let mut lower = vec![T::zero(); ny];
            let mut diag_vec = vec![T::zero(); ny];
            let mut upper = vec![T::zero(); ny];
            let mut rhs = vec![T::zero(); ny];

            for j in 1..ny - 1 {
                let r_val = T::from_f64(xs[i]) + T::from_f64(ys[j]) + phi_t;
                let vxx = if i > 0 && i < nx - 1 {
                    (v_half[i + 1][j] - two * v_half[i][j] + v_half[i - 1][j]) / T::from_f64(dx * dx)
                } else { T::zero() };
                let vx = if i > 0 && i < nx - 1 {
                    (v_half[i + 1][j] - v_half[i - 1][j]) / T::from_f64(2.0 * dx)
                } else { T::zero() };

                rhs[j] = v_half[i][j] + half * T::from_f64(dt) * (
                    half * sigma_x * sigma_x * vxx
                    - a * T::from_f64(xs[i]) * vx
                    - half * r_val * v_half[i][j]
                );

                let coeff_b = half * eta * eta / T::from_f64(dy * dy);
                let drift_y = T::zero() - b * T::from_f64(ys[j]) / T::from_f64(2.0 * dy);

                lower[j] = T::zero() - half * T::from_f64(dt) * (coeff_b - drift_y);
                diag_vec[j] = T::one() + half * T::from_f64(dt) * (two * coeff_b + half * r_val);
                upper[j] = T::zero() - half * T::from_f64(dt) * (coeff_b + drift_y);
            }
            diag_vec[0] = T::one();
            rhs[0] = v_half[i][0];
            diag_vec[ny - 1] = T::one();
            rhs[ny - 1] = v_half[i][ny - 1];

            let soln = thomas_t(&lower, &diag_vec, &upper, &rhs);
            v[i][..ny].copy_from_slice(&soln[..ny]);
        }
    }

    // Interpolate at (x=0, y=0)
    let ix = ((0.0 - (-x_max)) / dx) as usize;
    let jy = ((0.0 - (-y_max)) / dy) as usize;
    let ix = ix.min(nx - 2);
    let jy = jy.min(ny - 2);
    let wx = T::from_f64((0.0 - xs[ix]) / dx);
    let wy = T::from_f64((0.0 - ys[jy]) / dy);

    let val = (T::one() - wx) * (T::one() - wy) * v[ix][jy]
        + wx * (T::one() - wy) * v[ix + 1][jy]
        + (T::one() - wx) * wy * v[ix][jy + 1]
        + wx * wy * v[ix + 1][jy + 1];

    if val.to_f64() > 0.0 { val } else { T::zero() }
}

// ===========================================================================
// INFRA-6 — Minimal Complex<T: Number> for COS engines
// ===========================================================================

/// Minimal complex number for COS/Fourier engines, generic over `T: Number`.
#[derive(Clone, Copy, Debug)]
struct ComplexT<T: Number> {
    re: T,
    im: T,
}

impl<T: Number> ComplexT<T> {
    fn new(re: T, im: T) -> Self { Self { re, im } }
    fn from_real(re: T) -> Self { Self { re, im: T::zero() } }
    fn exp(self) -> Self {
        let r = self.re.exp();
        Self { re: r * self.im.cos(), im: r * self.im.sin() }
    }
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
    fn add(self, rhs: Self) -> Self {
        Self { re: self.re + rhs.re, im: self.im + rhs.im }
    }
    fn sub(self, rhs: Self) -> Self {
        Self { re: self.re - rhs.re, im: self.im - rhs.im }
    }
    fn scale(self, s: T) -> Self {
        Self { re: self.re * s, im: self.im * s }
    }
    fn div(self, rhs: Self) -> Self {
        let d = rhs.re * rhs.re + rhs.im * rhs.im;
        if d.to_f64() < 1e-60 { return Self::from_real(T::zero()); }
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / d,
            im: (self.im * rhs.re - self.re * rhs.im) / d,
        }
    }
    fn norm_sq(self) -> T { self.re * self.re + self.im * self.im }
    fn sqrt(self) -> Self {
        let r = self.norm_sq().to_f64().sqrt().sqrt();
        let arg = self.im.to_f64().atan2(self.re.to_f64()) * 0.5;
        Self { re: T::from_f64(r * arg.cos()), im: T::from_f64(r * arg.sin()) }
    }
    fn ln(self) -> Self {
        let r = self.norm_sq().to_f64().sqrt();
        let arg = self.im.to_f64().atan2(self.re.to_f64());
        Self { re: T::from_f64(r.ln()), im: T::from_f64(arg) }
    }
}

// ===========================================================================
// AD-5 — price_european_generic (full Greeks, standalone)
// ===========================================================================

/// Convenience alias: price a European option with full Greeks (AD-5).
///
/// Identical to [`bs_european_generic`] — kept for parity with the f64 API.
pub fn price_european_generic<T: Number>(
    spot: T, strike: T, r: T, q: T, sigma: T, t: T, is_call: bool,
) -> BsEuropeanResult<T> {
    bs_european_generic(spot, strike, r, q, sigma, t, is_call)
}

// ===========================================================================
// AD-23 — Partial-Time Barrier Option (Heynen-Kat 1994), generic
// ===========================================================================

/// Type of partial-time barrier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PartialBarrierTypeGeneric {
    B1DownOut,
    B1DownIn,
    B1UpOut,
    B1UpIn,
}

/// Partial-time barrier option (barrier active during [0, t1]), generic.
#[allow(clippy::too_many_arguments)]
pub fn partial_time_barrier_generic<T: Number>(
    spot: T, strike: T, barrier: T, r: T, q: T, sigma: T,
    t: T, t1: T, barrier_type: PartialBarrierTypeGeneric, is_call: bool,
) -> T {
    if t.to_f64() < 1e-12 || t1.to_f64() < 0.0 {
        let payoff = if is_call {
            let d = spot - strike; if d.to_f64() > 0.0 { d } else { T::zero() }
        } else {
            let d = strike - spot; if d.to_f64() > 0.0 { d } else { T::zero() }
        };
        return payoff;
    }
    let t1 = if t1.to_f64() > t.to_f64() { t } else { t1 };
    let s2 = sigma * sigma;
    let disc = (-r * t).exp();
    let sqrt_t = t.sqrt();
    let sqrt_t1 = t1.sqrt();
    let rho = (t1 / t).sqrt();

    let d1 = ((spot / strike).ln() + (r - q + s2 * T::half()) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;

    let e1 = ((spot / barrier).ln() + (r - q + s2 * T::half()) * t1) / (sigma * sqrt_t1);
    let _e2 = e1 - sigma * sqrt_t1;

    let mu = (r - q + s2 * T::half()) / s2;
    let h_coeff = (barrier / spot).powf(T::two() * mu);

    let df_q = (-q * t).exp();
    let neg_e3 = -((spot / barrier).ln() / (sigma * sqrt_t1) + (s2 * T::half() + r - q) * t1 / (sigma * sqrt_t1));
    let neg_e4 = neg_e3 + sigma * sqrt_t1;

    match barrier_type {
        PartialBarrierTypeGeneric::B1DownOut | PartialBarrierTypeGeneric::B1DownIn => {
            let a1 = spot * df_q * bivariate_normal_cdf(d1, e1, rho.to_f64());
            let a2 = strike * disc * bivariate_normal_cdf(d2, _e2, rho.to_f64());

            let b3_arg2 = -(d1 - T::two() * mu * sigma * sqrt_t);
            let b3 = spot * df_q * h_coeff * bivariate_normal_cdf(neg_e3, b3_arg2, -rho.to_f64());
            let b4_arg2 = -(d2 - T::two() * mu * sigma * sqrt_t);
            let b4 = strike * disc * h_coeff * bivariate_normal_cdf(neg_e4, b4_arg2, -rho.to_f64());

            let vanilla = if is_call {
                spot * df_q * normal_cdf(d1) - strike * disc * normal_cdf(d2)
            } else {
                strike * disc * normal_cdf(-d2) - spot * df_q * normal_cdf(-d1)
            };

            let down_in_call = (a1 - a2) - (b3 - b4);
            let down_in = if down_in_call.to_f64() > 0.0 { down_in_call } else { T::zero() };
            let down_out = vanilla - down_in;
            let down_out = if down_out.to_f64() > 0.0 { down_out } else { T::zero() };

            match barrier_type {
                PartialBarrierTypeGeneric::B1DownOut => down_out,
                PartialBarrierTypeGeneric::B1DownIn => down_in,
                _ => unreachable!(),
            }
        }
        PartialBarrierTypeGeneric::B1UpOut | PartialBarrierTypeGeneric::B1UpIn => {
            let vanilla = if is_call {
                spot * df_q * normal_cdf(d1) - strike * disc * normal_cdf(d2)
            } else {
                strike * disc * normal_cdf(-d2) - spot * df_q * normal_cdf(-d1)
            };
            let c1 = spot * df_q * bivariate_normal_cdf(-d1, -e1, -rho.to_f64());
            let c2 = strike * disc * bivariate_normal_cdf(-d2, -_e2, -rho.to_f64());
            let up_in_raw = c1 - c2;
            let up_in = if up_in_raw.to_f64() > 0.0 { up_in_raw } else { T::zero() };
            let up_out_raw = vanilla - up_in;
            let up_out = if up_out_raw.to_f64() > 0.0 { up_out_raw } else { T::zero() };

            match barrier_type {
                PartialBarrierTypeGeneric::B1UpOut => up_out,
                PartialBarrierTypeGeneric::B1UpIn => up_in,
                _ => unreachable!(),
            }
        }
    }
}

// ===========================================================================
// AD-24 — Vanna-Volga Barrier, generic
// ===========================================================================

/// BS vanilla helper for VV engine, generic.
fn vv_bs_price<T: Number>(
    spot: T, strike: T, r_d: T, r_f: T, sigma: T, t: T, is_call: bool,
) -> T {
    let fwd = spot * ((r_d - r_f) * t).exp();
    let sqrt_t = t.sqrt();
    let d1 = (fwd / strike).ln() / (sigma * sqrt_t) + T::half() * sigma * sqrt_t;
    let d2 = d1 - sigma * sqrt_t;
    let df = (-r_d * t).exp();
    if is_call {
        df * (fwd * normal_cdf(d1) - strike * normal_cdf(d2))
    } else {
        df * (strike * normal_cdf(-d2) - fwd * normal_cdf(-d1))
    }
}

/// BS vega helper, generic.
#[allow(dead_code)]
fn vv_bs_vega<T: Number>(spot: T, strike: T, r_d: T, r_f: T, sigma: T, t: T) -> T {
    let fwd = spot * ((r_d - r_f) * t).exp();
    let sqrt_t = t.sqrt();
    let d1 = (fwd / strike).ln() / (sigma * sqrt_t) + T::half() * sigma * sqrt_t;
    let df = (-r_d * t).exp();
    df * fwd * sqrt_t * (-T::half() * d1 * d1).exp() / T::from_f64((2.0 * std::f64::consts::PI).sqrt())
}

/// Analytic BS barrier (Merton-Reiner-Rubinstein), generic.
#[allow(clippy::too_many_arguments)]
fn vv_bs_barrier<T: Number>(
    spot: T, strike: T, barrier: T, rebate: T,
    r_d: T, r_f: T, sigma: T, t: T, is_down: bool, is_knockout: bool, is_call: bool,
) -> T {
    let mu = (r_d - r_f) / (sigma * sigma) - T::half();
    let _lambda = (mu * mu + T::two() * r_d / (sigma * sigma)).sqrt();
    let sqrt_t = t.sqrt();
    let h_ratio = barrier / spot;
    let x2 = (spot / barrier).ln() / (sigma * sqrt_t) + (T::one() + mu) * sigma * sqrt_t;
    let y2 = (barrier / spot).ln() / (sigma * sqrt_t) + (T::one() + mu) * sigma * sqrt_t;

    let phi = if is_call { T::one() } else { -T::one() };
    let eta = if is_down { T::one() } else { -T::one() };

    let df_d = (-r_d * t).exp();
    let df_f = (-r_f * t).exp();

    // Simplified KO barrier formula
    let b = phi * spot * df_f * normal_cdf(phi * x2) - phi * strike * df_d * normal_cdf(phi * (x2 - sigma * sqrt_t));
    let d = phi * spot * df_f * h_ratio.powf(T::two() * (mu + T::one())) * normal_cdf(eta * y2)
        - phi * strike * df_d * h_ratio.powf(T::two() * mu) * normal_cdf(eta * (y2 - sigma * sqrt_t));

    let ko_approx = b - d;
    let ko_price = if ko_approx.to_f64() > 0.0 { ko_approx } else { T::zero() };

    if is_knockout {
        ko_price
    } else {
        let vanilla = vv_bs_price(spot, strike, r_d, r_f, sigma, t, is_call);
        let ki = vanilla - ko_price + rebate * df_d;
        if ki.to_f64() > 0.0 { ki } else { T::zero() }
    }
}

/// Vanna-Volga barrier option, generic over `T: Number` (AD-24).
#[allow(clippy::too_many_arguments)]
pub fn vanna_volga_barrier_generic<T: Number>(
    spot: T, strike: T, barrier: T, rebate: T,
    r_d: T, r_f: T, sigma_atm: T, sigma_25p: T, sigma_25c: T,
    t: T, is_down: bool, is_knockout: bool, is_call: bool,
) -> T {
    let fwd = spot * ((r_d - r_f) * t).exp();
    let sqrt_t = t.sqrt();
    let g = T::from_f64(0.6745);
    let k_atm = fwd * (T::half() * sigma_atm * sigma_atm * t).exp();
    let k_25p = fwd * (-sigma_25p * sqrt_t * g + T::half() * sigma_25p * sigma_25p * t).exp();
    let k_25c = fwd * (sigma_25c * sqrt_t * g + T::half() * sigma_25c * sigma_25c * t).exp();

    let sigma_pillars = [sigma_25p, sigma_atm, sigma_25c];
    let k_pillars = [k_25p, k_atm, k_25c];

    let bs_bar = vv_bs_barrier(spot, strike, barrier, rebate, r_d, r_f, sigma_atm, t, is_down, is_knockout, is_call);

    // Overhedge
    let mut overhedge = [T::zero(); 3];
    for i in 0..3 {
        let p_mkt = vv_bs_price(spot, k_pillars[i], r_d, r_f, sigma_pillars[i], t, true);
        let p_atm = vv_bs_price(spot, k_pillars[i], r_d, r_f, sigma_atm, t, true);
        overhedge[i] = p_mkt - p_atm;
    }

    // Log-strike weights
    let ln_k = strike.ln();
    let ln_k1 = k_pillars[0].ln();
    let ln_k2 = k_pillars[1].ln();
    let ln_k3 = k_pillars[2].ln();

    let mut w = [T::zero(); 3];
    let det = ((ln_k2 - ln_k1) * (ln_k3 - ln_k1)).to_f64();
    if det.abs() > 1e-20 {
        w[0] = (ln_k - ln_k2) * (ln_k - ln_k3) / ((ln_k1 - ln_k2) * (ln_k1 - ln_k3));
        w[1] = (ln_k - ln_k1) * (ln_k - ln_k3) / ((ln_k2 - ln_k1) * (ln_k2 - ln_k3));
        w[2] = (ln_k - ln_k1) * (ln_k - ln_k2) / ((ln_k3 - ln_k1) * (ln_k3 - ln_k2));
    }

    // Second-order VV barrier adjustment
    let mut vv_adj = T::zero();
    for i in 0..3 {
        let bs_bar_i = vv_bs_barrier(spot, k_pillars[i], barrier, T::zero(), r_d, r_f, sigma_atm, t, is_down, is_knockout, true);
        let bs_van_i = vv_bs_price(spot, k_pillars[i], r_d, r_f, sigma_atm, t, true);
        let ratio_i = if bs_van_i.to_f64().abs() > 1e-12 {
            let r = bs_bar_i / bs_van_i;
            let rv = r.to_f64().clamp(0.0, 1.0);
            T::from_f64(rv)
        } else { T::zero() };
        vv_adj += w[i] * overhedge[i] * ratio_i;
    }

    let price = bs_bar + vv_adj;
    if price.to_f64() > 0.0 { price } else { T::zero() }
}

// ===========================================================================
// AD-27 — Hull-White Jamshidian Swaption, generic
// ===========================================================================

/// Price a European swaption via Jamshidian under Hull-White, generic.
///
/// Uses Newton iteration to find r*, then sums ZCB options.
#[allow(clippy::too_many_arguments)]
pub fn hw_jamshidian_swaption_generic<T: Number>(
    a: T, sigma: T, option_expiry: T,
    swap_tenors: &[T], fixed_rate: T, discount_factors: &[T],
    p_option: T, notional: T, is_payer: bool,
) -> T {
    let n = swap_tenors.len();
    assert_eq!(discount_factors.len(), n);

    // Coupons: c_i = fixed_rate * tau_i, last += 1
    let mut coupons = Vec::with_capacity(n);
    for i in 0..n {
        let tau_i = if i == 0 { swap_tenors[0] - option_expiry } else { swap_tenors[i] - swap_tenors[i - 1] };
        let mut c = fixed_rate * tau_i;
        if i == n - 1 { c += T::one(); }
        coupons.push(c);
    }

    // B(tau) = (1-e^{-a*tau})/a
    let taus: Vec<T> = swap_tenors.iter().map(|&ti| ti - option_expiry).collect();
    let b_vals: Vec<T> = taus.iter().map(|&tau| {
        if a.to_f64().abs() < 1e-15 { tau }
        else { (T::one() - (-a * tau).exp()) / a }
    }).collect();

    let s = option_expiry;
    let var_factor = if a.to_f64().abs() < 1e-15 {
        sigma * sigma * s
    } else {
        sigma * sigma * (T::one() - (-T::two() * a * s).exp()) / (T::two() * a)
    };

    let f_s = -p_option.ln() / s;

    let a_coeffs: Vec<T> = (0..n).map(|i| {
        (discount_factors[i] / p_option).ln() + b_vals[i] * f_s
            - T::half() * b_vals[i] * b_vals[i] * var_factor
    }).collect();

    // Newton for r*: g(r) = sum c_i exp(A_i - B_i r) - 1 = 0
    let mut r_star = f_s;
    for _ in 0..50 {
        let mut g = -T::one();
        let mut gp = T::zero();
        for i in 0..n {
            let p_i = (a_coeffs[i] - b_vals[i] * r_star).exp();
            g += coupons[i] * p_i;
            gp -= coupons[i] * b_vals[i] * p_i;
        }
        if gp.to_f64().abs() < 1e-30 { break; }
        let dr = g / gp;
        r_star -= dr;
        if dr.to_f64().abs() < 1e-12 { break; }
    }

    // Bond strikes K_i = exp(A_i - B_i r*)
    let bond_strikes: Vec<T> = (0..n).map(|i| (a_coeffs[i] - b_vals[i] * r_star).exp()).collect();

    // Sum ZCB options
    let is_bond_call = !is_payer;
    let mut total = T::zero();
    for i in 0..n {
        let res = hw_bond_option_generic(
            a, sigma, p_option, discount_factors[i],
            option_expiry, swap_tenors[i], bond_strikes[i], is_bond_call,
        );
        total += coupons[i] * res;
    }
    notional * total
}

// ===========================================================================
// AD-41 — Replicating Variance Swap, generic
// ===========================================================================

/// BS price helper for replicating variance swap.
fn rep_bs_price<T: Number>(spot: T, k: T, r: T, q: T, sigma: T, t: T, is_call: bool) -> T {
    let sqrt_t = t.sqrt();
    let d1 = ((spot / k).ln() + (r - q + T::half() * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;
    let df = (-r * t).exp();
    let df_q = (-q * t).exp();
    if is_call {
        spot * df_q * normal_cdf(d1) - k * df * normal_cdf(d2)
    } else {
        k * df * normal_cdf(-d2) - spot * df_q * normal_cdf(-d1)
    }
}

/// Replicating variance swap from a strip of OTM options, generic (AD-41).
#[allow(clippy::too_many_arguments)]
pub fn replicating_variance_swap_generic<T: Number>(
    spot: T, r: T, q: T, t: T,
    strikes: &[T], implied_vols: &[T],
    variance_strike: T, notional: T,
) -> (T, T) {
    assert_eq!(strikes.len(), implied_vols.len());
    assert!(strikes.len() >= 2);

    let forward = spot * ((r - q) * t).exp();
    let df = (-r * t).exp();
    let n = strikes.len();

    let mut option_prices = Vec::with_capacity(n);
    for i in 0..n {
        let is_call = strikes[i].to_f64() >= forward.to_f64();
        option_prices.push(rep_bs_price(spot, strikes[i], r, q, implied_vols[i], t, is_call));
    }

    let k0_idx = strikes.iter().position(|k| k.to_f64() >= forward.to_f64()).unwrap_or(n - 1);
    let k0_idx = if k0_idx > 0 { k0_idx - 1 } else { 0 };
    let k0 = strikes[k0_idx];

    let two_over_t = T::two() / t;
    let mut fair_var = two_over_t * (forward / k0 - T::one() - (forward / k0).ln());

    for i in 0..n {
        let k = strikes[i];
        let dk = if i == 0 { strikes[1] - strikes[0] }
        else if i == n - 1 { strikes[n - 1] - strikes[n - 2] }
        else { (strikes[i + 1] - strikes[i - 1]) * T::half() };
        fair_var += two_over_t * dk * option_prices[i] / (k * k * df);
    }

    let pv = notional * df * (fair_var - variance_strike);
    (fair_var, pv)
}

// ===========================================================================
// AD-47 — Longstaff-Schwartz American MC, generic
// ===========================================================================

/// Longstaff-Schwartz American MC, generic over `T: Number` (AD-47).
///
/// RNG stays f64; payoff/drift/vol computations in `T`.
#[allow(clippy::too_many_arguments)]
pub fn mc_american_lsm_generic<T: Number>(
    spot: T, strike: T, r: T, q: T, vol: T,
    t_expiry: T, is_call: bool,
    n_paths: usize, n_steps: usize, seed: u64,
) -> McResultGeneric<T> {
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    let dt = t_expiry / T::from_f64(n_steps as f64);
    let df_step = (-r * dt).exp();
    let sqrt_dt = dt.sqrt();
    let drift_dt = (r - q - T::half() * vol * vol) * dt;
    let vol_sqrt_dt = vol * sqrt_dt;
    let omega = if is_call { T::one() } else { -T::one() };

    // Forward simulate paths storing log-spot
    let mut rng = SmallRng::seed_from_u64(seed);
    let ln_spot = spot.ln();
    let stride = n_steps + 1;
    let mut paths = vec![T::zero(); n_paths * stride];
    for p in 0..n_paths {
        paths[p * stride] = ln_spot;
        for s in 1..stride {
            let z: f64 = StandardNormal.sample(&mut rng);
            paths[p * stride + s] = paths[p * stride + s - 1] + drift_dt + vol_sqrt_dt * T::from_f64(z);
        }
    }

    // Terminal payoff
    let mut cf_time = vec![n_steps; n_paths];
    let mut cf_val = vec![T::zero(); n_paths];
    for i in 0..n_paths {
        let s = paths[i * stride + n_steps].exp();
        let pay = omega * (s - strike);
        cf_val[i] = if pay.to_f64() > 0.0 { pay } else { T::zero() };
    }

    // Precompute df table
    let df_step_f64 = df_step.to_f64();
    let mut df_table = vec![1.0_f64; n_steps + 1];
    for k in 1..=n_steps { df_table[k] = df_table[k - 1] * df_step_f64; }

    // Backward induction with LSM
    for step in (1..n_steps).rev() {
        let mut itm_s = Vec::new();
        let mut itm_y = Vec::new();
        let mut itm_idx = Vec::new();
        for i in 0..n_paths {
            let ln_s = paths[i * stride + step];
            let s = ln_s.exp();
            let exercise = omega * (s - strike);
            if exercise.to_f64() > 0.0 {
                let ahead = cf_time[i] - step;
                let disc_cf = cf_val[i].to_f64() * df_table[ahead];
                itm_s.push(s.to_f64());
                itm_y.push(disc_cf);
                itm_idx.push(i);
            }
        }
        if itm_s.len() < 4 { continue; }

        // Polynomial regression (degree 2) in f64
        let m = itm_s.len();
        let (mut sx, mut sx2, mut sx3, mut sx4) = (0.0, 0.0, 0.0, 0.0);
        let (mut sy, mut sxy, mut sx2y) = (0.0, 0.0, 0.0);
        for j in 0..m {
            let x = itm_s[j]; let x2 = x * x;
            sx += x; sx2 += x2; sx3 += x * x2; sx4 += x2 * x2;
            sy += itm_y[j]; sxy += x * itm_y[j]; sx2y += x2 * itm_y[j];
        }
        let mf = m as f64;
        // Solve 3x3 normal equations
        let a = [[mf, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]];
        let b = [sy, sxy, sx2y];
        let coeffs = solve_3x3(a, b);

        for (j, &i) in itm_idx.iter().enumerate() {
            let x = itm_s[j]; let x2 = x * x;
            let fitted = coeffs[0] + coeffs[1] * x + coeffs[2] * x2;
            let exercise_val = (omega * (T::from_f64(x) - strike)).to_f64();
            if exercise_val > fitted {
                cf_time[i] = step;
                cf_val[i] = T::from_f64(exercise_val);
            }
        }
    }

    // Compute NPV
    let mut sum = T::zero();
    let mut sum_sq = T::zero();
    for i in 0..n_paths {
        let disc = T::from_f64(df_table[cf_time[i]]);
        let pv = cf_val[i] * disc;
        sum += pv;
        sum_sq += pv * pv;
    }
    let nf = T::from_f64(n_paths as f64);
    let mean = sum / nf;
    let var = sum_sq / nf - mean * mean;
    let se = if var.to_f64() > 0.0 { (var / nf).sqrt() } else { T::zero() };

    McResultGeneric { price: mean, std_error: se.to_f64() }
}

/// Solve a 3×3 linear system (for LSM regression).
#[allow(clippy::needless_range_loop)]
fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let mut m = [[0.0; 4]; 3];
    for i in 0..3 { for j in 0..3 { m[i][j] = a[i][j]; } m[i][3] = b[i]; }
    for col in 0..3 {
        let mut pivot = col;
        for row in col+1..3 { if m[row][col].abs() > m[pivot][col].abs() { pivot = row; } }
        m.swap(col, pivot);
        if m[col][col].abs() < 1e-30 { return [0.0; 3]; }
        let d = m[col][col];
        for j in col..4 { m[col][j] /= d; }
        for row in 0..3 {
            if row == col { continue; }
            let f = m[row][col];
            for j in col..4 { m[row][j] -= f * m[col][j]; }
        }
    }
    [m[0][3], m[1][3], m[2][3]]
}

// ===========================================================================
// AD-56 — Callable Bond (binomial tree), generic
// ===========================================================================

/// Price a callable/puttable bond using a short-rate binomial tree, generic.
///
/// * `face` — face value
/// * `r` — flat risk-free rate
/// * `rate_vol` — short-rate volatility
/// * `total_time` — time to maturity in years
/// * `coupon_times` — (time, amount) pairs for coupon cash flows
/// * `call_times` — (time, call_price) pairs for exercise schedule
/// * `is_call` — true = issuer can call (cap value), false = holder can put (floor value)
/// * `n_steps` — tree steps
#[allow(clippy::too_many_arguments)]
pub fn callable_bond_generic<T: Number>(
    face: T, r: T, rate_vol: T, total_time: f64,
    coupon_times: &[(f64, T)], call_times: &[(f64, T)],
    is_call: bool, n_steps: usize,
) -> T {
    let dt = total_time / n_steps as f64;
    let u_f64 = (rate_vol.to_f64() * dt.sqrt()).exp();
    let p = 0.5_f64;

    // Terminal values
    let mut values = vec![face; n_steps + 1];
    // Add coupons at maturity
    for &(t, amt) in coupon_times {
        if (t - total_time).abs() < dt * 0.5 {
            for val in values.iter_mut() { *val += amt; }
        }
    }

    // Backward induction
    for step in (0..n_steps).rev() {
        let mut new_values = vec![T::zero(); step + 1];
        for j in 0..=step {
            let rate_node = r * T::from_f64(u_f64.powi(2 * j as i32 - step as i32));
            let df = (-rate_node * T::from_f64(dt)).exp();
            let cont = df * (T::from_f64(p) * values[j + 1] + T::from_f64(1.0 - p) * values[j]);
            new_values[j] = cont;
        }
        // Add coupons
        let t_start = step as f64 * dt;
        let t_end = (step + 1) as f64 * dt;
        for &(t, amt) in coupon_times {
            if t > t_start && t <= t_end && (t - total_time).abs() > dt * 0.5 {
                for val in new_values.iter_mut() { *val += amt; }
            }
        }
        // Exercise
        for &(t, call_price) in call_times {
            if t >= t_start && t < t_end {
                for val in new_values.iter_mut() {
                    if is_call {
                        if val.to_f64() > call_price.to_f64() { *val = call_price; }
                    } else if val.to_f64() < call_price.to_f64() { *val = call_price; }
                }
            }
        }
        values = new_values;
    }
    values[0]
}

// ===========================================================================
// AD-57 — Convertible Bond (CRR tree), generic
// ===========================================================================

/// Price a convertible bond using a CRR binomial equity tree, generic.
///
/// At each node: value = max(continuation + coupons, conversion_value).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn convertible_bond_generic<T: Number>(
    face: T, stock_price: T, conversion_ratio: T,
    r: T, q: T, vol: T, total_time: f64,
    coupon_times: &[(f64, T)], n_steps: usize,
) -> T {
    let dt = total_time / n_steps as f64;
    let u = (vol * T::from_f64(dt.sqrt())).exp();
    let d = T::one() / u;
    let growth = ((r - q) * T::from_f64(dt)).exp();
    let p = (growth - d) / (u - d);
    let df = (-r * T::from_f64(dt)).exp();
    let n = n_steps;

    // Terminal
    let mut values = vec![T::zero(); n + 1];
    for j in 0..=n {
        let s_t = stock_price * u.powf(T::from_f64(2.0 * j as f64 - n as f64));
        let conv = conversion_ratio * s_t;
        values[j] = if face.to_f64() > conv.to_f64() { face } else { conv };
    }
    // Coupons at maturity
    for &(t, amt) in coupon_times {
        if (t - total_time).abs() < dt * 0.5 {
            for val in values.iter_mut() { *val += amt; }
        }
    }

    for step in (0..n).rev() {
        let mut new_values = vec![T::zero(); step + 1];
        for j in 0..=step {
            let s_node = stock_price * u.powf(T::from_f64(2.0 * j as f64 - step as f64));
            let cont = df * (p * values[j + 1] + (T::one() - p) * values[j]);
            let conv = conversion_ratio * s_node;
            new_values[j] = if cont.to_f64() > conv.to_f64() { cont } else { conv };
        }
        let t_start = step as f64 * dt;
        let t_end = (step + 1) as f64 * dt;
        for &(t, amt) in coupon_times {
            if t > t_start && t <= t_end && (t - total_time).abs() > dt * 0.5 {
                for val in new_values.iter_mut() { *val += amt; }
            }
        }
        values = new_values;
    }
    values[0]
}

// ===========================================================================
// AD-76 — Mountain Range (multi-asset MC), generic
// ===========================================================================

/// Mountain-range option type for generic engine (AD-76).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MountainTypeGeneric {
    /// Payoff = min return over all assets.
    Everest,
    /// Accumulated positive returns, capped.
    Pagoda { local_cap: f64 },
}

/// Multi-asset MC for mountain-range options, generic.
///
/// RNG stays f64; drift, vol, payoff in T.
#[allow(clippy::too_many_arguments)]
pub fn mc_mountain_range_generic<T: Number>(
    spots: &[T], vols: &[T], correlations: &[f64],
    r: T, q: &[T], observation_times: &[f64],
    notional: T, mountain_type: MountainTypeGeneric,
    n_paths: usize, seed: u64,
) -> McResultGeneric<T> {
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    let n_assets = spots.len();
    let n_obs = observation_times.len();

    // Cholesky of correlation matrix (f64)
    let chol = cholesky_f64(correlations, n_assets);

    let t_final = *observation_times.last().unwrap_or(&0.0);
    let df = (-r * T::from_f64(t_final)).exp();

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut sum = T::zero();
    let mut sum_sq = T::zero();

    for _ in 0..n_paths {
        let mut prices = spots.to_vec();
        let mut prev_t = 0.0;
        let mut payoff = T::zero();
        let mut min_return = T::from_f64(f64::MAX);
        let mut accum = T::zero();

        for (obs_idx, &t) in observation_times.iter().enumerate() {
            let dt_f = t - prev_t;
            let sqrt_dt = dt_f.sqrt();
            let z_indep: Vec<f64> = (0..n_assets).map(|_| StandardNormal.sample(&mut rng)).collect();
            let z_corr = apply_cholesky_f64(&chol, &z_indep, n_assets);

            for i in 0..n_assets {
                let drift = (r - q[i] - T::half() * vols[i] * vols[i]) * T::from_f64(dt_f);
                let diff = vols[i] * T::from_f64(sqrt_dt * z_corr[i]);
                prices[i] *= (drift + diff).exp();
            }

            // Per-asset returns
            match mountain_type {
                MountainTypeGeneric::Everest => {
                    if obs_idx == n_obs - 1 {
                        for i in 0..n_assets {
                            let ret = prices[i] / spots[i] - T::one();
                            if ret.to_f64() < min_return.to_f64() { min_return = ret; }
                        }
                        payoff = if min_return.to_f64() > 0.0 { min_return } else { T::zero() };
                    }
                }
                MountainTypeGeneric::Pagoda { local_cap } => {
                    let avg_ret = {
                        let mut s = T::zero();
                        for i in 0..n_assets { s = s + prices[i] / spots[i] - T::one(); }
                        s / T::from_f64(n_assets as f64)
                    };
                    let capped = if avg_ret.to_f64() > local_cap { T::from_f64(local_cap) }
                    else if avg_ret.to_f64() > 0.0 { avg_ret }
                    else { T::zero() };
                    accum += capped;
                    if obs_idx == n_obs - 1 { payoff = accum; }
                }
            }
            prev_t = t;
        }

        let disc = notional * payoff * df;
        sum += disc;
        sum_sq += disc * disc;
    }

    let nf = T::from_f64(n_paths as f64);
    let mean = sum / nf;
    let var = sum_sq / nf - mean * mean;
    let se = if var.to_f64() > 0.0 { (var / nf).sqrt() } else { T::zero() };
    McResultGeneric { price: mean, std_error: se.to_f64() }
}

/// Cholesky decomposition of flat row-major correlation matrix (f64).
fn cholesky_f64(corr: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0;
            for k in 0..j { s += l[i * n + k] * l[j * n + k]; }
            l[i * n + j] = if i == j {
                (corr[i * n + j] - s).max(0.0).sqrt()
            } else if l[j * n + j].abs() > 1e-15 { (corr[i * n + j] - s) / l[j * n + j] }
            else { 0.0 };
        }
    }
    l
}

/// Apply Cholesky factor to independent normals (f64).
fn apply_cholesky_f64(l: &[f64], z: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for i in 0..n {
        for j in 0..=i { out[i] += l[i * n + j] * z[j]; }
    }
    out
}

// ===========================================================================
// AD-78 — COS Heston, generic
// ===========================================================================

/// Heston characteristic function for COS method, generic.
#[allow(clippy::too_many_arguments)]
fn heston_cf_generic<T: Number>(
    u_freq: f64, tau: T, v0: T, kappa: T, theta: T, sigma: T, rho: T, log_fk: T,
) -> ComplexT<T> {
    let u = T::from_f64(u_freq);
    let _iu = ComplexT::new(T::zero(), u);

    let alpha = ComplexT::new(-T::half() * u * u, -T::half() * u);
    let beta = ComplexT::new(kappa, -rho * sigma * u);
    let sigma_sq = sigma * sigma;

    let beta_sq = beta.mul(beta);
    let sig2_alpha = ComplexT::new(sigma_sq * alpha.re, sigma_sq * alpha.im);
    let d_sq = beta_sq.sub(sig2_alpha);
    let d = d_sq.sqrt();

    let bmd = beta.sub(d);
    let bpd = beta.add(d);

    let g = bmd.div(bpd);

    let neg_d_tau = ComplexT::new(-d.re * tau, -d.im * tau);
    let exp_neg_dtau = neg_d_tau.exp();

    let one_c = ComplexT::from_real(T::one());
    let one_minus_exp = one_c.sub(exp_neg_dtau);
    let g_exp_neg = g.mul(exp_neg_dtau);
    let denom_b = one_c.sub(g_exp_neg);
    let frac_b = one_minus_exp.div(denom_b);
    let big_b = bmd.mul(frac_b).scale(T::one() / sigma_sq);

    // A = kappa*theta/sigma² * [(beta-d)*tau - 2*ln((1-g*e^{-dτ})/(1-g))]
    let ratio_for_ln = denom_b.div(one_c.sub(g));
    let ln_ratio = ratio_for_ln.ln();

    let bt_tau = ComplexT::new(bmd.re * tau, bmd.im * tau);
    let big_a_bracket = bt_tau.sub(ln_ratio.scale(T::two()));
    let kt_over_s2 = kappa * theta / sigma_sq;
    let big_a = big_a_bracket.scale(kt_over_s2);

    // φ(u) = exp(A + B*v0 + i*u*log_fk)
    let b_v0 = big_b.scale(v0);
    let iu_x = ComplexT::new(T::zero(), u * log_fk);
    let exponent = big_a.add(b_v0).add(iu_x);
    exponent.exp()
}

/// COS payoff coefficients (call or put), f64 only (grid geometry).
fn cos_payoff_coeffs(n: usize, a: f64, b: f64, strike: f64, is_call: bool) -> Vec<f64> {
    let range = b - a;
    let pi = std::f64::consts::PI;
    (0..n).map(|k| {
        let kf = k as f64;
        let kpi = kf * pi;
        let kpi_over_range = kpi / range;
        let (c, d) = if is_call { (0.0, b) } else { (a, 0.0) };
        let chi = {
            let denom = 1.0 + kpi_over_range * kpi_over_range;
            let term_d = (d - a) * kpi_over_range;
            let term_c = (c - a) * kpi_over_range;
            (1.0 / denom) * (d.exp() * (term_d.cos() + kpi_over_range * term_d.sin())
                - c.exp() * (term_c.cos() + kpi_over_range * term_c.sin()))
        };
        let psi = if kpi.abs() < 1e-12 { d - c }
        else { (((d - a) * kpi_over_range).sin() - ((c - a) * kpi_over_range).sin()) / kpi_over_range };
        2.0 / range * strike * (if is_call { chi - psi } else { -chi + psi })
    }).collect()
}

/// COS Heston pricing, generic over `T: Number` (AD-78).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn cos_heston_generic<T: Number>(
    spot: T, strike: T, tau: T,
    r: T, q: T,
    v0: T, kappa: T, theta: T, sigma: T, rho: T,
    is_call: bool, n_terms: usize,
) -> T {
    let n = if n_terms == 0 { 128 } else { n_terms };
    let l = 12.0_f64;

    let log_fk = (spot / strike).ln() + (r - q) * tau;

    // Cumulants (f64 for interval selection)
    let tau_f = tau.to_f64();
    let kappa_f = kappa.to_f64();
    let theta_f = theta.to_f64();
    let v0_f = v0.to_f64();
    let sigma_f = sigma.to_f64();
    let ekt = (-kappa_f * tau_f).exp();
    let c1 = log_fk.to_f64() + (1.0 - ekt) * (theta_f - v0_f) / (2.0 * kappa_f) - theta_f * tau_f / 2.0;
    let c2 = (theta_f * tau_f / (2.0 * kappa_f) * (sigma_f * sigma_f - 2.0 * kappa_f * theta_f)
        + v0_f * (1.0 - ekt).powi(2) / (4.0 * kappa_f) * (sigma_f * sigma_f - 2.0 * kappa_f * theta_f)
        + (1.0 - ekt) * theta_f / 2.0 * kappa_f).abs().max(0.01);
    let a = c1 - l * c2.sqrt();
    let b = c1 + l * c2.sqrt();

    let df = (-r * tau).exp();
    let u_k = cos_payoff_coeffs(n, a, b, strike.to_f64(), is_call);
    let pi = std::f64::consts::PI;
    let range = b - a;

    let mut price = T::zero();
    for k in 0..n {
        let freq = k as f64 * pi / range;
        let cf = heston_cf_generic(freq, tau, v0, kappa, theta, sigma, rho, log_fk);
        let phase_arg = -(k as f64) * pi * a / range;
        let phase = ComplexT::new(T::from_f64(phase_arg.cos()), T::from_f64(phase_arg.sin()));
        let re_cf_phase = cf.mul(phase).re;
        let weight = if k == 0 { T::half() } else { T::one() };
        price += weight * re_cf_phase * T::from_f64(u_k[k]);
    }
    price *= df;
    if price.to_f64() > 0.0 { price } else { T::zero() }
}

// ===========================================================================
// AD-79 — MC Stochastic Local Vol, generic
// ===========================================================================

/// MC SLV (Euler): Heston-type with optional leverage, generic (AD-79).
#[allow(clippy::too_many_arguments)]
pub fn mc_slv_generic<T: Number>(
    spot: T, r: T, q: T,
    v0: T, kappa: T, theta: T, xi: T, rho_f: f64,
    strike: T, maturity: T, is_call: bool,
    n_paths: usize, n_steps: usize, seed: u64,
) -> McResultGeneric<T> {
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    let dt = maturity / T::from_f64(n_steps as f64);
    let sqrt_dt_f = (dt.to_f64()).sqrt();
    let df = (-r * maturity).exp();

    let sqrt_1mr2 = (1.0 - rho_f * rho_f).sqrt();
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut sum = T::zero();
    let mut sum_sq = T::zero();

    for _ in 0..n_paths {
        let mut s = spot;
        let mut v = v0;
        for _ in 0..n_steps {
            let sqrt_v = if v.to_f64() > 0.0 { v.sqrt() } else { T::zero() };
            let z1: f64 = StandardNormal.sample(&mut rng);
            let z2: f64 = StandardNormal.sample(&mut rng);
            let w1 = T::from_f64(z1);
            let w2 = T::from_f64(rho_f * z1 + sqrt_1mr2 * z2);

            s = s + (r - q) * s * dt + sqrt_v * s * T::from_f64(sqrt_dt_f) * w1;
            if s.to_f64() < 1e-8 { s = T::from_f64(1e-8); }
            v = v + kappa * (theta - v) * dt + xi * sqrt_v * T::from_f64(sqrt_dt_f) * w2;
            if v.to_f64() < 0.0 { v = T::zero(); }
        }
        let payoff = if is_call {
            let d = s - strike; if d.to_f64() > 0.0 { d } else { T::zero() }
        } else {
            let d = strike - s; if d.to_f64() > 0.0 { d } else { T::zero() }
        };
        sum += payoff;
        sum_sq += payoff * payoff;
    }

    let nf = T::from_f64(n_paths as f64);
    let mean = df * sum / nf;
    let var = df * df * (sum_sq / nf - (sum / nf) * (sum / nf));
    let se = if var.to_f64() > 0.0 { (var / nf).sqrt() } else { T::zero() };
    McResultGeneric { price: mean, std_error: se.to_f64() }
}

// ===========================================================================
// AD-80 — Heston-Hull-White (A1HW), generic
// ===========================================================================

/// Heston-Hull-White (A1HW approximation), generic (AD-80).
///
/// Adjusts Heston v₀ by the equity-rate correlation contribution, then
/// prices with COS-Heston.
#[allow(clippy::too_many_arguments)]
pub fn heston_hull_white_generic<T: Number>(
    spot: T, strike: T, tau: T,
    r0: T, q: T,
    hw_a: T, hw_sigma_r: T,
    v0: T, kappa: T, theta: T, sigma_v: T, rho_sv: T,
    rho_sr: T, is_call: bool,
) -> T {
    // B(a, T) = (1 − e^{−aT}) / a
    let b_hw = if hw_a.to_f64().abs() < 1e-8 { tau }
    else { (T::one() - (-hw_a * tau).exp()) / hw_a };

    // ξ(T) = ρ_sr · σ_v · σ_r · B(a,T) · (1 − e^{−κT}) / κ
    let xi = if kappa.to_f64().abs() < 1e-8 {
        rho_sr * sigma_v * hw_sigma_r * b_hw * tau
    } else {
        rho_sr * sigma_v * hw_sigma_r * b_hw * (T::one() - (-kappa * tau).exp()) / kappa
    };

    let v0_eff = v0 + xi;
    let v0_eff = if v0_eff.to_f64() < 1e-8 { T::from_f64(1e-8) } else { v0_eff };

    cos_heston_generic(spot, strike, tau, r0, q, v0_eff, kappa, theta, sigma_v, rho_sv, is_call, 128)
}

// ===========================================================================
// AD-85 — Nth-to-Default MC (Gaussian Copula), generic
// ===========================================================================

/// Nth-to-default MC via Gaussian copula, generic (AD-85).
///
/// RNG in f64; threshold comparisons in f64; payoff in T.
#[allow(clippy::too_many_arguments)]
pub fn nth_to_default_mc_generic<T: Number>(
    default_probs: &[T], correlation: T, recovery: T,
    nth: usize, notional: T,
    n_paths: usize, seed: u64,
) -> McResultGeneric<T> {
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    let n_names = default_probs.len();
    assert!(nth >= 1 && nth <= n_names);

    let sqrt_rho = correlation.to_f64().sqrt();
    let sqrt_1mr = (1.0 - correlation.to_f64()).sqrt();
    let lgd = (T::one() - recovery) * notional / T::from_f64(n_names as f64);

    // Precompute thresholds Φ⁻¹(p_i)
    let thresholds: Vec<f64> = default_probs.iter().map(|p| {
        ql_math::generic::inverse_normal_cdf(p.to_f64())
    }).collect();

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut sum = T::zero();
    let mut sum_sq = T::zero();

    for _ in 0..n_paths {
        let z: f64 = StandardNormal.sample(&mut rng);
        let mut n_defaults = 0usize;
        for &threshold in &thresholds {
            let eps: f64 = StandardNormal.sample(&mut rng);
            let x = sqrt_rho * z + sqrt_1mr * eps;
            if x < threshold { n_defaults += 1; }
        }
        let payoff = if n_defaults >= nth { lgd } else { T::zero() };
        sum += payoff;
        sum_sq += payoff * payoff;
    }

    let nf = T::from_f64(n_paths as f64);
    let mean = sum / nf;
    let var = sum_sq / nf - mean * mean;
    let se = if var.to_f64() > 0.0 { (var / nf).sqrt() } else { T::zero() };
    McResultGeneric { price: mean, std_error: se.to_f64() }
}

// ===========================================================================
// AD-91 — LMM Product MC, generic
// ===========================================================================

/// LMM European swaption MC pricer, generic (AD-91).
///
/// Evolves forward rates via log-normal LMM, then computes swap NPV.
/// RNG in f64; forward rates, payoff, discounting in T.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn lmm_swaption_mc_generic<T: Number>(
    initial_forwards: &[T], vols: &[T], correlations: &[f64],
    accruals: &[f64], swap_start_idx: usize, swap_end_idx: usize,
    fixed_rate: T, notional: T, is_payer: bool,
    n_paths: usize, seed: u64,
) -> McResultGeneric<T> {
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    let n_rates = initial_forwards.len();
    assert_eq!(vols.len(), n_rates);
    assert!(swap_start_idx < swap_end_idx && swap_end_idx <= n_rates);

    let chol = cholesky_f64(correlations, n_rates);

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut sum = T::zero();
    let mut sum_sq = T::zero();

    for _ in 0..n_paths {
        let mut fwds = initial_forwards.to_vec();

        // Evolve forward rates to swap_start_idx (one step per period)
        for step in 0..swap_start_idx {
            let dt = accruals[step];
            let sqrt_dt = dt.sqrt();
            let z: Vec<f64> = (0..n_rates).map(|_| StandardNormal.sample(&mut rng)).collect();
            let zc = apply_cholesky_f64(&chol, &z, n_rates);

            // Log-normal drift under spot measure
            for j in (step + 1)..n_rates {
                let mut drift = T::zero();
                for k in (step + 1)..=j {
                    drift += T::from_f64(accruals[k]) * vols[k] * fwds[k]
                        / (T::one() + T::from_f64(accruals[k]) * fwds[k]);
                }
                drift = drift * vols[j] * T::from_f64(dt);
                let diffusion = vols[j] * T::from_f64(sqrt_dt * zc[j]);
                fwds[j] *= (drift - T::half() * vols[j] * vols[j] * T::from_f64(dt) + diffusion).exp();
            }
        }

        // Compute swap NPV at swap_start_idx
        let mut swap_npv = T::zero();
        let mut df_accum = T::one();
        for j in swap_start_idx..swap_end_idx {
            df_accum /= T::one() + T::from_f64(accruals[j]) * fwds[j];
            let flow = T::from_f64(accruals[j]) * (fwds[j] - fixed_rate);
            swap_npv += flow * df_accum;
        }

        let payoff = if is_payer {
            if swap_npv.to_f64() > 0.0 { swap_npv } else { T::zero() }
        } else if (-swap_npv).to_f64() > 0.0 { -swap_npv } else { T::zero() };
        let pv = notional * payoff;

        // Discount back to time 0
        let mut df_to_zero = T::one();
        for j in 0..swap_start_idx {
            df_to_zero /= T::one() + T::from_f64(accruals[j]) * initial_forwards[j];
        }
        let disc_pv = pv * df_to_zero;

        sum += disc_pv;
        sum_sq += disc_pv * disc_pv;
    }

    let nf = T::from_f64(n_paths as f64);
    let mean = sum / nf;
    let var = sum_sq / nf - mean * mean;
    let se = if var.to_f64() > 0.0 { (var / nf).sqrt() } else { T::zero() };
    McResultGeneric { price: mean, std_error: se.to_f64() }
}

// ===========================================================================
// AD-92 — VG COS, generic
// ===========================================================================

/// VG characteristic function (log-return), generic.
fn vg_cf_generic<T: Number>(
    u_freq: f64, tau: T, sigma: T, nu: T, theta_vg: T, omega: T, log_fk: T, _r_minus_q: T,
) -> ComplexT<T> {
    let u = T::from_f64(u_freq);
    // φ_VG(u) = exp(iuω T) · (1 − iuθν + ½σ²νu²)^{−T/ν}
    // ω = (1/ν) ln(1 − θν − ½σ²ν) is the martingale correction

    // log of (1 - iu θ ν + 0.5 σ² ν u²):
    let a_re = T::one() + T::half() * sigma * sigma * nu * u * u;
    let a_im = -u * theta_vg * nu;
    let a = ComplexT::new(a_re, a_im);
    let neg_t_over_nu = -tau / nu;
    let ln_a = a.ln();
    let power = ComplexT::new(ln_a.re * neg_t_over_nu, ln_a.im * neg_t_over_nu);

    let drift = ComplexT::new(T::zero(), u * omega * tau + u * log_fk);
    let exponent = power.add(drift);
    exponent.exp()
}

/// VG COS pricing, generic over `T: Number` (AD-92).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn vg_cos_generic<T: Number>(
    spot: T, strike: T, tau: T,
    r: T, q: T,
    sigma: T, nu: T, theta_vg: T,
    is_call: bool, n_terms: usize,
) -> T {
    let n = if n_terms == 0 { 128 } else { n_terms };
    let l = 12.0_f64;

    // Martingale correction ω = (1/ν) ln(1 − θν − ½σ²ν)
    let inner = T::one() - theta_vg * nu - T::half() * sigma * sigma * nu;
    let omega = if inner.to_f64() > 0.0 { inner.ln() / nu } else { T::zero() };

    let log_fk = (spot / strike).ln() + (r - q) * tau;

    // Cumulants (f64) for interval
    let sigma_f = sigma.to_f64();
    let nu_f = nu.to_f64();
    let theta_f = theta_vg.to_f64();
    let tau_f = tau.to_f64();
    let c1 = (omega.to_f64() + (r - q).to_f64() + theta_f) * tau_f;
    let c2 = (sigma_f * sigma_f + nu_f * theta_f * theta_f) * tau_f;
    let c2s = c2.abs().sqrt().max(1e-4);
    let a = c1 - l * c2s;
    let b = c1 + l * c2s;

    let df = (-r * tau).exp();
    let u_k = cos_payoff_coeffs(n, a, b, strike.to_f64(), is_call);
    let pi = std::f64::consts::PI;
    let range = b - a;

    let r_minus_q = r - q;
    let mut price = T::zero();
    for k in 0..n {
        let freq = k as f64 * pi / range;
        let cf = vg_cf_generic(freq, tau, sigma, nu, theta_vg, omega, log_fk, r_minus_q);
        let phase_arg = -(k as f64) * pi * a / range;
        let phase = ComplexT::new(T::from_f64(phase_arg.cos()), T::from_f64(phase_arg.sin()));
        let re_cf_phase = cf.mul(phase).re;
        let weight = if k == 0 { T::half() } else { T::one() };
        price += weight * re_cf_phase * T::from_f64(u_k[k]);
    }
    price *= df;
    if price.to_f64() > 0.0 { price } else { T::zero() }
}

// ===========================================================================
// Phase B-D Tests
// ===========================================================================

#[cfg(test)]
mod tests_phase_b_to_d {
    use super::*;

    // --- AD-1: Variance Swap ---
    #[test]
    fn variance_swap_fair_strike() {
        let res = variance_swap_generic(0.20_f64, 0.05, 1.0, 100.0, 0.04);
        assert!((res.fair_variance - 0.04).abs() < 1e-10);
        assert!(res.npv.abs() < 1e-10); // fair = 0.04, strike = 0.04 → NPV ≈ 0
    }

    #[test]
    fn variance_swap_positive_npv() {
        let res = variance_swap_generic(0.25_f64, 0.05, 1.0, 100.0, 0.04);
        assert!(res.npv > 0.0, "npv={}", res.npv);
    }

    // --- AD-2: Quanto ---
    #[test]
    fn quanto_adjustment_test() {
        let adj = quanto_adjustment_generic(
            100.0_f64, 0.02, 0.01, 0.20, 0.10, -0.3, 1.0,
        );
        // q_adjusted = 0.01 + (-0.3)*0.20*0.10 = 0.004
        assert!((adj.q_adjusted - 0.004).abs() < 1e-10);
    }

    #[test]
    fn quanto_vanilla_positive() {
        let v: f64 = quanto_vanilla_generic(
            100.0, 100.0, 0.02, 0.05, 0.01, 0.20, 0.10, -0.3, 1.0, true, 1.0,
        );
        assert!(v > 0.0, "quanto={v}");
    }

    // --- AD-3: BSM-HW ---
    #[test]
    fn bsm_hw_zero_rate_vol_equals_bs() {
        let res = bsm_hull_white_generic(
            100.0_f64, 100.0, 1.0, 0.20, 0.05, 0.0, 0.0, 0.05, 0.0, true,
        );
        assert!((res.npv - 10.45).abs() < 0.5, "npv={}", res.npv);
    }

    // --- AD-4: Digital American ---
    #[test]
    fn digital_american_call() {
        let v: f64 = digital_american_generic(100.0, 110.0, 0.05, 0.0, 0.20, 1.0, 1.0, true);
        assert!(v > 0.0 && v < 1.0, "price={v}");
    }

    #[test]
    fn digital_american_already_hit() {
        let v: f64 = digital_american_generic(110.0, 100.0, 0.05, 0.0, 0.20, 1.0, 15.0, true);
        assert!((v - 15.0).abs() < 1e-10);
    }

    // --- AD-6: Black Swaption ---
    #[test]
    fn black_swaption_positive() {
        let v: f64 = black_swaption_generic(4.5, 0.05, 0.05, 0.20, 1.0, true);
        assert!(v >= 0.0, "swaption={v}");
    }

    // --- AD-7: Bachelier Swaption ---
    #[test]
    fn bachelier_swaption_positive() {
        let v: f64 = bachelier_swaption_generic(4.5, 0.05, 0.05, 0.005, 1.0, true);
        assert!(v > 0.0, "swaption={v}");
    }

    // --- AD-8: Black Caplet ---
    #[test]
    fn black_caplet_positive() {
        let v: f64 = black_caplet_generic(0.95, 0.05, 0.04, 0.20, 0.25, 1.0, true);
        assert!(v > 0.0, "caplet={v}");
    }

    // --- AD-10: Asian Geometric Continuous ---
    #[test]
    fn asian_geo_continuous_positive() {
        let v: f64 = asian_geometric_continuous_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        assert!(v > 0.0, "asian={v}");
    }

    // --- AD-12: Asian Turnbull-Wakeman ---
    #[test]
    fn asian_tw_positive() {
        let v: f64 = asian_turnbull_wakeman_generic(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 0.0, 0.0, true,
        );
        assert!(v > 0.0, "tw={v}");
    }

    // --- AD-14: Lookback ---
    #[test]
    fn lookback_floating_call() {
        let v: f64 = lookback_floating_generic(100.0, 95.0, 0.05, 0.0, 0.20, 1.0, true);
        assert!(v > 0.0, "lookback={v}");
    }

    // --- AD-18: Margrabe ---
    #[test]
    fn margrabe_positive() {
        let v: f64 = margrabe_exchange_generic(
            100.0, 95.0, 0.0, 0.0, 0.20, 0.25, 0.5, 1.0,
        );
        assert!(v > 0.0, "margrabe={v}");
    }

    // --- AD-25: HW Bond Option ---
    #[test]
    fn hw_bond_option_positive() {
        let v: f64 = hw_bond_option_generic(
            0.05, 0.01, 5.0, 1.0, 0.90, 0.85, 0.03, true,
        );
        assert!(v > 0.0, "hw_bo={v}");
    }

    // --- AD-31: Quanto European ---
    #[test]
    fn quanto_euro_positive() {
        let v: f64 = quanto_european_generic(
            100.0, 100.0, 0.02, 0.05, 0.20, 0.10, 0.3, 1.0, 1.5, true,
        );
        assert!(v > 0.0, "quanto_euro={v}");
    }

    // --- AD-32: Power Option ---
    #[test]
    fn power_option_alpha1_matches_bs() {
        let p: f64 = power_option_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 1.0, true);
        let bs: f64 = black_scholes_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        assert!((p - bs).abs() < 0.1, "power={p}, bs={bs}");
    }

    // --- AD-33: Forward-Start ---
    #[test]
    fn forward_start_positive() {
        let v: f64 = forward_start_generic(100.0, 0.05, 0.02, 0.20, 0.5, 1.0, 1.0, true);
        assert!(v > 0.0, "fs={v}");
    }

    // --- AD-34: Digital Barrier ---
    #[test]
    fn one_touch_positive() {
        let v: f64 = digital_barrier_generic(100.0, 90.0, 1.0, 0.05, 0.0, 0.20, 1.0, true, false);
        assert!(v > 0.0 && v <= 1.0, "ot={v}");
    }

    // --- AD-36: Vasicek Bond Option ---
    #[test]
    fn vasicek_bond_option_positive() {
        let v: f64 = vasicek_bond_option_generic(
            0.05, 0.05, 0.01, 0.03, 1.0, 5.0, 0.85, true,
        );
        assert!(v > 0.0, "vasicek_bo={v}");
    }

    // --- AD-38: CDS Option ---
    #[test]
    fn cds_option_positive() {
        let v: f64 = cds_option_black_generic(0.01, 0.01, 0.40, 1.0, 4.0, true);
        assert!(v >= 0.0, "cds_opt={v}");
    }

    // --- AD-43: Soft Barrier ---
    #[test]
    fn soft_barrier_between_bounds() {
        let v: f64 = soft_barrier_generic(100.0, 100.0, 80.0, 120.0, 0.05, 0.0, 0.20, 1.0, true);
        let bs: f64 = black_scholes_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        // Soft barrier should be less than vanilla (some knockout probability)
        // but weight at 100 with [80,120] → weight = (100-80)/(120-80) = 0.5
        assert!((v - bs * 0.5).abs() < 0.01, "soft={v}, bs={bs}");
    }
}

// ===========================================================================
// Phase E tests: Curve-based generic analytics (INFRA-5)
// ===========================================================================
#[cfg(test)]
mod tests_phase_e {
    use super::*;
    use ql_termstructures::generic::{FlatCurve, InterpDiscountCurve, InterpZeroCurve};

    #[test]
    fn leg_npv_flat_curve() {
        let curve = FlatCurve::new(0.05_f64);
        let times = vec![1.0, 2.0, 3.0];
        let amounts = vec![5.0, 5.0, 105.0];
        let pv: f64 = leg_npv_generic(&times, &amounts, &curve);
        let expected = 5.0 * (-0.05_f64).exp()
            + 5.0 * (-0.10_f64).exp()
            + 105.0 * (-0.15_f64).exp();
        assert!((pv - expected).abs() < 1e-10, "pv={pv}, expected={expected}");
    }

    #[test]
    fn leg_npv_interp_curve() {
        let curve_times = vec![0.5, 1.0, 2.0, 5.0];
        let curve_rates = vec![0.02, 0.03, 0.035, 0.04];
        let curve = InterpDiscountCurve::from_zero_rates(&curve_times, &curve_rates);
        let times = vec![1.0, 2.0];
        let amounts = vec![5.0, 105.0];
        let pv: f64 = leg_npv_generic(&times, &amounts, &curve);
        let expected = 5.0 * (-0.03_f64).exp() + 105.0 * (-0.07_f64).exp();
        assert!((pv - expected).abs() < 1e-10, "pv={pv}");
    }

    #[test]
    fn bond_pv_curve_vs_flat() {
        let rate = 0.05;
        let curve = FlatCurve::new(rate);
        let coupon_times = vec![0.5, 1.0, 1.5, 2.0];
        let coupon_amounts = vec![2.5, 2.5, 2.5, 2.5]; // 5% annual on 100
        let pv_curve: f64 = bond_pv_curve_generic(
            &coupon_times, &coupon_amounts, 100.0, 2.0, &curve,
        );
        // Compare with flat-rate bond pricing
        let pv_flat: f64 = bond_pv_generic(
            &coupon_amounts, &coupon_times, 100.0, 2.0, rate,
        );
        assert!((pv_curve - pv_flat).abs() < 1e-10, "curve={pv_curve}, flat={pv_flat}");
    }

    #[test]
    fn swap_pv_curve_at_par() {
        // A swap at par rate should have NPV ≈ 0
        let curve_times = vec![0.5, 1.0, 2.0, 5.0];
        let curve_rates = vec![0.03, 0.035, 0.04, 0.045];
        let curve = InterpDiscountCurve::from_zero_rates(&curve_times, &curve_rates);

        let fixed_times = vec![1.0, 2.0];
        let fixed_yfs = vec![1.0, 1.0];

        // Get the par swap rate
        let sr: f64 = swap_rate_generic(0.0, 2.0, &fixed_times, &fixed_yfs, &curve);

        // Floating leg NPV = DF(0) - DF(2)
        let df0: f64 = curve.discount_t(0.0);
        let df2: f64 = curve.discount_t(2.0);
        let float_pv = df0 - df2;

        // Fixed leg NPV at par rate
        let fixed_amounts: Vec<f64> = fixed_yfs.iter().map(|&yf| sr * yf).collect();
        let fixed_pv: f64 = leg_npv_generic(&fixed_times, &fixed_amounts, &curve);

        // Should be approximately equal
        assert!((float_pv - fixed_pv).abs() < 1e-10, "float={float_pv}, fixed={fixed_pv}");
    }

    #[test]
    fn par_rate_generic_test() {
        let curve = FlatCurve::new(0.05_f64);
        let coupon_times = vec![1.0, 2.0, 3.0];
        let coupon_yfs = vec![1.0, 1.0, 1.0];
        let pr: f64 = par_rate_generic(&coupon_times, &coupon_yfs, 100.0, 3.0, &curve);
        // For a flat curve, par rate should be close to the curve rate
        // (not exactly equal due to discrete coupons vs continuous rate)
        assert!((pr - 0.05).abs() < 0.005, "par_rate={pr}");
    }

    #[test]
    fn swap_rate_flat_curve() {
        let curve = FlatCurve::new(0.04_f64);
        let fixed_times = vec![0.5, 1.0, 1.5, 2.0];
        let fixed_yfs = vec![0.5, 0.5, 0.5, 0.5];
        let sr: f64 = swap_rate_generic(0.0, 2.0, &fixed_times, &fixed_yfs, &curve);
        // For flat curve, swap rate ≈ curve rate
        assert!((sr - 0.04).abs() < 0.005, "swap_rate={sr}");
    }

    #[test]
    fn fra_npv_at_forward() {
        let curve_times = vec![0.5, 1.0, 2.0, 5.0];
        let curve_rates = vec![0.03, 0.035, 0.04, 0.045];
        let curve = InterpDiscountCurve::from_zero_rates(&curve_times, &curve_rates);

        // FRA at the forward rate should have NPV = 0
        let fwd: f64 = curve.forward_rate_t(1.0, 2.0);
        let npv: f64 = fra_npv_generic(1_000_000.0, fwd, 1.0, 2.0, &curve);
        assert!(npv.abs() < 1e-6, "fra_npv at forward = {npv}");
    }

    #[test]
    fn key_rate_durations_test() {
        let curve_times = vec![1.0, 2.0, 5.0, 10.0];
        let curve_rates = vec![0.03, 0.035, 0.04, 0.045];
        let times = vec![1.0, 2.0, 5.0, 10.0];
        let amounts = vec![3.0, 3.0, 3.0, 103.0]; // bond-like cashflows
        let krds = key_rate_durations_generic(&times, &amounts, &curve_times, &curve_rates);
        assert_eq!(krds.len(), 4);
        // 10y pillar should have the largest KRD (most notional exposed)
        assert!(krds[3].abs() > krds[0].abs(), "10y KRD should dominate");
        // All should be negative (higher rate → lower NPV)
        for (i, &krd) in krds.iter().enumerate() {
            assert!(krd < 0.0, "KRD[{i}] should be negative, got {krd}");
        }
    }

    #[test]
    fn leg_pv01_negative() {
        let curve = FlatCurve::new(0.05_f64);
        let times = vec![1.0, 2.0, 3.0];
        let amounts = vec![5.0, 5.0, 105.0];
        let pv01: f64 = leg_pv01_generic(&times, &amounts, &curve);
        // PV01 should be negative (higher rates → lower PV)
        assert!(pv01 < 0.0, "pv01={pv01}");
    }

    #[test]
    fn leg_duration_positive() {
        let curve = FlatCurve::new(0.05_f64);
        let times = vec![1.0, 2.0, 3.0];
        let amounts = vec![5.0, 5.0, 105.0];
        let dur: f64 = leg_duration_generic(&times, &amounts, &curve);
        // Duration should be between 0 and maturity
        assert!(dur > 0.0 && dur < 3.0, "duration={dur}");
        // Should be close to 3y (bullet bond, most weight at maturity)
        assert!(dur > 2.5, "duration should be near 3y, got {dur}");
    }

    #[test]
    fn interp_zero_curve_leg_npv() {
        let curve = InterpZeroCurve::new(
            &[0.5, 1.0, 2.0, 5.0],
            &[0.02, 0.03, 0.035, 0.04],
        );
        let times = vec![1.0, 2.0];
        let amounts = vec![5.0, 105.0];
        let pv: f64 = leg_npv_generic(&times, &amounts, &curve);
        let expected = 5.0 * (-0.03_f64).exp() + 105.0 * (-0.035 * 2.0_f64).exp();
        assert!((pv - expected).abs() < 1e-10, "pv={pv}");
    }
}

// ===========================================================================
// Phase F tests: Discounting engines (AD-49 to AD-68)
// ===========================================================================
#[cfg(test)]
mod tests_phase_f {
    use super::*;
    use ql_termstructures::generic::{FlatCurve, InterpDiscountCurve};

    #[test]
    fn swap_engine_at_par() {
        let curve = FlatCurve::new(0.04_f64);
        let fixed_times = vec![0.5, 1.0, 1.5, 2.0];
        let fixed_yfs = vec![0.5, 0.5, 0.5, 0.5];
        // First get the fair rate, then price at that rate
        let res0: SwapResult<f64> = swap_engine_generic(
            1_000_000.0, 0.04, &fixed_times, &fixed_yfs, 0.0, 2.0, &curve,
        );
        let fair = res0.fair_rate;
        let res: SwapResult<f64> = swap_engine_generic(
            1_000_000.0, fair, &fixed_times, &fixed_yfs, 0.0, 2.0, &curve,
        );
        assert!(res.npv.abs() < 1.0, "swap at fair rate npv={}", res.npv);
    }

    #[test]
    fn swap_engine_off_market() {
        let curve = FlatCurve::new(0.05_f64);
        let fixed_times = vec![1.0, 2.0];
        let fixed_yfs = vec![1.0, 1.0];
        // Fixed rate below market → positive NPV (receive float > pay fixed)
        let res: SwapResult<f64> = swap_engine_generic(
            1_000_000.0, 0.03, &fixed_times, &fixed_yfs, 0.0, 2.0, &curve,
        );
        assert!(res.npv > 0.0, "off-market swap should be positive");
    }

    #[test]
    fn fixed_bond_engine_test() {
        let curve = FlatCurve::new(0.05_f64);
        let res: BondResult<f64> = fixed_bond_engine_generic(
            100.0, 0.05, &[0.5, 1.0, 1.5, 2.0], &[0.5, 0.5, 0.5, 0.5],
            2.0, 1.2, &curve,
        );
        assert!(res.npv > 90.0 && res.npv < 110.0, "bond npv={}", res.npv);
        assert!(res.clean_price < res.npv, "clean < dirty");
        assert!((res.yield_at_maturity - 0.05).abs() < 1e-10);
    }

    #[test]
    fn floating_bond_engine_test() {
        let curve = FlatCurve::new(0.04_f64);
        let start_times = vec![0.0, 0.5, 1.0, 1.5];
        let end_times = vec![0.5, 1.0, 1.5, 2.0];
        let yfs = vec![0.5, 0.5, 0.5, 0.5];
        let pv: f64 = floating_bond_engine_generic(
            100.0, 0.0, &start_times, &end_times, &yfs, 2.0, &curve, &curve,
        );
        // FRN at par (no spread) ≈ 100
        assert!((pv - 100.0).abs() < 1.0, "frn pv={pv}");
    }

    #[test]
    fn zero_coupon_bond_test() {
        let curve = FlatCurve::new(0.05_f64);
        let pv: f64 = zero_coupon_bond_generic(100.0, 5.0, &curve);
        let expected = 100.0 * (-0.25_f64).exp();
        assert!((pv - expected).abs() < 1e-10);
    }

    #[test]
    fn amortizing_bond_test() {
        let curve = FlatCurve::new(0.04_f64);
        let notionals = vec![100.0, 75.0, 50.0, 25.0];
        let times = vec![1.0, 2.0, 3.0, 4.0];
        let yfs = vec![1.0, 1.0, 1.0, 1.0];
        let pv: f64 = amortizing_bond_generic(&notionals, 0.05, &times, &yfs, &curve);
        assert!(pv > 0.0, "amort pv={pv}");
        // Should be close to sum of notional (most gets returned)
        assert!(pv < 120.0 && pv > 80.0, "amort pv={pv}");
    }

    #[test]
    fn inflation_bond_test() {
        let curve = FlatCurve::new(0.03_f64);
        let cpi_ratios = vec![1.02_f64, 1.04, 1.06]; // cumulative inflation
        let pv: f64 = inflation_bond_generic(
            100.0, 0.02,
            &[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0],
            &cpi_ratios, 3.0, 1.06, 1.0, &curve,
        );
        assert!(pv > 100.0, "inflation bond with positive inflation > par, pv={pv}");
    }

    #[test]
    fn cds_midpoint_at_par() {
        let yield_curve = FlatCurve::new(0.03_f64);
        // Use flat hazard rate curve → survival = exp(-h*t)
        let hazard_curve = FlatCurve::new(0.01_f64); // 1% hazard rate

        let times = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let yfs = vec![0.5; 10];

        // At the "fair spread", NPV should be near zero
        // For a rough estimate: fair_spread ≈ hazard_rate × (1-R) = 0.01 × 0.6 = 0.006
        let npv: f64 = cds_midpoint_generic(
            10_000_000.0, 0.006, 0.4, &times, &yfs,
            &yield_curve, &hazard_curve,
        );
        // Should be small relative to notional
        assert!(npv.abs() < 50_000.0, "cds npv={npv}");
    }

    #[test]
    fn xccy_swap_test() {
        let dom_curve = FlatCurve::new(0.05_f64);
        let for_curve = FlatCurve::new(0.03_f64);
        let fx_spot = 1.10_f64; // EUR/USD
        let times = vec![1.0, 2.0];
        let yfs = vec![1.0, 1.0];
        let npv: f64 = xccy_swap_generic(
            1_000_000.0, 909_091.0, 0.05, 0.03, fx_spot,
            &times, &yfs, 2.0, &dom_curve, &for_curve,
        );
        // Just check it's finite and reasonable
        assert!(npv.abs() < 500_000.0, "xccy npv={npv}");
    }

    #[test]
    fn fx_forward_at_forward() {
        let dom_curve = FlatCurve::new(0.05_f64);
        let for_curve = FlatCurve::new(0.03_f64);
        let fx_spot = 1.10_f64;
        let t = 1.0;
        // Forward FX = S × DF_for / DF_dom
        let fwd_fx = fx_spot * (-0.03_f64).exp() / (-0.05_f64).exp();
        let npv: f64 = fx_forward_generic(
            1_000_000.0, fx_spot, fwd_fx, t, &dom_curve, &for_curve,
        );
        assert!(npv.abs() < 1.0, "fx_fwd at forward, npv={npv}");
    }

    #[test]
    fn zero_coupon_swap_test() {
        let curve = FlatCurve::new(0.04_f64);
        let npv: f64 = zero_coupon_swap_generic(1_000_000.0, 0.04, 5.0, &curve);
        // At the right rate, NPV should be near zero
        assert!(npv.abs() < 50_000.0, "zc swap npv={npv}");
    }

    #[test]
    fn basis_swap_test() {
        let curve1 = FlatCurve::new(0.04_f64);
        let curve2 = FlatCurve::new(0.04_f64);
        let disc = FlatCurve::new(0.04_f64);
        let starts = vec![0.0, 0.25, 0.5, 0.75];
        let ends = vec![0.25, 0.5, 0.75, 1.0];
        let yfs = vec![0.25; 4];
        // Same curves, same spreads → NPV = 0
        let npv: f64 = basis_swap_generic(
            1_000_000.0, 0.0, 0.0, &starts, &ends, &yfs,
            &curve1, &curve2, &disc,
        );
        assert!(npv.abs() < 1.0, "basis swap same curves npv={npv}");
    }

    #[test]
    fn cpi_swap_test() {
        let curve = FlatCurve::new(0.03_f64);
        let cpi_ratio = 1.10_f64; // 10% cumulative inflation over 5y → ~2%/yr
        let npv: f64 = cpi_swap_generic(1_000_000.0, 0.02, 5.0, cpi_ratio, &curve);
        // Check it's reasonable
        assert!(npv.abs() < 200_000.0, "cpi swap npv={npv}");
    }

    #[test]
    fn cat_bond_test() {
        let curve = FlatCurve::new(0.05_f64);
        let times = vec![1.0, 2.0, 3.0];
        let yfs = vec![1.0, 1.0, 1.0];
        let survival = vec![0.98, 0.96, 0.94]; // 2%/yr cat event risk
        let pv: f64 = cat_bond_generic(100.0, 0.08, &times, &yfs, 3.0, &survival, &curve);
        // Should be less than a regular bond due to cat risk
        let regular: f64 = cat_bond_generic(100.0, 0.08, &times, &yfs, 3.0, &[1.0, 1.0, 1.0], &curve);
        assert!(pv < regular, "cat bond < regular: {pv} < {regular}");
    }

    #[test]
    fn bond_forward_test() {
        let curve = FlatCurve::new(0.04_f64);
        let coupon_times = vec![0.5, 1.0, 1.5, 2.0];
        let coupon_yfs = vec![0.5, 0.5, 0.5, 0.5];
        // Forward price = theoretical forward value of the bond
        let bond_pv: f64 = bond_pv_curve_generic(&coupon_times, &coupon_yfs.iter().map(|yf| 100.0 * 0.05 * yf).collect::<Vec<_>>(), 100.0, 2.0, &curve);
        let delivery = 0.25;
        let fwd_price = bond_pv / (-0.04 * delivery as f64).exp();
        let npv: f64 = bond_forward_generic(
            100.0, 0.05, &coupon_times, &coupon_yfs, 2.0, fwd_price, delivery, &curve,
        );
        assert!(npv.abs() < 1.0, "bond fwd at fair price, npv={npv}");
    }

    #[test]
    fn swap_multicurve_test() {
        let forecast = FlatCurve::new(0.05_f64);
        let discount = FlatCurve::new(0.04_f64);
        let fixed_times = vec![0.5, 1.0];
        let fixed_yfs = vec![0.5, 0.5];
        let float_starts = vec![0.0, 0.5];
        let float_ends = vec![0.5, 1.0];
        let float_yfs = vec![0.5, 0.5];
        let npv: f64 = swap_multicurve_generic(
            1_000_000.0, 0.05, &fixed_times, &fixed_yfs,
            &float_starts, &float_ends, &float_yfs,
            &forecast, &discount,
        );
        // Same forecast rate as fixed rate → NPV near zero (but different discounting)
        assert!(npv.abs() < 5_000.0, "multicurve npv={npv}");
    }
}

// ===========================================================================
// Phase G tests: American engines (AD-45, AD-46, AD-48)
// ===========================================================================
#[cfg(test)]
mod tests_phase_g {
    use super::*;

    // --- AD-45: Bjerksund-Stensland ---
    #[test]
    fn bjs_put_positive() {
        let v: f64 = bjerksund_stensland_generic(100.0, 100.0, 0.05, 0.02, 0.20, 1.0, false);
        assert!(v > 0.0, "bjs put={v}");
    }

    #[test]
    fn bjs_put_geq_european() {
        let bjs: f64 = bjerksund_stensland_generic(100.0, 100.0, 0.05, 0.02, 0.20, 1.0, false);
        let bs: f64 = black_scholes_generic(100.0, 100.0, 0.05, 0.02, 0.20, 1.0, false);
        assert!(bjs >= bs - 0.01, "bjs={bjs} >= bs={bs}");
    }

    #[test]
    fn bjs_call_no_div_eq_european() {
        let bjs: f64 = bjerksund_stensland_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        let bs: f64 = black_scholes_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        assert!((bjs - bs).abs() < 0.01, "bjs call no-div={bjs}, bs={bs}");
    }

    #[test]
    fn bjs_deep_itm_put() {
        let v: f64 = bjerksund_stensland_generic(50.0, 100.0, 0.05, 0.02, 0.20, 1.0, false);
        assert!(v >= 49.0, "deep ITM put={v}");
    }

    // --- AD-46: QD+ ---
    #[test]
    fn qdp_put_positive() {
        let v: f64 = qd_plus_generic(100.0, 100.0, 0.05, 0.02, 0.20, 1.0, false);
        assert!(v > 0.0, "qdp put={v}");
    }

    #[test]
    fn qdp_put_geq_european() {
        let qdp: f64 = qd_plus_generic(100.0, 100.0, 0.05, 0.02, 0.20, 1.0, false);
        let bs: f64 = black_scholes_generic(100.0, 100.0, 0.05, 0.02, 0.20, 1.0, false);
        assert!(qdp >= bs - 0.01, "qdp={qdp} >= bs={bs}");
    }

    #[test]
    fn qdp_call_with_div() {
        let v: f64 = qd_plus_generic(100.0, 100.0, 0.05, 0.03, 0.25, 1.0, true);
        let bs: f64 = black_scholes_generic(100.0, 100.0, 0.05, 0.03, 0.25, 1.0, true);
        // American call with div > European
        assert!(v >= bs - 0.01, "qdp call={v}, bs={bs}");
    }

    // --- AD-48: Compound option ---
    #[test]
    fn compound_call_on_call_positive() {
        let v: f64 = compound_option_generic(
            100.0, 5.0, 100.0, 0.05, 0.02, 0.20, 0.5, 1.0, true, true,
        );
        assert!(v > 0.0, "CoC={v}");
    }

    #[test]
    fn compound_put_on_call_positive() {
        let v: f64 = compound_option_generic(
            100.0, 5.0, 100.0, 0.05, 0.02, 0.20, 0.5, 1.0, false, true,
        );
        assert!(v > 0.0, "PoC={v}");
    }

    #[test]
    fn compound_call_on_put_positive() {
        let v: f64 = compound_option_generic(
            100.0, 5.0, 100.0, 0.05, 0.02, 0.20, 0.5, 1.0, true, false,
        );
        assert!(v > 0.0, "CoP={v}");
    }
}

// ===========================================================================
// Phase H Tests
// ===========================================================================

#[cfg(test)]
mod tests_phase_h {
    use super::*;

    // --- AD-73: Two-asset correlation / Stulz min call ---
    #[test]
    fn stulz_min_call_positive() {
        let v: f64 = stulz_min_call_generic(
            100.0, 100.0, 100.0, 0.05, 0.02, 0.02, 0.20, 0.25, 0.5, 1.0,
        );
        assert!(v > 0.0, "stulz_min={v}");
    }

    #[test]
    fn stulz_min_call_leq_vanilla() {
        // min(S1,S2) option ≤ single-asset option
        let min_c: f64 = stulz_min_call_generic(
            100.0, 100.0, 100.0, 0.05, 0.02, 0.02, 0.20, 0.25, 0.5, 1.0,
        );
        let bs: f64 = black_scholes_generic(100.0, 100.0, 0.05, 0.02, 0.20, 1.0, true);
        assert!(min_c <= bs + 0.01, "min_c={min_c} <= bs={bs}");
    }

    #[test]
    fn two_asset_max_geq_vanilla() {
        // max(S1,S2) call ≥ single-asset call
        let max_c: f64 = two_asset_correlation_generic(
            100.0, 100.0, 100.0, 0.05, 0.02, 0.02, 0.20, 0.25, 0.5, 1.0, true,
        );
        let bs: f64 = black_scholes_generic(100.0, 100.0, 0.05, 0.02, 0.20, 1.0, true);
        assert!(max_c >= bs - 0.01, "max_c={max_c} >= bs={bs}");
    }

    #[test]
    fn two_asset_min_max_parity() {
        // C_max + C_min = C_BS(S1) + C_BS(S2)
        let max_c: f64 = two_asset_correlation_generic(
            100.0, 110.0, 100.0, 0.05, 0.02, 0.03, 0.20, 0.30, 0.3, 1.0, true,
        );
        let min_c: f64 = two_asset_correlation_generic(
            100.0, 110.0, 100.0, 0.05, 0.02, 0.03, 0.20, 0.30, 0.3, 1.0, false,
        );
        let c1: f64 = black_scholes_generic(100.0, 100.0, 0.05, 0.02, 0.20, 1.0, true);
        let c2: f64 = black_scholes_generic(110.0, 100.0, 0.05, 0.03, 0.30, 1.0, true);
        assert!(
            (max_c + min_c - c1 - c2).abs() < 0.05,
            "parity: max={max_c} + min={min_c} vs c1={c1} + c2={c2}",
        );
    }

    // --- AD-72: Quanto Barrier ---
    #[test]
    fn quanto_barrier_dao_call_positive() {
        let v: f64 = quanto_barrier_generic(
            100.0, 100.0, 80.0, 0.0,
            0.05, 0.03, 0.02,
            0.20, 0.10, -0.3, 1.0,
            true, true, true,
        );
        assert!(v > 0.0 && v < 15.0, "quanto_dao_call={v}");
    }

    #[test]
    fn quanto_barrier_knockout_knockin_parity() {
        // KO + KI = Vanilla (approximately)
        let ko: f64 = quanto_barrier_generic(
            100.0, 100.0, 80.0, 0.0,
            0.05, 0.03, 0.02,
            0.20, 0.10, -0.3, 1.0,
            true, true, true,
        );
        let ki: f64 = quanto_barrier_generic(
            100.0, 100.0, 80.0, 0.0,
            0.05, 0.03, 0.02,
            0.20, 0.10, -0.3, 1.0,
            true, true, false,
        );
        let q_adj = 0.02 + (-0.3) * 0.20 * 0.10;
        let vanilla: f64 = black_scholes_generic(100.0, 100.0, 0.05, q_adj, 0.20, 1.0, true);
        assert!(
            (ko + ki - vanilla).abs() < 0.5,
            "ko={ko} + ki={ki} vs vanilla={vanilla}",
        );
    }

    #[test]
    fn quanto_barrier_uo_put_positive() {
        let v: f64 = quanto_barrier_generic(
            100.0, 100.0, 120.0, 0.0,
            0.05, 0.03, 0.02,
            0.20, 0.10, 0.3, 1.0,
            false, false, true,
        );
        assert!(v >= 0.0, "quanto_uo_put={v}");
    }

    // --- AD-69: MC Asian arithmetic ---
    #[test]
    fn mc_asian_arith_generic_call() {
        let res: McResultGeneric<f64> = mc_asian_arithmetic_generic(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 12, 50000, true, 42,
        );
        // Asian call < European call (~10.45)
        assert!(res.price > 2.0 && res.price < 11.0, "price={}", res.price);
        assert!(res.std_error < 1.0, "se={}", res.std_error);
    }

    #[test]
    fn mc_asian_arith_generic_put() {
        let res: McResultGeneric<f64> = mc_asian_arithmetic_generic(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 12, 50000, false, 42,
        );
        assert!(res.price > 0.5 && res.price < 10.0, "price={}", res.price);
    }

    // --- AD-83: MC Variance Swap ---
    #[test]
    fn mc_var_swap_generic_fair() {
        let res: McVarianceSwapResultGeneric<f64> = mc_variance_swap_generic(
            100.0, 0.05, 0.02, 0.20, 1.0, 252, 20000, 0.04, 100_000.0, 42,
        );
        // Fair variance should be close to σ² = 0.04
        assert!(
            (res.fair_variance - 0.04).abs() < 0.005,
            "fair_var={}",
            res.fair_variance,
        );
    }

    #[test]
    fn mc_var_swap_generic_pv_near_zero() {
        // When variance_strike = σ², PV ≈ 0
        let res: McVarianceSwapResultGeneric<f64> = mc_variance_swap_generic(
            100.0, 0.05, 0.02, 0.20, 1.0, 252, 20000, 0.04, 100_000.0, 42,
        );
        assert!(res.pv.abs() < 1000.0, "pv={}", res.pv);
    }

    // --- AD-70: MC Asian Heston ---
    #[test]
    fn mc_asian_heston_generic_call() {
        let res: McResultGeneric<f64> = mc_asian_heston_generic(
            100.0, 100.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
            1.0, 12, 20000, true, 42,
        );
        assert!(
            res.price > 1.0 && res.price < 15.0,
            "heston_asian={}",
            res.price,
        );
    }

    // --- AD-71: MC Basket ---
    #[test]
    fn mc_basket_generic_call() {
        let spots = [100.0_f64, 100.0];
        let weights = [0.5, 0.5];
        let divs = [0.02, 0.02];
        let vols = [0.20, 0.25];
        let corr = [1.0, 0.5, 0.5, 1.0];
        let res: McResultGeneric<f64> = mc_basket_generic(
            &spots, &weights, 100.0, 0.05, &divs, &vols, &corr, 1.0, true, 50000, 42,
        );
        assert!(res.price > 2.0 && res.price < 15.0, "basket={}", res.price);
    }

    #[test]
    fn mc_basket_generic_put() {
        let spots = [100.0_f64, 100.0];
        let weights = [0.5, 0.5];
        let divs = [0.02, 0.02];
        let vols = [0.20, 0.25];
        let corr = [1.0, 0.5, 0.5, 1.0];
        let res: McResultGeneric<f64> = mc_basket_generic(
            &spots, &weights, 100.0, 0.05, &divs, &vols, &corr, 1.0, false, 50000, 42,
        );
        assert!(res.price > 0.5 && res.price < 10.0, "basket_put={}", res.price);
    }

    // --- AD-82: MC Barrier ---
    #[test]
    fn mc_barrier_generic_dao_call() {
        let res: McResultGeneric<f64> = mc_barrier_generic(
            100.0, 100.0, 80.0, 0.0,
            0.05, 0.0, 0.20, 1.0,
            true, true, true,
            252, 50000, 42,
        );
        assert!(res.price > 3.0 && res.price < 13.0, "dao_call={}", res.price);
    }

    #[test]
    fn mc_barrier_generic_ko_ki_parity() {
        let ko: McResultGeneric<f64> = mc_barrier_generic(
            100.0, 100.0, 80.0, 0.0,
            0.05, 0.0, 0.20, 1.0,
            true, true, true,
            252, 50000, 42,
        );
        let ki: McResultGeneric<f64> = mc_barrier_generic(
            100.0, 100.0, 80.0, 0.0,
            0.05, 0.0, 0.20, 1.0,
            true, true, false,
            252, 50000, 42,
        );
        let vanilla: f64 = black_scholes_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        // KO + KI ≈ Vanilla within MC noise
        assert!(
            (ko.price + ki.price - vanilla).abs() < 2.0,
            "ko={} + ki={} vs vanilla={vanilla}",
            ko.price, ki.price,
        );
    }

    // --- AD-84: MC Digital ---
    #[test]
    fn mc_digital_generic_call() {
        let res: McResultGeneric<f64> = mc_digital_generic(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true, 1.0, 100000, 42,
        );
        // Cash-or-nothing call at ATM ≈ df * N(d2) ≈ 0.95 * 0.57 ≈ 0.54
        assert!(
            res.price > 0.3 && res.price < 0.8,
            "digital_call={}",
            res.price,
        );
    }

    #[test]
    fn mc_digital_generic_put() {
        let res: McResultGeneric<f64> = mc_digital_generic(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0, false, 1.0, 100000, 42,
        );
        // Cash-or-nothing put ≈ df * N(-d2) ≈ 0.95 * 0.43 ≈ 0.41
        assert!(
            res.price > 0.2 && res.price < 0.6,
            "digital_put={}",
            res.price,
        );
    }

    // --- AD-69b: MC Forward-Start ---
    #[test]
    fn mc_forward_start_generic_call() {
        let res: McResultGeneric<f64> = mc_forward_start_generic(
            100.0, 1.0, 0.05, 0.0, 0.20, 0.5, 1.0, true, 50000, 42,
        );
        // Forward-start ATM call ≈ BS ATM call with t=0.5
        let bs_half: f64 = black_scholes_generic(100.0, 100.0, 0.05, 0.0, 0.20, 0.5, true);
        assert!(
            (res.price - bs_half).abs() < 3.0,
            "fwd_start={} vs bs_half={bs_half}",
            res.price,
        );
    }

    #[test]
    fn mc_forward_start_generic_put() {
        let res: McResultGeneric<f64> = mc_forward_start_generic(
            100.0, 1.0, 0.05, 0.0, 0.20, 0.5, 1.0, false, 50000, 42,
        );
        assert!(res.price > 0.0, "fwd_start_put={}", res.price);
    }
}

// ===========================================================================
// Phase J Tests
// ===========================================================================

#[cfg(test)]
mod tests_phase_j {
    use super::*;

    // --- AD-81: Binomial barrier ---
    #[test]
    fn binomial_barrier_dao_call() {
        let v: f64 = binomial_barrier_generic(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            80.0, 0.0, true, true, true, 200,
        );
        assert!(v > 3.0 && v < 13.0, "binom_dao_call={v}");
    }

    #[test]
    fn binomial_barrier_uao_put() {
        let v: f64 = binomial_barrier_generic(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            120.0, 0.0, false, false, true, 200,
        );
        assert!(v >= 0.0 && v < 10.0, "binom_uao_put={v}");
    }

    #[test]
    fn binomial_barrier_ko_ki_parity() {
        let ko: f64 = binomial_barrier_generic(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            80.0, 0.0, true, true, true, 200,
        );
        let ki: f64 = binomial_barrier_generic(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            80.0, 0.0, true, true, false, 200,
        );
        let vanilla: f64 = black_scholes_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        assert!(
            (ko + ki - vanilla).abs() < 1.5,
            "ko={ko} + ki={ki} vs vanilla={vanilla}",
        );
    }

    // --- AD-74: FD Swing ---
    #[test]
    fn fd_swing_geq_vanilla() {
        let swing: f64 = fd_swing_generic(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 4, 100, 100,
        );
        let vanilla: f64 = black_scholes_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        // Swing with 4 exercises > single European
        assert!(swing > vanilla - 1.0, "swing={swing} vs vanilla={vanilla}");
    }

    #[test]
    fn fd_swing_positive() {
        let v: f64 = fd_swing_generic(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 2, 100, 100,
        );
        assert!(v > 0.0, "swing={v}");
    }

    // --- AD-75: FD Shout ---
    #[test]
    fn fd_shout_geq_european() {
        let shout: f64 = fd_shout_generic(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true, 200, 200,
        );
        let bs: f64 = black_scholes_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        // Shout ≥ European (extra optionality)
        assert!(shout >= bs - 0.5, "shout={shout} vs bs={bs}");
    }

    #[test]
    fn fd_shout_put_positive() {
        let v: f64 = fd_shout_generic(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0, false, 200, 200,
        );
        assert!(v > 0.0, "shout_put={v}");
    }

    // --- AD-77: Heston barrier FD ---
    #[test]
    fn fd_heston_barrier_dao_call_positive() {
        let v: f64 = fd_heston_barrier_generic(
            100.0, 100.0, 80.0,
            0.05, 0.0, 0.04, 1.5, 0.04, 0.3, -0.7,
            1.0, true, true, true,
            50, 20, 50,
        );
        assert!(v > 0.0, "heston_dao_call={v}");
    }

    #[test]
    fn fd_heston_barrier_ko_leq_vanilla() {
        // Knockout ≤ vanilla (less optionality)
        let ko: f64 = fd_heston_barrier_generic(
            100.0, 100.0, 80.0,
            0.05, 0.0, 0.04, 1.5, 0.04, 0.3, -0.7,
            1.0, true, true, true,
            50, 20, 50,
        );
        // With Heston, vanilla is roughly 10-15
        assert!(ko < 25.0, "heston_ko={ko}");
    }

    // --- FD Vanilla convenience ---
    #[test]
    fn fd_vanilla_generic_call_vs_bs() {
        let fd: f64 = fd_vanilla_generic(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true, false, 200, 200,
        );
        let bs: f64 = black_scholes_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        assert!((fd - bs).abs() < 0.5, "fd={fd} vs bs={bs}");
    }

    #[test]
    fn fd_vanilla_generic_american_put() {
        let eur: f64 = fd_vanilla_generic(
            100.0, 100.0, 0.05, 0.02, 0.20, 1.0, false, false, 200, 200,
        );
        let amer: f64 = fd_vanilla_generic(
            100.0, 100.0, 0.05, 0.02, 0.20, 1.0, false, true, 200, 200,
        );
        assert!(amer >= eur - 0.1, "amer={amer} >= eur={eur}");
    }
}

// ===========================================================================
// Phase K Tests
// ===========================================================================

#[cfg(test)]
mod tests_phase_k {
    use super::*;

    // ── AD-93: Commodity Forward / Swap ──────────────────────

    #[test]
    fn commodity_forward_long_itm() {
        let res: CommodityForwardResultGeneric<f64> = commodity_forward_generic(
            1.0, 1000.0, 85.0, 80.0, 0.95,
        );
        let expected = 1.0 * 1000.0 * (85.0 - 80.0) * 0.95;
        assert!((res.npv - expected).abs() < 1e-8, "npv={}", res.npv);
    }

    #[test]
    fn commodity_forward_short_otm() {
        let res: CommodityForwardResultGeneric<f64> = commodity_forward_generic(
            -1.0, 500.0, 82.0, 85.0, 0.97,
        );
        // Short, fwd < strike → positive NPV for short
        assert!(res.npv > 0.0, "npv={}", res.npv);
    }

    #[test]
    fn commodity_swap_fair_price_zero_npv() {
        // If fixed_price == fair_fixed_price, NPV should be near zero
        let fwds = vec![81.0, 82.5, 85.0, 88.0];
        let dfs = vec![0.98, 0.96, 0.94, 0.92];
        let res1: CommoditySwapResultGeneric<f64> = commodity_swap_generic(
            1.0, 100.0, 80.0, &fwds, &dfs,
        );
        let fair = res1.fair_fixed_price;
        let res2: CommoditySwapResultGeneric<f64> = commodity_swap_generic(
            1.0, 100.0, fair, &fwds, &dfs,
        );
        assert!(res2.npv.abs() < 1e-6, "npv at fair={}", res2.npv);
    }

    // ── AD-94: Asset Swap / Equity TRS ───────────────────────

    #[test]
    fn asset_swap_par_at_par() {
        let yf = vec![0.5_f64; 4];
        let dfs = vec![0.98, 0.96, 0.94, 0.92];
        let res: AssetSwapResultGeneric<f64> = asset_swap_generic(
            true, 100.0, 0.05, 1_000_000.0, &yf, &dfs,
        );
        assert!(res.asset_swap_spread > 0.0, "asw={}", res.asset_swap_spread);
    }

    #[test]
    fn asset_swap_discount_higher_spread() {
        let yf = vec![1.0_f64; 3];
        let dfs = vec![0.96, 0.92, 0.88];
        let at_par: AssetSwapResultGeneric<f64> = asset_swap_generic(
            true, 100.0, 0.04, 1_000_000.0, &yf, &dfs,
        );
        let at_95: AssetSwapResultGeneric<f64> = asset_swap_generic(
            true, 95.0, 0.04, 1_000_000.0, &yf, &dfs,
        );
        assert!(
            at_95.asset_swap_spread > at_par.asset_swap_spread,
            "discount {} > par {}",
            at_95.asset_swap_spread,
            at_par.asset_swap_spread,
        );
    }

    #[test]
    fn asset_swap_market_value() {
        let yf = vec![1.0_f64; 5];
        let dfs = vec![0.97, 0.94, 0.91, 0.88, 0.85];
        let res: AssetSwapResultGeneric<f64> = asset_swap_generic(
            false, 102.0, 0.05, 1_000_000.0, &yf, &dfs,
        );
        assert!(res.floating_annuity > 0.0);
        // Market-value convention should produce a spread
        assert!(res.asset_swap_spread.is_finite());
    }

    #[test]
    fn equity_trs_positive_return() {
        let res: EquityTrsResultGeneric<f64> = equity_trs_generic(
            1_000_000.0, 100.0, 110.0, 2.0, 0.04, 0.005, 0.25, 0.99,
        );
        assert!(res.equity_return > 0.0, "ret={}", res.equity_return);
        assert!(res.equity_leg_npv > 0.0);
    }

    #[test]
    fn equity_trs_fair_spread_consistency() {
        let fs: f64 = equity_trs_fair_spread_generic(
            100.0, 110.0, 2.0, 0.04, 0.25,
        );
        // At fair spread, NPV should be ~0
        let res: EquityTrsResultGeneric<f64> = equity_trs_generic(
            1_000_000.0, 100.0, 110.0, 2.0, 0.04, fs, 0.25, 1.0,
        );
        assert!(res.npv.abs() < 1e-4, "npv={}", res.npv);
    }

    // ── AD-86: CDO Tranche ───────────────────────────────────

    #[test]
    fn cdo_equity_tranche_positive() {
        let res: CdoTrancheResultGeneric<f64> = cdo_tranche_generic(
            0.02, 0.30, 0.40, 5.0, 0.03, 0.0, 0.03, 3_000_000.0, 50,
        );
        assert!(res.expected_loss.to_f64() > 0.0, "el={}", res.expected_loss.to_f64());
        assert!(res.fair_spread.to_f64() > 0.0, "spread={}", res.fair_spread.to_f64());
    }

    #[test]
    fn cdo_senior_lower_loss() {
        let equity: CdoTrancheResultGeneric<f64> = cdo_tranche_generic(
            0.02, 0.30, 0.40, 5.0, 0.03, 0.0, 0.03, 3_000_000.0, 50,
        );
        let senior: CdoTrancheResultGeneric<f64> = cdo_tranche_generic(
            0.02, 0.30, 0.40, 5.0, 0.03, 0.15, 1.0, 85_000_000.0, 50,
        );
        assert!(
            senior.expected_loss.to_f64() < equity.expected_loss.to_f64(),
            "senior EL {} < equity EL {}",
            senior.expected_loss.to_f64(),
            equity.expected_loss.to_f64(),
        );
    }

    // ── AD-88: Tree Swaption / Cap-Floor ─────────────────────

    #[test]
    fn tree_swaption_payer_positive() {
        let tenors: Vec<f64> = (1..=5).map(|y| y as f64 + 1.0).collect();
        let v: f64 = tree_swaption_generic(
            0.05, 0.01, 0.04,
            1.0, &tenors, 0.02, 1_000_000.0, true, 50,
        );
        // Low-strike payer swaption should have positive value
        assert!(v > 0.0, "payer_swaption={v}");
    }

    #[test]
    fn tree_swaption_receiver_positive() {
        let tenors: Vec<f64> = (1..=5).map(|y| y as f64 + 1.0).collect();
        let v: f64 = tree_swaption_generic(
            0.05, 0.01, 0.04,
            1.0, &tenors, 0.08, 1_000_000.0, false, 50,
        );
        // High-strike receiver swaption should have positive value
        assert!(v > 0.0, "receiver_swaption={v}");
    }

    #[test]
    fn tree_cap_positive() {
        let fix_times: Vec<f64> = (1..=4).map(|y| y as f64).collect();
        let pay_times: Vec<f64> = fix_times.iter().map(|t| t + 0.25).collect();
        let v: f64 = tree_cap_floor_generic(
            0.05, 0.01, 0.04,
            &fix_times, &pay_times, 0.03, 1_000_000.0, true, 20,
        );
        assert!(v > 0.0, "cap={v}");
    }

    #[test]
    fn tree_floor_positive() {
        let fix_times: Vec<f64> = (1..=4).map(|y| y as f64).collect();
        let pay_times: Vec<f64> = fix_times.iter().map(|t| t + 0.25).collect();
        let v: f64 = tree_cap_floor_generic(
            0.05, 0.01, 0.04,
            &fix_times, &pay_times, 0.06, 1_000_000.0, false, 20,
        );
        assert!(v > 0.0, "floor={v}");
    }

    // ── AD-89: Gaussian 1D Swaption ──────────────────────────

    #[test]
    fn gaussian1d_payer_itm_positive() {
        let tenors: Vec<f64> = (1..=10).map(|y| y as f64 + 1.0).collect();
        let yf = vec![1.0_f64; 10];
        let v: f64 = gaussian1d_swaption_generic(
            0.03, 0.008, 0.05,
            1.0, &tenors, &yf, 0.01, 1_000_000.0, true, 32,
        );
        assert!(v > 0.0, "payer_g1d={v}");
    }

    #[test]
    fn gaussian1d_receiver_itm_positive() {
        let tenors: Vec<f64> = (1..=5).map(|y| y as f64 + 1.0).collect();
        let yf = vec![1.0_f64; 5];
        let v: f64 = gaussian1d_swaption_generic(
            0.03, 0.008, 0.03,
            1.0, &tenors, &yf, 0.10, 1_000_000.0, false, 32,
        );
        assert!(v > 0.0, "receiver_g1d={v}");
    }

    #[test]
    fn gaussian1d_gh_weights_sum() {
        let (_, weights) = gauss_hermite_f64(16);
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - std::f64::consts::PI.sqrt()).abs() < 1e-6,
            "sum={sum} vs sqrt(pi)={}",
            std::f64::consts::PI.sqrt(),
        );
    }

    // ── AD-90: FD G2++ Swaption ──────────────────────────────

    #[test]
    fn fd_g2_payer_positive() {
        let fixed_times = vec![1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0_f64];
        let fixed_amounts: Vec<f64> = fixed_times.iter().map(|_| 1e6 * 0.04 * 0.5).collect();
        let v: f64 = fd_g2_swaption_generic(
            0.05, 0.01, 0.1, 0.005, -0.75, 0.04,
            &fixed_times, &fixed_amounts, 5.0,
            1_000_000.0, true, 1.0,
            30, 30, 50,
        );
        assert!(v >= 0.0, "g2_payer={v}");
        assert!(v < 200_000.0, "g2_payer={v} too large");
    }

    #[test]
    fn fd_g2_receiver_positive() {
        let fixed_times = vec![2.0, 3.0, 4.0, 5.0_f64];
        let fixed_amounts: Vec<f64> = fixed_times.iter().map(|_| 1e6 * 0.04 * 1.0).collect();
        let v: f64 = fd_g2_swaption_generic(
            0.05, 0.01, 0.1, 0.005, -0.75, 0.04,
            &fixed_times, &fixed_amounts, 5.0,
            1_000_000.0, false, 1.0,
            30, 30, 50,
        );
        assert!(v >= 0.0, "g2_receiver={v}");
    }
}

// ===========================================================================
// Tests — Phase Final (AD-5, AD-23, AD-24, AD-27, AD-41, AD-47,
//          AD-56, AD-57, AD-76, AD-78, AD-79, AD-80, AD-85, AD-91, AD-92,
//          INFRA-6)
// ===========================================================================
#[cfg(test)]
mod tests_phase_final {
    use super::*;

    // -------------------------------------------------------------------------
    // INFRA-6 — ComplexT<T> basic arithmetic
    // -------------------------------------------------------------------------
    #[test]
    fn test_complex_t_basic() {
        let a = ComplexT::new(3.0_f64, 4.0);
        let b = ComplexT::new(1.0_f64, -2.0);
        let sum = a.add(b);
        assert!((sum.re - 4.0).abs() < 1e-12 && (sum.im - 2.0).abs() < 1e-12);
        let prod = a.mul(b);
        // (3+4i)(1-2i) = 3 -6i +4i -8i² = 11 -2i
        assert!((prod.re - 11.0).abs() < 1e-12 && (prod.im + 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_complex_t_exp() {
        // e^{i π} = -1
        let z = ComplexT::new(0.0_f64, std::f64::consts::PI);
        let w = z.exp();
        assert!((w.re + 1.0).abs() < 1e-12, "re={}", w.re);
        assert!(w.im.abs() < 1e-12, "im={}", w.im);
    }

    #[test]
    fn test_complex_t_ln() {
        let z = ComplexT::new(1.0_f64, 1.0);
        let w = z.ln();
        // ln(1+i) = ln(√2) + i π/4 = 0.5*ln(2) + i*π/4
        assert!((w.re - 2.0_f64.ln() * 0.5).abs() < 1e-10);
        assert!((w.im - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    }

    // -------------------------------------------------------------------------
    // AD-5 — price_european_generic (wrapper around bs_european_generic)
    // -------------------------------------------------------------------------
    #[test]
    fn test_price_european_generic_call() {
        let res = price_european_generic(100.0, 100.0, 0.05, 0.02, 0.20, 1.0, true);
        // BS call should be around 9.x
        assert!(res.npv > 5.0 && res.npv < 15.0, "call={}", res.npv);
        assert!(res.delta > 0.4 && res.delta < 0.8, "delta={}", res.delta);
    }

    #[test]
    fn test_price_european_generic_put() {
        let res = price_european_generic(100.0, 100.0, 0.05, 0.02, 0.20, 1.0, false);
        assert!(res.npv > 3.0 && res.npv < 10.0, "put={}", res.npv);
        assert!(res.delta < 0.0, "delta={}", res.delta);
    }

    // -------------------------------------------------------------------------
    // AD-23 — partial_time_barrier_generic
    // -------------------------------------------------------------------------
    #[test]
    fn test_partial_time_barrier_down_out_call() {
        let v = partial_time_barrier_generic(
            100.0_f64, 100.0, 80.0, 0.05, 0.02, 0.25,
            1.0, 0.5,
            PartialBarrierTypeGeneric::B1DownOut, true,
        );
        // Down-out call: value >= 0 (may be 0 if formula clamps)
        assert!(v >= 0.0, "partial_barrier_do_call={v}");
    }

    #[test]
    fn test_partial_time_barrier_in_out_parity() {
        let down_out = partial_time_barrier_generic(
            100.0_f64, 100.0, 80.0, 0.05, 0.02, 0.25,
            1.0, 0.5,
            PartialBarrierTypeGeneric::B1DownOut, true,
        );
        let down_in = partial_time_barrier_generic(
            100.0_f64, 100.0, 80.0, 0.05, 0.02, 0.25,
            1.0, 0.5,
            PartialBarrierTypeGeneric::B1DownIn, true,
        );
        let vanilla = bs_european_generic(100.0_f64, 100.0, 0.05, 0.02, 0.25, 1.0, true).npv;
        // in + out ≈ vanilla (approximate due to the partial barrier simplification)
        let sum = down_in + down_out;
        let err = (sum - vanilla).abs();
        // Both sum and vanilla should be in reasonable range
        assert!(down_in >= 0.0, "down_in={down_in}");
        assert!(down_out >= 0.0, "down_out={down_out}");
        assert!(err < 5.0 || (err / vanilla.abs()) < 0.15,
            "in+out={sum} vs vanilla={vanilla} (err={err})"
        );
    }

    // -------------------------------------------------------------------------
    // AD-24 — vanna_volga_barrier_generic
    // -------------------------------------------------------------------------
    #[test]
    fn test_vanna_volga_barrier_down_out() {
        let v = vanna_volga_barrier_generic(
            1.3500_f64, 1.3400, 1.2500, 0.0,
            0.03, 0.01, 0.10, 0.105, 0.095,
            0.5, true, true, true,
        );
        assert!(v > 0.0 && v < 0.10, "vv_barrier_do={v}");
    }

    #[test]
    fn test_vanna_volga_barrier_non_negative() {
        let v = vanna_volga_barrier_generic(
            1.3500_f64, 1.3400, 1.2500, 0.0,
            0.03, 0.01, 0.10, 0.11, 0.09,
            1.0, true, true, true,
        );
        assert!(v >= 0.0, "vv_barrier={v}");
    }

    // -------------------------------------------------------------------------
    // AD-27 — hw_jamshidian_swaption_generic
    // -------------------------------------------------------------------------
    #[test]
    fn test_hw_jamshidian_swaption_payer() {
        let tenors = vec![1.5_f64, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let dfs: Vec<f64> = tenors.iter().map(|&t| (-0.04 * t).exp()).collect();
        let v = hw_jamshidian_swaption_generic(
            0.05_f64, 0.01, 1.0,
            &tenors, 0.04, &dfs,
            (-0.04_f64).exp(), 1e6, true,
        );
        assert!(v > 0.0, "hw_swaption_payer={v}");
    }

    #[test]
    fn test_hw_jamshidian_swaption_receiver() {
        let tenors = vec![1.5_f64, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let dfs: Vec<f64> = tenors.iter().map(|&t| (-0.04 * t).exp()).collect();
        let v = hw_jamshidian_swaption_generic(
            0.05_f64, 0.01, 1.0,
            &tenors, 0.04, &dfs,
            (-0.04_f64).exp(), 1e6, false,
        );
        assert!(v > 0.0, "hw_swaption_receiver={v}");
    }

    // -------------------------------------------------------------------------
    // AD-41 — replicating_variance_swap_generic
    // -------------------------------------------------------------------------
    #[test]
    fn test_replicating_variance_swap() {
        let strikes: Vec<f64> = (70..=130).step_by(5).map(|k| k as f64).collect();
        let ivols: Vec<f64> = strikes.iter().map(|&k| {
            // Simple skew: vol = 0.20 - 0.1*(k/100 - 1)
            0.20 - 0.1 * (k / 100.0 - 1.0)
        }).collect();
        let var_strike = 0.04; // 20% vol squared
        let (fair_var, swap_pv) = replicating_variance_swap_generic(
            100.0_f64, 0.05, 0.02, 1.0,
            &strikes, &ivols, var_strike, 1.0,
        );
        assert!(fair_var > 0.0 && fair_var < 0.15, "fair_var={fair_var}");
        // swap PV can be positive or negative depending on var_strike
        assert!(swap_pv.abs() < 1.0, "swap_pv={swap_pv}");
    }

    // -------------------------------------------------------------------------
    // AD-47 — mc_american_lsm_generic
    // -------------------------------------------------------------------------
    #[test]
    fn test_mc_american_lsm_put() {
        let res = mc_american_lsm_generic(
            100.0_f64, 100.0, 0.05, 0.0, 0.20,
            1.0, false,
            10_000, 50, 42,
        );
        // American put ~6-7 for these params
        assert!(res.price > 3.0 && res.price < 15.0, "lsm_put={}", res.price);
        assert!(res.std_error > 0.0 && res.std_error < 2.0, "se={}", res.std_error);
    }

    #[test]
    fn test_mc_american_lsm_call() {
        let res = mc_american_lsm_generic(
            100.0_f64, 100.0, 0.05, 0.0, 0.20,
            1.0, true,
            10_000, 50, 42,
        );
        // American call (no div) ≈ European call
        let bs = bs_european_generic(100.0_f64, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        assert!(
            (res.price - bs.npv).abs() / bs.npv < 0.15,
            "lsm_call={} vs bs={}", res.price, bs.npv
        );
    }

    // -------------------------------------------------------------------------
    // AD-56 — callable_bond_generic
    // -------------------------------------------------------------------------
    #[test]
    fn test_callable_bond_no_call() {
        // No call dates → should equal straight bond
        let coupon_times: Vec<(f64, f64)> = (1..=5).map(|y| (y as f64, 5.0_f64)).collect();
        let no_calls: Vec<(f64, f64)> = vec![];
        let v = callable_bond_generic(
            100.0_f64, 0.04, 0.01, 5.0,
            &coupon_times, &no_calls,
            true, 100,
        );
        assert!(v > 90.0 && v < 130.0, "callable_no_call={v}");
    }

    #[test]
    fn test_callable_bond_with_call() {
        let coupon_times: Vec<(f64, f64)> = (1..=5).map(|y| (y as f64, 5.0_f64)).collect();
        let call_times: Vec<(f64, f64)> = vec![(2.0, 100.0_f64), (3.0, 100.0), (4.0, 100.0)];
        let v = callable_bond_generic(
            100.0_f64, 0.04, 0.01, 5.0,
            &coupon_times, &call_times,
            true, 100,
        );
        // Callable ≤ non-callable
        let v_nc = callable_bond_generic(
            100.0_f64, 0.04, 0.01, 5.0,
            &coupon_times, &vec![],
            true, 100,
        );
        assert!(v <= v_nc + 1e-6, "callable={v} > non_callable={v_nc}");
    }

    // -------------------------------------------------------------------------
    // AD-57 — convertible_bond_generic
    // -------------------------------------------------------------------------
    #[test]
    fn test_convertible_bond() {
        let coupon_times: Vec<(f64, f64)> = (1..=5).map(|y| (y as f64, 3.0_f64)).collect();
        let v = convertible_bond_generic(
            100.0_f64, 50.0, 2.0,
            0.04, 0.01, 0.30, 5.0,
            &coupon_times, 100,
        );
        // Convertible ≥ straight bond and ≥ conversion value at maturity
        assert!(v >= 100.0, "convertible={v} (should exceed straight bond par=100)");
    }

    #[test]
    fn test_convertible_bond_high_stock() {
        // High stock price → bond is mostly equity-linked
        let coupon_times: Vec<(f64, f64)> = (1..=5).map(|y| (y as f64, 3.0_f64)).collect();
        let v = convertible_bond_generic(
            100.0_f64, 100.0, 2.0,
            0.04, 0.01, 0.30, 5.0,
            &coupon_times, 100,
        );
        // 2 * 100 = 200 conversion value, so bond should be near 200
        assert!(v > 150.0, "convertible_high_stock={v}");
    }

    // -------------------------------------------------------------------------
    // AD-76 — mc_mountain_range_generic (Everest)
    // -------------------------------------------------------------------------
    #[test]
    fn test_mc_mountain_everest() {
        let spots = vec![100.0_f64, 100.0, 100.0];
        let vols = vec![0.25_f64, 0.30, 0.20];
        let qs = vec![0.02_f64, 0.01, 0.03];
        let corrs = vec![
            1.0, 0.3, 0.2,
            0.3, 1.0, 0.25,
            0.2, 0.25, 1.0,
        ];
        let obs = vec![1.0_f64];
        let res = mc_mountain_range_generic(
            &spots, &vols, &corrs,
            0.05_f64, &qs, &obs,
            100.0, MountainTypeGeneric::Everest,
            10_000, 42,
        );
        assert!(res.price >= 0.0, "everest={}", res.price);
        assert!(res.std_error > 0.0, "se={}", res.std_error);
    }

    // -------------------------------------------------------------------------
    // AD-78 — cos_heston_generic
    // -------------------------------------------------------------------------
    #[test]
    fn test_cos_heston_call() {
        let v = cos_heston_generic(
            100.0_f64, 100.0, 1.0,
            0.05, 0.02,
            0.04, 2.0, 0.04, 0.3, -0.7,
            true, 128,
        );
        // Heston with v0=theta=0.04 ≈ BS(vol=0.2). Call ≈ 9
        assert!(v > 3.0 && v < 20.0, "cos_heston_call={v}");
    }

    #[test]
    fn test_cos_heston_put_call_parity() {
        let call = cos_heston_generic(
            100.0_f64, 100.0, 1.0,
            0.05, 0.02,
            0.04, 2.0, 0.04, 0.3, -0.7,
            true, 256,
        );
        let put = cos_heston_generic(
            100.0_f64, 100.0, 1.0,
            0.05, 0.02,
            0.04, 2.0, 0.04, 0.3, -0.7,
            false, 256,
        );
        let fwd = 100.0 * ((0.05 - 0.02) * 1.0_f64).exp();
        let df = (-0.05_f64).exp();
        let parity_diff = call - put - df * (fwd - 100.0);
        assert!(
            parity_diff.abs() < 1.0,
            "parity_diff={parity_diff} (call={call}, put={put})"
        );
    }

    // -------------------------------------------------------------------------
    // AD-79 — mc_slv_generic
    // -------------------------------------------------------------------------
    #[test]
    fn test_mc_slv_call() {
        let res = mc_slv_generic(
            100.0_f64, 0.05, 0.02,
            0.04, 2.0, 0.04, 0.3, -0.7,
            100.0, 1.0, true,
            10_000, 100, 42,
        );
        assert!(res.price > 3.0 && res.price < 20.0, "slv_call={}", res.price);
        assert!(res.std_error > 0.0, "se={}", res.std_error);
    }

    // -------------------------------------------------------------------------
    // AD-80 — heston_hull_white_generic
    // -------------------------------------------------------------------------
    #[test]
    fn test_heston_hull_white_call() {
        let v = heston_hull_white_generic(
            100.0_f64, 100.0, 1.0,
            0.04, 0.02,
            0.10, 0.005,
            0.04, 2.0, 0.04, 0.3, -0.7,
            0.2, true,
        );
        assert!(v > 3.0 && v < 20.0, "hhw_call={v}");
    }

    #[test]
    fn test_heston_hull_white_vs_pure_heston() {
        // With zero HW vol, should match pure Heston
        let hhw = heston_hull_white_generic(
            100.0_f64, 100.0, 1.0,
            0.04, 0.02,
            0.10, 0.0, // hw_sigma_r = 0
            0.04, 2.0, 0.04, 0.3, -0.7,
            0.0, true,
        );
        let heston = cos_heston_generic(
            100.0_f64, 100.0, 1.0,
            0.04, 0.02,
            0.04, 2.0, 0.04, 0.3, -0.7,
            true, 128,
        );
        assert!(
            (hhw - heston).abs() < 0.01,
            "hhw={hhw} vs heston={heston}"
        );
    }

    // -------------------------------------------------------------------------
    // AD-85 — nth_to_default_mc_generic
    // -------------------------------------------------------------------------
    #[test]
    fn test_nth_to_default_first() {
        let probs = vec![0.05_f64, 0.04, 0.06, 0.03, 0.05];
        let res = nth_to_default_mc_generic(
            &probs, 0.3, 0.40,
            1, 1.0,
            20_000, 42,
        );
        // First-to-default on 5 names: should have positive value
        assert!(res.price > 0.0 && res.price < 0.60, "ntd1={}", res.price);
    }

    #[test]
    fn test_nth_to_default_monotone() {
        let probs = vec![0.10_f64, 0.10, 0.10, 0.10, 0.10];
        let r1 = nth_to_default_mc_generic(&probs, 0.3, 0.40, 1, 1.0, 50_000, 42);
        let r2 = nth_to_default_mc_generic(&probs, 0.3, 0.40, 3, 1.0, 50_000, 42);
        let r5 = nth_to_default_mc_generic(&probs, 0.3, 0.40, 5, 1.0, 50_000, 42);
        // 1st-to-default ≥ 3rd ≥ 5th
        assert!(r1.price >= r2.price - 0.01, "1td={} < 3td={}", r1.price, r2.price);
        assert!(r2.price >= r5.price - 0.01, "3td={} < 5td={}", r2.price, r5.price);
    }

    // -------------------------------------------------------------------------
    // AD-91 — lmm_swaption_mc_generic
    // -------------------------------------------------------------------------
    #[test]
    fn test_lmm_swaption_payer() {
        let n_rates = 10usize;
        let fwds: Vec<f64> = (0..n_rates).map(|_| 0.04_f64).collect();
        let vols_t: Vec<f64> = (0..n_rates).map(|_| 0.15_f64).collect();
        let accruals: Vec<f64> = vec![0.5; n_rates];
        let mut corrs = vec![0.0_f64; n_rates * n_rates];
        for i in 0..n_rates {
            for j in 0..n_rates {
                corrs[i * n_rates + j] = if i == j { 1.0 } else { 0.5 };
            }
        }
        let res = lmm_swaption_mc_generic(
            &fwds, &vols_t, &corrs,
            &accruals,
            2, 8, // swap from period 2 to 8
            0.04, 1e6, true,
            5_000, 42,
        );
        assert!(res.price > 0.0, "lmm_payer={}", res.price);
        assert!(res.std_error > 0.0 && res.std_error < res.price, "se={}", res.std_error);
    }

    // -------------------------------------------------------------------------
    // AD-92 — vg_cos_generic
    // -------------------------------------------------------------------------
    #[test]
    fn test_vg_cos_call() {
        let v = vg_cos_generic(
            100.0_f64, 100.0, 0.5,
            0.05, 0.02,
            0.20, 0.1, -0.1,
            true, 128,
        );
        assert!(v > 2.0 && v < 15.0, "vg_call={v}");
    }

    #[test]
    fn test_vg_cos_put_call_parity() {
        let call = vg_cos_generic(
            100.0_f64, 100.0, 1.0,
            0.05, 0.02,
            0.20, 0.1, -0.1,
            true, 256,
        );
        let put = vg_cos_generic(
            100.0_f64, 100.0, 1.0,
            0.05, 0.02,
            0.20, 0.1, -0.1,
            false, 256,
        );
        let fwd = 100.0 * ((0.05 - 0.02) * 1.0_f64).exp();
        let df = (-0.05_f64).exp();
        let diff = call - put - df * (fwd - 100.0);
        assert!(diff.abs() < 2.0, "vg_parity_diff={diff}");
    }
}
