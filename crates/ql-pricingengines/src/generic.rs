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
use ql_math::generic::{normal_cdf, normal_pdf, black_scholes_generic, discount_factor};

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
        let rhs = omega * (s_star - strike) - euro;
        let lhs = a_coeff;

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
        s_star = s_star - step;

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
        price = price + w * bs;

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
        pv = pv + T::from_f64(amt) * discount_factor(rate, T::from_f64(ti));
    }

    // Notional repayment at maturity
    pv = pv + T::from_f64(notional) * discount_factor(rate, T::from_f64(maturity));

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
        pv = pv + T::from_f64(amt) * discount_factor(rate, T::from_f64(ti));
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
    let half = T::half();

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
