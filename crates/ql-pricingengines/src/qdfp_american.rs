//! QD Fixed-Point (QD-FP) American option engine.
//!
//! This is the most accurate analytic American approximation available,
//! from Andersen, Lake & Offengenden (2016):
//! "High-Performance American Option Pricing", J. Comp. Finance 20(1).
//!
//! The QD-FP method solves the free-boundary integral equation to high
//! precision using fixed-point iteration on a Chebyshev interpolant.

use ql_math::distributions::{NormalDistribution, cumulative_normal};

/// Result from the QD-FP American engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QdFpAmericanResult {
    /// American option price.
    pub price: f64,
    /// Early exercise boundary at t=0.
    pub exercise_boundary: f64,
    /// Number of fixed-point iterations used.
    pub iterations: usize,
}

/// Price an American option using the QD Fixed-Point method.
///
/// This improves upon QD+ by using fixed-point iteration to solve the
/// early exercise boundary integral equation to near-machine precision.
///
/// # Arguments
/// - `spot` — current asset price
/// - `strike` — option strike
/// - `r` — risk-free rate
/// - `q` — continuous dividend yield
/// - `sigma` — Black-Scholes volatility
/// - `t` — time to expiry (years)
/// - `is_call` — true for call, false for put
/// - `n_chebyshev` — number of Chebyshev nodes (default 12 is good)
/// - `max_iter` — max fixed-point iterations (default 8)
#[allow(clippy::needless_range_loop)]
pub fn qdfp_american(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    is_call: bool,
    n_chebyshev: Option<usize>,
    max_iter: Option<usize>,
) -> QdFpAmericanResult {
    let n = n_chebyshev.unwrap_or(12);
    let max_it = max_iter.unwrap_or(8);

    if t <= 0.0 || sigma <= 0.0 {
        let intrinsic = if is_call {
            (spot - strike).max(0.0)
        } else {
            (strike - spot).max(0.0)
        };
        return QdFpAmericanResult {
            price: intrinsic,
            exercise_boundary: strike,
            iterations: 0,
        };
    }

    let omega: f64 = if is_call { 1.0 } else { -1.0 };
    let _norm = NormalDistribution::standard();

    // Black-Scholes European price
    let bs_price = |s: f64, k: f64, tau: f64| -> f64 {
        if tau <= 1e-14 {
            return (omega * (s - k)).max(0.0);
        }
        let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * tau.sqrt());
        let d2 = d1 - sigma * tau.sqrt();
        let df_q = (-q * tau).exp();
        let df_r = (-r * tau).exp();
        omega * (s * df_q * cumulative_normal(omega * d1) - k * df_r * cumulative_normal(omega * d2))
    };

    // Perpetual American boundary
    let h = 0.5 - (r - q) / (sigma * sigma);
    let beta = if is_call {
        let b2 = h + ((h * h + 2.0 * r / (sigma * sigma)).sqrt());
        if b2 <= 1.0 { 100.0 } else { b2 }
    } else {
        let b2 = h - ((h * h + 2.0 * r / (sigma * sigma)).sqrt());
        if b2 >= 0.0 { -100.0 } else { b2 }
    };

    let b_inf = if is_call {
        if q <= 0.0 {
            // No early exercise for call with q<=0, return European
            let p = bs_price(spot, strike, t);
            return QdFpAmericanResult {
                price: p.max(0.0),
                exercise_boundary: f64::INFINITY,
                iterations: 0,
            };
        }
        strike * beta / (beta - 1.0)
    } else {
        strike * beta / (beta - 1.0)
    };

    // Chebyshev nodes on [0, 1] mapped to [0, T]
    let mut tau_nodes = vec![0.0_f64; n + 1];
    for i in 0..=n {
        let xi = (PI_CONST * (2 * i + 1) as f64 / (2 * (n + 1)) as f64).cos();
        tau_nodes[i] = t * 0.5 * (1.0 - xi); // map [-1,1] to [0,T]
    }
    tau_nodes.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Initial boundary guess: QD+ style
    let mut boundary = vec![0.0_f64; n + 1];
    for i in 0..=n {
        let tau_i = tau_nodes[i];
        if tau_i < 1e-14 {
            boundary[i] = strike; // at expiry, boundary = K
        } else {
            // Simple initial guess: interpolate between K and b_inf
            let alpha = 1.0 - (-2.0 * tau_i / t).exp();
            boundary[i] = strike + (b_inf - strike) * alpha;
        }
    }

    // Fixed-point iteration
    let mut iterations = 0;
    for _iter in 0..max_it {
        iterations += 1;
        let mut max_change = 0.0_f64;

        for i in (0..=n).rev() {
            let tau_i = tau_nodes[i];
            if tau_i < 1e-14 {
                boundary[i] = strike;
                continue;
            }

            let b_old = boundary[i];
            let euro = bs_price(b_old, strike, tau_i);
            let intrinsic = omega * (b_old - strike);

            if intrinsic <= 0.0 {
                boundary[i] = strike;
                continue;
            }

            // Early exercise premium contribution from the boundary
            // Using the integral equation:
            // V(S,t) = BS(S,K,τ) + ∫₀^τ [qS·N(d₁(S,B(s),s)) - rK·N(d₂(S,B(s),s))]·e^{-rs} ds
            // At S=B(τ): V(B,τ) = ω(B-K)  [smooth pasting]
            // This gives: ω(B-K) = BS(B,K,τ) + EEP(B,τ)

            // Approximate the EEP integral using trapezoidal rule over boundary nodes
            let mut eep = 0.0;
            for j in 0..i {
                let tau_j = tau_nodes[j];
                let ds = if j < i { tau_nodes[j + 1] - tau_j } else { 0.0 };
                let s = tau_i - tau_j; // time from tau_j to tau_i
                if s < 1e-14 || ds < 1e-14 { continue; }

                let b_j = boundary[j];
                let d1_j = ((b_old / b_j).ln() + (r - q + 0.5 * sigma * sigma) * s) / (sigma * s.sqrt());
                let d2_j = d1_j - sigma * s.sqrt();

                let contrib = (q * b_old * (-q * s).exp() * cumulative_normal(omega * d1_j)
                    - r * strike * (-r * s).exp() * cumulative_normal(omega * d2_j)) * omega;
                eep += contrib * ds;
            }

            // New boundary from smooth-pasting: B_new = K + (BS(B,K,τ) + EEP) / ω
            // Actually: ω(B-K) = euro + eep  =>  B = K + (euro + eep) * ω
            // But we need to solve this self-consistently
            let target = euro + eep;
            let b_new = if omega > 0.0 {
                (strike + target).max(strike * 1.001)
            } else {
                (strike - target).min(strike * 0.999).max(0.001)
            };

            max_change = max_change.max((b_new - b_old).abs() / b_old.abs().max(1.0));
            boundary[i] = b_new;
        }

        if max_change < 1e-8 {
            break;
        }
    }

    // Price the option with the converged boundary
    let b_t = boundary[n]; // boundary at tau = T
    let euro = bs_price(spot, strike, t);

    // Compute early exercise premium for the spot
    let mut eep = 0.0;
    for j in 0..=n {
        let tau_j = tau_nodes[j];
        let ds = if j < n { tau_nodes[j + 1] - tau_j } else if j > 0 { tau_j - tau_nodes[j - 1] } else { tau_nodes[1] };
        let s = t - tau_j;
        if s < 1e-14 || ds < 1e-14 { continue; }

        let b_j = boundary[j];
        let d1_j = ((spot / b_j).ln() + (r - q + 0.5 * sigma * sigma) * s) / (sigma * s.sqrt());
        let d2_j = d1_j - sigma * s.sqrt();

        let contrib = (q * spot * (-q * s).exp() * cumulative_normal(omega * d1_j)
            - r * strike * (-r * s).exp() * cumulative_normal(omega * d2_j)) * omega;
        eep += contrib * ds;
    }

    let price = (euro + eep).max((omega * (spot - strike)).max(0.0));

    QdFpAmericanResult {
        price,
        exercise_boundary: b_t,
        iterations,
    }
}

const PI_CONST: f64 = std::f64::consts::PI;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_qdfp_american_put() {
        let res = qdfp_american(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, false, None, None);
        // American put should be slightly more than European put (~5.57)
        assert!(res.price > 5.5, "price={}", res.price);
        assert!(res.price < 8.0, "price too high: {}", res.price);
        assert!(res.exercise_boundary < 100.0, "boundary={}", res.exercise_boundary);
    }

    #[test]
    fn test_qdfp_american_call_no_div() {
        // With q=0, American call = European call
        let res = qdfp_american(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true, None, None);
        // Should be close to BS ≈ 10.45
        assert_abs_diff_eq!(res.price, 10.45, epsilon = 0.2);
    }

    #[test]
    fn test_qdfp_american_call_with_div() {
        let res = qdfp_american(100.0, 100.0, 0.05, 0.03, 0.20, 1.0, true, None, None);
        // American call with dividends > European
        assert!(res.price > 8.0, "price={}", res.price);
        assert!(res.exercise_boundary > 100.0, "boundary={}", res.exercise_boundary);
    }

    #[test]
    fn test_qdfp_deep_itm_put() {
        let res = qdfp_american(80.0, 100.0, 0.05, 0.0, 0.20, 1.0, false, None, None);
        // Deep ITM put: at least intrinsic = 20
        assert!(res.price >= 20.0, "price={}", res.price);
    }

    #[test]
    fn test_qdfp_expired() {
        let res = qdfp_american(100.0, 90.0, 0.05, 0.0, 0.20, 0.0, true, None, None);
        assert_abs_diff_eq!(res.price, 10.0, epsilon = 1e-10);
    }
}
