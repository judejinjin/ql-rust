//! Gaussian 1-factor (G1D) swaption & European-exercise pricing engine.
//!
//! This engine works with any `Gsr1d` model (which includes Hull-White as
//! a special case with constant parameters). It prices European swaptions
//! by conditioning on the model state at option expiry and performing
//! numerical integration (Gauss-Hermite quadrature).
//!
//! ## Method
//!
//! For a European payer swaption with expiry $T_e$ and swap from $T_s$ to
//! $T_n$ at fixed rate $K$:
//!
//! $$V = \int_{-\infty}^{\infty} \max\left(\text{swap}(x), 0\right)
//!       \cdot \frac{1}{\sqrt{2\pi\zeta}} \exp\left(-\frac{x^2}{2\zeta}\right) \, dx$$
//!
//! where $\zeta = \zeta(T_e)$ and the swap value at state $x$ is:
//!
//! $$\text{swap}(x) = N \left[\sum_i \tau_i\,(K - S(x))\,P(T_e, T_i|x)\right]$$
//!
//! We evaluate this via Gauss-Hermite quadrature with change of variable
//! $x = \sqrt{2\zeta}\, u$.

use ql_models::gsr::Gsr1d;

/// Result of a Gaussian 1-factor engine computation.
#[derive(Debug, Clone)]
pub struct Gaussian1dResult {
    /// Net present value.
    pub npv: f64,
}

/// Price a European swaption using Gauss-Hermite quadrature under the
/// GSR / Gaussian 1-factor model.
///
/// # Parameters
/// - `model` — calibrated Gsr1d model (with initial curve and parameters)
/// - `option_expiry` — time to swaption expiry in years
/// - `swap_tenors` — payment times of the swap fixed leg (in years)
/// - `year_fractions` — day-count fractions for each swap period
/// - `fixed_rate` — the strike rate of the swaption
/// - `notional` — swap notional
/// - `is_payer` — `true` for payer swaption, `false` for receiver
/// - `n_quad` — number of Gauss-Hermite quadrature points (16–64 typical)
#[allow(clippy::too_many_arguments)]
pub fn gaussian1d_swaption(
    model: &Gsr1d,
    option_expiry: f64,
    swap_tenors: &[f64],
    year_fractions: &[f64],
    fixed_rate: f64,
    notional: f64,
    is_payer: bool,
    n_quad: usize,
) -> Gaussian1dResult {
    if swap_tenors.is_empty() || year_fractions.is_empty() {
        return Gaussian1dResult { npv: 0.0 };
    }

    let n = swap_tenors.len().min(year_fractions.len());
    let zeta = model.zeta(option_expiry);
    let sqrt_zeta = zeta.max(1e-30).sqrt();
    let pm_te = model.market_discount(option_expiry);

    // Gauss-Hermite nodes and weights
    let (nodes, weights) = gauss_hermite(n_quad);

    let sign = if is_payer { 1.0 } else { -1.0 };
    let mut integral = 0.0;

    for (node, weight) in nodes.iter().zip(weights.iter()) {
        // Change of variable: x = sqrt(2 * zeta) * u
        let x = (2.0_f64).sqrt() * sqrt_zeta * node;

        // Compute swap value at this state
        let swap_start_bond = if option_expiry < swap_tenors[0] {
            model.zero_bond(option_expiry, swap_tenors[0], x)
        } else {
            1.0
        };
        let swap_end_bond = model.zero_bond(option_expiry, swap_tenors[n - 1], x);

        let mut annuity = 0.0;
        for j in 0..n {
            annuity += year_fractions[j] * model.zero_bond(option_expiry, swap_tenors[j], x);
        }

        let swap_value = sign * notional * (swap_start_bond - swap_end_bond - fixed_rate * annuity);

        // Payout under conditional measure, weighted by Gauss-Hermite weight
        // The exp(-u^2) factor is already in the GH weight
        let payoff = swap_value.max(0.0);
        integral += weight * payoff;
    }

    // Normalise: GH quadrature integrates f(u) * exp(-u^2),
    // we need (1/sqrt(pi)) * integral * P^M(0, Te)
    let npv = pm_te * integral / std::f64::consts::PI.sqrt();

    Gaussian1dResult { npv }
}

/// Price a zero-coupon bond option in the GSR / G1D model.
///
/// Call: max(P(Te, Tb) - K, 0)  or  Put: max(K - P(Te, Tb), 0).
pub fn gaussian1d_zcb_option(
    model: &Gsr1d,
    option_expiry: f64,
    bond_maturity: f64,
    strike: f64,
    is_call: bool,
    n_quad: usize,
) -> Gaussian1dResult {
    let zeta = model.zeta(option_expiry);
    let sqrt_zeta = zeta.max(1e-30).sqrt();
    let pm_te = model.market_discount(option_expiry);

    let (nodes, weights) = gauss_hermite(n_quad);
    let sign = if is_call { 1.0 } else { -1.0 };

    let mut integral = 0.0;
    for (node, weight) in nodes.iter().zip(weights.iter()) {
        let x = (2.0_f64).sqrt() * sqrt_zeta * node;
        let bond = model.zero_bond(option_expiry, bond_maturity, x);
        let payoff = (sign * (bond - strike)).max(0.0);
        integral += weight * payoff;
    }

    let npv = pm_te * integral / std::f64::consts::PI.sqrt();
    Gaussian1dResult { npv }
}

// ── Gauss-Hermite quadrature via Golub-Welsch ────────────

/// Compute Gauss-Hermite quadrature nodes and weights for n points.
///
/// Uses the Golub-Welsch algorithm: eigenvalue decomposition of the
/// tridiagonal Jacobi matrix for physicists' Hermite polynomials
/// with weight function exp(-x²).
///
/// The Jacobi matrix J has:
///   J[i,i] = 0
///   J[i,i+1] = J[i+1,i] = sqrt((i+1) / 2)
///
/// Eigenvalues → nodes, weights = sqrt(π) · v[0]² for each eigenvector v.
fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    use nalgebra::DMatrix;

    assert!(n >= 2, "need at least 2 quadrature points");

    let mut j = DMatrix::<f64>::zeros(n, n);
    for i in 0..n - 1 {
        let off = ((i + 1) as f64 / 2.0).sqrt();
        j[(i, i + 1)] = off;
        j[(i + 1, i)] = off;
    }

    // Symmetric eigenvalue decomposition
    let eig = j.symmetric_eigen();
    let eigenvalues = eig.eigenvalues;
    let eigenvectors = eig.eigenvectors;

    let sqrt_pi = std::f64::consts::PI.sqrt();

    // Collect (node, weight) pairs and sort by node
    let mut pairs: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let node = eigenvalues[i];
            let v0 = eigenvectors[(0, i)];
            let weight = sqrt_pi * v0 * v0;
            (node, weight)
        })
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let nodes = pairs.iter().map(|p| p.0).collect();
    let weights = pairs.iter().map(|p| p.1).collect();
    (nodes, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Create a simple flat-rate GSR model (constant a, sigma).
    fn flat_gsr(a: f64, sigma: f64, flat_rate: f64) -> Gsr1d {
        // Build initial curve: flat discount factors at annual grid out to 40Y
        let mut curve = Vec::new();
        curve.push((0.0, 1.0));
        for y in 1..=40 {
            let t = y as f64;
            curve.push((t, (-flat_rate * t).exp()));
        }
        Gsr1d::new(
            vec![],             // no breakpoints → constant params
            vec![a],
            vec![sigma],
            curve,
        )
    }

    #[test]
    fn zcb_option_atm_has_value() {
        let model = flat_gsr(0.05, 0.01, 0.04);
        let te = 1.0;
        let tb = 10.0;
        // ATM strike = forward ZCB price
        let fwd_price = model.market_discount(tb) / model.market_discount(te);
        let call = gaussian1d_zcb_option(&model, te, tb, fwd_price, true, 32);
        assert!(call.npv > 0.0, "ATM call should have positive value");
    }

    #[test]
    fn zcb_put_call_parity() {
        let model = flat_gsr(0.05, 0.01, 0.04);
        let te = 2.0;
        let tb = 7.0;
        let k = 0.80;
        let call = gaussian1d_zcb_option(&model, te, tb, k, true, 32);
        let put = gaussian1d_zcb_option(&model, te, tb, k, false, 32);

        // Put-call parity: C - P = P(0,Te)*[Fwd - K]
        let fwd = model.market_discount(tb) / model.market_discount(te);
        let pm_te = model.market_discount(te);
        let parity_lhs = call.npv - put.npv;
        let parity_rhs = pm_te * (fwd - k);

        assert_abs_diff_eq!(parity_lhs, parity_rhs, epsilon = 1e-6);
    }

    #[test]
    fn payer_swaption_positive_for_low_strike() {
        let model = flat_gsr(0.03, 0.008, 0.05);
        let tenors: Vec<f64> = (1..=10).map(|y| y as f64 + 1.0).collect(); // 2Y - 11Y
        let yf: Vec<f64> = vec![1.0; 10]; // annual fractions
        let low_strike = 0.01; // way below par rate

        let result = gaussian1d_swaption(
            &model, 1.0, &tenors, &yf, low_strike, 1_000_000.0, true, 32,
        );
        assert!(result.npv > 0.0, "Payer swaption at low strike should have large value");
    }

    #[test]
    fn receiver_swaption_positive_for_high_strike() {
        let model = flat_gsr(0.03, 0.008, 0.03);
        let tenors: Vec<f64> = (1..=5).map(|y| y as f64 + 1.0).collect();
        let yf = vec![1.0; 5];
        let high_strike = 0.10;

        let result = gaussian1d_swaption(
            &model, 1.0, &tenors, &yf, high_strike, 1_000_000.0, false, 32,
        );
        assert!(result.npv > 0.0, "Receiver swaption at high strike should have positive value");
    }

    #[test]
    fn gh_quadrature_integrates_constant() {
        // ∫ exp(-x^2) dx = sqrt(π) → our weights should sum to sqrt(π)
        let (_, weights) = gauss_hermite(16);
        let sum: f64 = weights.iter().sum();
        assert_abs_diff_eq!(sum, std::f64::consts::PI.sqrt(), epsilon = 1e-8);
    }

    #[test]
    fn gh_quadrature_32_integrates_polynomial() {
        // ∫ x^2 exp(-x^2) dx = sqrt(π)/2
        let (nodes, weights) = gauss_hermite(32);
        let integral: f64 = nodes.iter().zip(weights.iter())
            .map(|(x, w)| w * x * x)
            .sum();
        let expected = std::f64::consts::PI.sqrt() / 2.0;
        assert_abs_diff_eq!(integral, expected, epsilon = 1e-6);
    }
}
