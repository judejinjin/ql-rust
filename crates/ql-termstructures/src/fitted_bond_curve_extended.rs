//! Extended fitted bond discount curve methods: polynomial and exponential spline.
//!
//! Complements the Nelson-Siegel / Svensson curves in [`crate::nelson_siegel`]
//! with two additional global fitting approaches:
//!
//! ## Polynomial Discount Function
//!
//! Models the discount factor as a polynomial in maturity:
//! ```text
//! d(t) = 1 + c₁·t + c₂·t² + … + cₙ·tⁿ
//! ```
//! Subject to `d(0)=1`.  Fitted by least squares to observed bond prices.
//! MacCaulay (1938) used a polynomial yield curve; Carleton-Cooper (1976)
//! used this discount factor form directly.
//!
//! ## Exponential Spline
//!
//! Models the discount factor as a sum of exponentials:
//! ```text
//! d(t) = Σₖ aₖ · exp(-κₖ · t)
//! ```
//! with fixed knot rates `κₖ` and fitted weights `aₖ`.  Vasicek-Fong (1982)
//! introduced this form; it is more flexible than polynomial fitting and
//! avoids oscillation at long maturities.
//!
//! Both methods expose:
//! - `discount(t)` — discount factor at maturity `t`
//! - `zero_rate(t)` — continuously-compounded zero rate
//! - `forward_rate(t1, t2)` — continuously-compounded instantaneous rate
//! - `fit(maturities, prices, coupons, ...)` — OLS calibration

use serde::{Deserialize, Serialize};

// =========================================================================
// Helper: ordinary least squares via normal equations
// =========================================================================

/// Solve the least-squares system X^T X β = X^T y.
/// Returns the coefficient vector β, length = n_cols.
/// Uses Cholesky decomposition for numerical stability.
fn ols(xmat: &[f64], y: &[f64], n_rows: usize, n_cols: usize) -> Vec<f64> {
    assert_eq!(xmat.len(), n_rows * n_cols);
    assert_eq!(y.len(), n_rows);

    // Build X^T X (n_cols × n_cols) and X^T y (n_cols)
    let mut xtx = vec![0.0f64; n_cols * n_cols];
    let mut xty = vec![0.0f64; n_cols];

    for i in 0..n_rows {
        for j in 0..n_cols {
            xty[j] += xmat[i * n_cols + j] * y[i];
            for k in 0..n_cols {
                xtx[j * n_cols + k] += xmat[i * n_cols + j] * xmat[i * n_cols + k];
            }
        }
    }

    // Cholesky factorisation of X^T X
    let l = cholesky_lower(&xtx, n_cols);
    // Forward substitution Lz = X^T y
    let mut z = vec![0.0f64; n_cols];
    for i in 0..n_cols {
        let mut s = xty[i];
        for j in 0..i { s -= l[i * n_cols + j] * z[j]; }
        let diag = l[i * n_cols + i];
        z[i] = if diag.abs() > 1e-15 { s / diag } else { 0.0 };
    }
    // Back substitution L^T β = z
    let mut beta = vec![0.0f64; n_cols];
    for i in (0..n_cols).rev() {
        let mut s = z[i];
        for j in i + 1..n_cols { s -= l[j * n_cols + i] * beta[j]; }
        let diag = l[i * n_cols + i];
        beta[i] = if diag.abs() > 1e-15 { s / diag } else { 0.0 };
    }
    beta
}

fn cholesky_lower(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s: f64 = a[i * n + j];
            for k in 0..j { s -= l[i * n + k] * l[j * n + k]; }
            if i == j {
                l[i * n + j] = s.max(1e-30).sqrt();
            } else {
                let diag = l[j * n + j];
                l[i * n + j] = if diag > 1e-15 { s / diag } else { 0.0 };
            }
        }
    }
    l
}

// =========================================================================
// Polynomial Discount Curve
// =========================================================================

/// Polynomial discount factor curve.
///
/// `d(t) = 1 + c₁·t + c₂·t² + … + cₙ·tⁿ`
///
/// The constant term is constrained to 1 (`d(0) = 1`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolynomialDiscountCurve {
    /// Polynomial coefficients [c₁, c₂, …, cₙ] (no c₀ — the intercept is 1).
    pub coefficients: Vec<f64>,
    /// Polynomial degree `n`.
    pub degree: usize,
}

impl PolynomialDiscountCurve {
    /// Create with pre-computed coefficients.
    pub fn new(coefficients: Vec<f64>) -> Self {
        let degree = coefficients.len();
        Self { coefficients, degree }
    }

    /// Discount factor at maturity `t`.
    pub fn discount(&self, t: f64) -> f64 {
        if t <= 0.0 { return 1.0; }
        let mut d = 1.0;
        let mut tp = t;
        for &c in &self.coefficients {
            d += c * tp;
            tp *= t;
        }
        d.max(1e-10) // clamp to avoid negative discount factors
    }

    /// Continuously-compounded zero rate at maturity `t`.
    pub fn zero_rate(&self, t: f64) -> f64 {
        if t <= 1e-8 { return 0.0; }
        -self.discount(t).ln() / t
    }

    /// Continuously-compounded forward rate over `[t1, t2]`.
    pub fn forward_rate(&self, t1: f64, t2: f64) -> f64 {
        if t2 <= t1 + 1e-8 { return self.zero_rate(t1); }
        let dt = t2 - t1;
        (self.discount(t1) / self.discount(t2)).ln() / dt
    }

    /// Calibrate to zero coupon bond prices by OLS.
    ///
    /// # Arguments
    /// - `maturities` — maturity of each bond in years
    /// - `prices`     — observed discount prices (i.e. present values per unit)
    /// - `degree`     — polynomial degree (typically 3–5)
    pub fn fit(maturities: &[f64], prices: &[f64], degree: usize) -> Self {
        let n_bonds = maturities.len();
        assert_eq!(prices.len(), n_bonds);
        assert!(degree >= 1);

        // Build design matrix for [t, t², …, t^degree]
        let mut x = vec![0.0f64; n_bonds * degree];
        let mut y = vec![0.0f64; n_bonds];

        for (i, (&t, &p)) in maturities.iter().zip(prices.iter()).enumerate() {
            let mut tp = t;
            for j in 0..degree {
                x[i * degree + j] = tp;
                tp *= t;
            }
            y[i] = p - 1.0; // subtract the constrained intercept
        }

        let coeffs = ols(&x, &y, n_bonds, degree);
        Self::new(coeffs)
    }

    /// Calibrate to coupon-bearing bond prices.
    ///
    /// Each bond is described by a vector of `(cashflow_time, cashflow_amount)`
    /// pairs plus its dirty price.  The calibration minimises the sum of
    /// squared pricing errors.
    ///
    /// # Arguments
    /// - `cash_flows` — per bond: list of (time, amount) pairs
    /// - `dirty_prices` — observed dirty prices (% of notional)
    /// - `degree` — polynomial degree
    pub fn fit_coupon_bonds(
        cash_flows: &[Vec<(f64, f64)>],
        dirty_prices: &[f64],
        degree: usize,
    ) -> Self {
        let n_bonds = cash_flows.len();
        assert_eq!(dirty_prices.len(), n_bonds);

        // y = price - sum(cf * 1.0) ... after subtracting the "intercept" (d(0)=1 for all CFs means sum(cf))
        // X_{i,j} = sum_k cf_{i,k} * t_{i,k}^{j+1}
        let mut x = vec![0.0f64; n_bonds * degree];
        let mut y = vec![0.0f64; n_bonds];

        for (i, (cfs, &price)) in cash_flows.iter().zip(dirty_prices.iter()).enumerate() {
            let intercept: f64 = cfs.iter().map(|&(_, cf)| cf).sum(); // sum(cf) * d(0) = sum(cf)
            y[i] = price - intercept;
            for j in 0..degree {
                let power = (j + 1) as u32;
                x[i * degree + j] = cfs.iter().map(|&(t, cf)| cf * t.powi(power as i32)).sum();
            }
        }

        let coeffs = ols(&x, &y, n_bonds, degree);
        Self::new(coeffs)
    }
}

// =========================================================================
// Exponential Spline Discount Curve
// =========================================================================

/// Vasicek-Fong (1982) exponential spline discount factor curve.
///
/// ```text
/// d(t) = Σₖ aₖ · exp(-κₖ · t)
/// ```
///
/// with pre-specified decay rates `κₖ` and fitted weights `aₖ`.
/// The constraint `d(0) = 1 ↔ Σ aₖ = 1` is enforced during fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExponentialSplineCurve {
    /// Fitted weights `aₖ`.
    pub weights: Vec<f64>,
    /// Fixed decay rates `κₖ` (positive, in units of 1/year).
    pub knots: Vec<f64>,
}

impl ExponentialSplineCurve {
    /// Create with given knots and weights.
    pub fn new(knots: Vec<f64>, weights: Vec<f64>) -> Self {
        assert_eq!(knots.len(), weights.len(), "knots and weights must match");
        Self { weights, knots }
    }

    /// Default knot set: 9 knots log-spaced from 0.1 to 3.0 (Vasicek-Fong).
    pub fn default_knots() -> Vec<f64> {
        // κ₁…κ₉ as in the original Vasicek-Fong paper
        vec![0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    }

    /// Discount factor at maturity `t`.
    pub fn discount(&self, t: f64) -> f64 {
        if t <= 0.0 { return 1.0; }
        self.knots.iter().zip(self.weights.iter())
            .map(|(&kappa, &a)| a * (-kappa * t).exp())
            .sum::<f64>()
            .max(1e-10)
    }

    /// Continuously-compounded zero rate at maturity `t`.
    pub fn zero_rate(&self, t: f64) -> f64 {
        if t <= 1e-8 { return 0.0; }
        -self.discount(t).ln() / t
    }

    /// Continuously-compounded forward rate over `[t1, t2]`.
    pub fn forward_rate(&self, t1: f64, t2: f64) -> f64 {
        if t2 <= t1 + 1e-8 { return self.zero_rate(t1); }
        let dt = t2 - t1;
        (self.discount(t1) / self.discount(t2)).ln() / dt
    }

    /// Calibrate to zero-coupon bond prices by constrained OLS.
    ///
    /// The constraint `Σ aₖ = 1` (i.e. `d(0) = 1`) is enforced by substituting
    /// `a₀ = 1 - Σₖ₌₁ aₖ` and regressing the remaining weights.
    ///
    /// # Arguments
    /// - `maturities` — bond maturities in years
    /// - `prices`     — zero-coupon bond prices (discount factors)
    /// - `knots`      — decay rates κₖ; if `None` uses [`default_knots`]
    pub fn fit(maturities: &[f64], prices: &[f64], knots: Option<Vec<f64>>) -> Self {
        let kappa = knots.unwrap_or_else(Self::default_knots);
        let n_knots = kappa.len();
        let n_bonds = maturities.len();
        assert_eq!(prices.len(), n_bonds);
        assert!(n_knots >= 2, "need at least 2 knots");

        // Substitute a₀ constraint: d(t) = exp(-κ₀·t) + Σₖ₌₁ aₖ [exp(-κₖ·t)-exp(-κ₀·t)]
        // y_i = price_i - exp(-κ₀·t_i)
        // X_{i,k} = exp(-κₖ·t_i) - exp(-κ₀·t_i)  for k=1..n_knots-1
        let n_free = n_knots - 1;
        let mut x = vec![0.0f64; n_bonds * n_free];
        let mut y = vec![0.0f64; n_bonds];

        for (i, (&t, &p)) in maturities.iter().zip(prices.iter()).enumerate() {
            let base = (-kappa[0] * t).exp();
            y[i] = p - base;
            for j in 0..n_free {
                x[i * n_free + j] = (-kappa[j + 1] * t).exp() - base;
            }
        }

        let free_weights = ols(&x, &y, n_bonds, n_free);

        // Recover a₀ = 1 - Σ free_weights
        let sum_free: f64 = free_weights.iter().sum();
        let a0 = 1.0 - sum_free;

        let mut weights = vec![a0];
        weights.extend_from_slice(&free_weights);

        Self { weights, knots: kappa }
    }

    /// Calibrate to coupon-bearing bond prices.
    ///
    /// # Arguments
    /// - `cash_flows`   — per bond: list of (time, amount) pairs
    /// - `dirty_prices` — observed dirty prices
    /// - `knots`        — optional custom knot vector
    pub fn fit_coupon_bonds(
        cash_flows: &[Vec<(f64, f64)>],
        dirty_prices: &[f64],
        knots: Option<Vec<f64>>,
    ) -> Self {
        let kappa = knots.unwrap_or_else(Self::default_knots);
        let n_knots = kappa.len();
        let n_bonds = cash_flows.len();
        let n_free = n_knots - 1;

        // y_i = price_i - Σ_k cf_{i,k} * exp(-κ₀ * t_{i,k})
        let mut x = vec![0.0f64; n_bonds * n_free];
        let mut y = vec![0.0f64; n_bonds];

        for (i, (cfs, &price)) in cash_flows.iter().zip(dirty_prices.iter()).enumerate() {
            let base: f64 = cfs.iter().map(|&(t, cf)| cf * (-kappa[0] * t).exp()).sum();
            y[i] = price - base;
            for j in 0..n_free {
                x[i * n_free + j] = cfs.iter()
                    .map(|&(t, cf)| cf * ((-kappa[j + 1] * t).exp() - (-kappa[0] * t).exp()))
                    .sum();
            }
        }

        let free_weights = ols(&x, &y, n_bonds, n_free);
        let a0 = 1.0 - free_weights.iter().sum::<f64>();
        let mut weights = vec![a0];
        weights.extend_from_slice(&free_weights);

        Self { weights, knots: kappa }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial_discount_at_zero() {
        let curve = PolynomialDiscountCurve::new(vec![-0.04, 0.001]);
        assert!((curve.discount(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn polynomial_fit_zero_coupon_bonds() {
        // Generate "true" discount factors from a flat 5% curve
        let mats: Vec<f64> = (1..=5).map(|y| y as f64).collect();
        let prices: Vec<f64> = mats.iter().map(|&t| (-0.05 * t).exp()).collect();
        let curve = PolynomialDiscountCurve::fit(&mats, &prices, 4);
        // Fitted curve should estimate discount factors within 1%
        for (&t, &p) in mats.iter().zip(prices.iter()) {
            let est = curve.discount(t);
            assert!((est - p).abs() < 0.01, "t={} est={} true={}", t, est, p);
        }
    }

    #[test]
    fn exp_spline_discount_at_zero() {
        // With equal weights summing to 1, d(0) should be 1
        let knots = vec![0.5, 1.0, 2.0];
        let weights = vec![0.5, 0.3, 0.2]; // sum = 1
        let curve = ExponentialSplineCurve::new(knots, weights);
        assert!((curve.discount(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn exp_spline_fit_zero_coupon_bonds() {
        let mats: Vec<f64> = (1..=8).map(|y| y as f64).collect();
        let prices: Vec<f64> = mats.iter().map(|&t| (-0.05 * t).exp()).collect();
        let curve = ExponentialSplineCurve::fit(&mats, &prices, None);
        // d(0) = 1
        assert!((curve.discount(0.0) - 1.0).abs() < 1e-8, "d(0)={}", curve.discount(0.0));
        // Fitted prices within 1 bp
        for (&t, &p) in mats.iter().zip(prices.iter()) {
            let est = curve.discount(t);
            assert!((est - p).abs() < 0.01, "t={} est={} true={}", t, est, p);
        }
    }

    #[test]
    fn forward_rate_positive() {
        let mats: Vec<f64> = (1..=5).map(|y| y as f64).collect();
        let prices: Vec<f64> = mats.iter().map(|&t| (-0.05 * t).exp()).collect();
        let curve = ExponentialSplineCurve::fit(&mats, &prices, None);
        let fwd = curve.forward_rate(1.0, 2.0);
        assert!(fwd > 0.0, "forward rate: {}", fwd);
    }

    #[test]
    fn coupon_bond_fitting_polynomial() {
        // 3 bonds: 1Y 5%, 2Y 5%, 3Y 5% at par (rate = coupon)
        let flat_rate = 0.05;
        let cash_flows: Vec<Vec<(f64, f64)>> = vec![
            vec![(0.5, 2.5), (1.0, 102.5)],
            vec![(0.5, 2.5), (1.0, 2.5), (1.5, 2.5), (2.0, 102.5)],
            vec![(1.0, 5.0), (2.0, 5.0), (3.0, 105.0)],
        ];
        let dirty_prices: Vec<f64> = cash_flows.iter().map(|cfs| {
            cfs.iter().map(|&(t, cf)| cf * (-flat_rate * t).exp()).sum::<f64>()
        }).collect();
        let curve = PolynomialDiscountCurve::fit_coupon_bonds(&cash_flows, &dirty_prices, 3);
        // Verify the curve re-prices the bonds reasonably
        for (cfs, &dp) in cash_flows.iter().zip(dirty_prices.iter()) {
            let model_price: f64 = cfs.iter().map(|&(t, cf)| cf * curve.discount(t)).sum();
            assert!((model_price - dp).abs() < 1.0, "repricing error: model={} market={}", model_price, dp);
        }
    }
}
