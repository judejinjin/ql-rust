//! Catastrophe (CAT) bonds — insurance-linked securities.
//!
//! A CAT bond is a fixed-income instrument where the principal (and possibly
//! coupons) is reduced or eliminated if a specified catastrophe event occurs
//! (hurricane, earthquake, etc.).
//!
//! ## Loss Triggers
//!
//! - **Indemnity**: actual loss exceeds threshold  
//! - **Industry index**: industry-wide loss index (PCS, PERILS)
//! - **Parametric**: physical parameter threshold (e.g., wind speed > 150 mph)
//! - **Modelled loss**: catastrophe model output exceeds trigger level
//!
//! ## Pricing Model
//!
//! Uses a compound Poisson process for catastrophe events:
//! - Events arrive at rate λ (per year)
//! - Each event causes a loss L ~ F_L (e.g., Pareto, Lognormal)
//! - If cumulative loss > attachment, principal is eroded
//! - Full loss at the exhaustion point
//!
//! Reference: Lane (2000), *Pricing Risk Transfer Transactions*;  
//!            Burnecki & Kukla (2003), *Pricing of Zero-Coupon and Coupon CAT Bonds*.

/// CAT bond instrument specification.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CatBond {
    /// Face value / notional.
    pub notional: f64,
    /// Annual coupon rate (e.g., 0.08 for 8%).
    pub coupon_rate: f64,
    /// Bond maturity in years.
    pub maturity: f64,
    /// Number of coupon periods per year (e.g., 4 for quarterly).
    pub coupon_frequency: usize,
    /// Attachment point: fraction of notional at which losses begin to erode principal.
    pub attachment: f64,
    /// Exhaustion point: fraction of notional at which principal is fully wiped out.
    pub exhaustion: f64,
    /// Loss trigger type.
    pub trigger: CatTrigger,
}

/// Type of catastrophe trigger.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum CatTrigger {
    /// Indemnity: actual loss of the sponsor.
    Indemnity,
    /// Industry index (e.g., PCS).
    IndustryIndex,
    /// Parametric (physical measurement threshold).
    Parametric {
        /// Name of the physical parameter.
        parameter_name: String,
    },
    /// Modelled loss from a catastrophe model.
    ModelledLoss,
}

/// Parameters for the catastrophe loss model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CatLossModel {
    /// Expected number of catastrophe events per year (Poisson intensity).
    pub event_rate: f64,
    /// Loss severity distribution type.
    pub severity: SeverityDistribution,
}

/// Loss severity distribution.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SeverityDistribution {
    /// Lognormal(μ, σ): loss = exp(μ + σ·Z)
    Lognormal {
        /// Mean of the log-loss.
        mu: f64,
        /// Standard deviation of the log-loss.
        sigma: f64,
    },
    /// Pareto(α, x_min): P(L > x) = (x_min/x)^α for x ≥ x_min
    Pareto {
        /// Tail index.
        alpha: f64,
        /// Minimum loss threshold.
        x_min: f64,
    },
    /// Exponential(λ): loss = Exp(λ)
    Exponential {
        /// Rate parameter.
        lambda: f64,
    },
    /// Fixed loss amount per event.
    Fixed {
        /// Loss amount per event.
        amount: f64,
    },
}

/// Result from CAT bond pricing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CatBondResult {
    /// Fair price (present value of expected cash flows).
    pub price: f64,
    /// Price as percentage of notional.
    pub price_pct: f64,
    /// Expected loss (as fraction of notional).
    pub expected_loss: f64,
    /// Probability of any principal loss.
    pub prob_loss: f64,
    /// Probability of total loss (exhaustion).
    pub prob_exhaustion: f64,
    /// Number of Monte Carlo paths used.
    pub n_paths: usize,
    /// Implied spread above risk-free (approximate).
    pub implied_spread: f64,
}

/// Price a CAT bond using Monte Carlo simulation of the catastrophe loss process.
///
/// # Arguments
/// - `bond`       — CAT bond specification
/// - `loss_model` — catastrophe loss model parameters
/// - `risk_free_rate` — continuously compounded risk-free rate
/// - `n_paths`    — number of MC simulation paths
/// - `seed`       — PRNG seed
pub fn price_cat_bond(
    bond: &CatBond,
    loss_model: &CatLossModel,
    risk_free_rate: f64,
    n_paths: usize,
    seed: u64,
) -> CatBondResult {
    let mut rng = SimpleRng::new(seed);
    let mut sum_pv = 0.0;
    let mut sum_loss_frac = 0.0;
    let mut n_loss_paths = 0usize;
    let mut n_exhaust_paths = 0usize;

    let dt_coupon = 1.0 / bond.coupon_frequency as f64;
    let n_coupons = (bond.maturity * bond.coupon_frequency as f64).ceil() as usize;

    for _ in 0..n_paths {
        // Simulate catastrophe events over [0, maturity]
        let cum_loss = simulate_cumulative_loss(
            &mut rng,
            loss_model,
            bond.maturity,
            bond.notional,
        );

        // Determine principal recovery
        let loss_fraction = tranche_loss(cum_loss / bond.notional, bond.attachment, bond.exhaustion);
        let remaining_principal = 1.0 - loss_fraction;

        if loss_fraction > 0.0 {
            n_loss_paths += 1;
        }
        if loss_fraction >= 1.0 - 1e-10 {
            n_exhaust_paths += 1;
        }
        sum_loss_frac += loss_fraction;

        // Discount cash flows
        let mut pv = 0.0;

        // Coupon payments (assume loss only affects final principal, coupons paid on remaining)
        for k in 1..=n_coupons {
            let t = k as f64 * dt_coupon;
            if t > bond.maturity + 1e-10 {
                break;
            }
            let coupon = bond.notional * bond.coupon_rate * dt_coupon * remaining_principal;
            pv += coupon * (-risk_free_rate * t).exp();
        }

        // Principal redemption at maturity
        pv += bond.notional * remaining_principal * (-risk_free_rate * bond.maturity).exp();

        sum_pv += pv;
    }

    let price = sum_pv / n_paths as f64;
    let expected_loss = sum_loss_frac / n_paths as f64;
    let prob_loss = n_loss_paths as f64 / n_paths as f64;
    let prob_exhaustion = n_exhaust_paths as f64 / n_paths as f64;

    // Implied spread: solve price = Σ coupon·df + principal·df where coupon = (rfr + spread) * N * dt
    // Approximate: spread ≈ (par - price) / (duration * notional)
    let duration_approx = (1.0 - (-risk_free_rate * bond.maturity).exp()) / risk_free_rate;
    let par = bond.notional; // par value
    let implied_spread = if duration_approx > 1e-10 {
        ((par - price) / (duration_approx * bond.notional)).max(0.0)
    } else {
        0.0
    };

    CatBondResult {
        price,
        price_pct: price / bond.notional * 100.0,
        expected_loss,
        prob_loss,
        prob_exhaustion,
        n_paths,
        implied_spread,
    }
}

/// Deterministic (closed-form) CAT bond price under Poisson-exponential model.
///
/// For simple cases where severity is exponential, we can compute:
/// E[remaining principal] = Σ_n P(N=n) · E[tranche survival | N=n]
///
/// This is a faster alternative when the loss distribution allows analytical aggregation.
pub fn cat_bond_analytic_exponential(
    bond: &CatBond,
    event_rate: f64,
    mean_loss: f64,
    risk_free_rate: f64,
) -> f64 {
    // Probability of no loss: exp(-λT)
    let lambda_t = event_rate * bond.maturity;
    let _p_no_event = (-lambda_t).exp();

    // Very rough: E[loss] ≈ λT · mean_loss / notional, capped at exhaustion
    let expected_severity = event_rate * bond.maturity * mean_loss / bond.notional;
    let loss_frac = tranche_loss(expected_severity, bond.attachment, bond.exhaustion);
    let remaining = 1.0 - loss_frac;

    // PV of coupons + principal
    let dt = 1.0 / bond.coupon_frequency as f64;
    let n_coupons = (bond.maturity * bond.coupon_frequency as f64).ceil() as usize;
    let mut pv = 0.0;
    for k in 1..=n_coupons {
        let t = k as f64 * dt;
        if t > bond.maturity + 1e-10 {
            break;
        }
        pv += bond.notional * bond.coupon_rate * dt * remaining * (-risk_free_rate * t).exp();
    }
    pv += bond.notional * remaining * (-risk_free_rate * bond.maturity).exp();
    pv
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Tranche loss: fraction of tranche notional lost given aggregate loss fraction.
fn tranche_loss(aggregate_loss_frac: f64, attachment: f64, exhaustion: f64) -> f64 {
    if aggregate_loss_frac <= attachment {
        0.0
    } else if aggregate_loss_frac >= exhaustion {
        1.0
    } else {
        (aggregate_loss_frac - attachment) / (exhaustion - attachment)
    }
}

fn simulate_cumulative_loss(
    rng: &mut SimpleRng,
    model: &CatLossModel,
    maturity: f64,
    _notional: f64,
) -> f64 {
    // Simulate Poisson number of events
    let lambda_t = model.event_rate * maturity;
    let n_events = poisson_sample(rng, lambda_t);

    let mut total_loss = 0.0;
    for _ in 0..n_events {
        total_loss += sample_severity(rng, &model.severity);
    }
    total_loss
}

fn poisson_sample(rng: &mut SimpleRng, lambda: f64) -> usize {
    // Knuth's algorithm for small λ
    if lambda < 30.0 {
        let l = (-lambda).exp();
        let mut k = 0;
        let mut p = 1.0;
        loop {
            k += 1;
            p *= rng.uniform();
            if p < l {
                break;
            }
        }
        k - 1
    } else {
        // Normal approximation for large λ
        let z = rng.normal();
        (lambda + z * lambda.sqrt()).round().max(0.0) as usize
    }
}

fn sample_severity(rng: &mut SimpleRng, dist: &SeverityDistribution) -> f64 {
    match dist {
        SeverityDistribution::Lognormal { mu, sigma } => {
            let z = rng.normal();
            (mu + sigma * z).exp()
        }
        SeverityDistribution::Pareto { alpha, x_min } => {
            let u = rng.uniform().max(1e-15);
            x_min / u.powf(1.0 / alpha)
        }
        SeverityDistribution::Exponential { lambda } => {
            let u = rng.uniform().max(1e-15);
            -u.ln() / lambda
        }
        SeverityDistribution::Fixed { amount } => *amount,
    }
}

// ---------------------------------------------------------------------------
// Simple RNG
// ---------------------------------------------------------------------------

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn sample_bond() -> CatBond {
        CatBond {
            notional: 100_000_000.0,
            coupon_rate: 0.08,
            maturity: 3.0,
            coupon_frequency: 4,
            attachment: 0.10,
            exhaustion: 0.30,
            trigger: CatTrigger::Indemnity,
        }
    }

    #[test]
    fn cat_bond_no_loss_equals_plain_bond() {
        let bond = sample_bond();
        let model = CatLossModel {
            event_rate: 0.0, // no events
            severity: SeverityDistribution::Fixed { amount: 0.0 },
        };
        let res = price_cat_bond(&bond, &model, 0.05, 5_000, 42);
        // Should be like a plain bond
        assert!(res.expected_loss < 0.01, "expected loss: {}", res.expected_loss);
        assert!(res.prob_loss < 0.01, "prob loss: {}", res.prob_loss);
        // Price should be close to sum of discounted coupons + principal
        assert!(res.price > 0.90 * bond.notional, "price too low: {}", res.price);
    }

    #[test]
    fn cat_bond_high_cat_risk_lower_price() {
        let bond = sample_bond();
        let low_risk = CatLossModel {
            event_rate: 0.1,
            severity: SeverityDistribution::Exponential { lambda: 1.0 / 5_000_000.0 },
        };
        let high_risk = CatLossModel {
            event_rate: 2.0,
            severity: SeverityDistribution::Exponential { lambda: 1.0 / 50_000_000.0 },
        };
        let res_low = price_cat_bond(&bond, &low_risk, 0.05, 10_000, 42);
        let res_high = price_cat_bond(&bond, &high_risk, 0.05, 10_000, 42);
        assert!(res_high.price < res_low.price,
                "high-risk ({}) should be cheaper than low-risk ({})", res_high.price, res_low.price);
    }

    #[test]
    fn cat_bond_prob_loss_bounded() {
        let bond = sample_bond();
        let model = CatLossModel {
            event_rate: 0.5,
            severity: SeverityDistribution::Lognormal { mu: 16.0, sigma: 1.0 },
        };
        let res = price_cat_bond(&bond, &model, 0.05, 10_000, 99);
        assert!(res.prob_loss >= 0.0 && res.prob_loss <= 1.0);
        assert!(res.prob_exhaustion <= res.prob_loss);
    }

    #[test]
    fn tranche_loss_function() {
        assert_abs_diff_eq!(tranche_loss(0.05, 0.10, 0.30), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(tranche_loss(0.10, 0.10, 0.30), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(tranche_loss(0.20, 0.10, 0.30), 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(tranche_loss(0.30, 0.10, 0.30), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(tranche_loss(0.50, 0.10, 0.30), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn cat_bond_pareto_severity() {
        let bond = sample_bond();
        let model = CatLossModel {
            event_rate: 1.0,
            severity: SeverityDistribution::Pareto { alpha: 2.0, x_min: 1_000_000.0 },
        };
        let res = price_cat_bond(&bond, &model, 0.05, 10_000, 77);
        assert!(res.price > 0.0, "price should be positive: {}", res.price);
        assert!(res.price <= bond.notional * 1.5, "price too high: {}", res.price);
    }

    #[test]
    fn cat_bond_analytic_no_loss() {
        let bond = sample_bond();
        let price = cat_bond_analytic_exponential(&bond, 0.0, 0.0, 0.05);
        // Should be a plain bond value
        assert!(price > 0.90 * bond.notional, "analytic price: {}", price);
    }
}
