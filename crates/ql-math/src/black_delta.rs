//! Black-Scholes delta calculator with FX delta conventions.
//!
//! Implements the `BlackDeltaCalculator` from QuantLib, supporting multiple
//! delta conventions used in FX markets:
//!
//! - **Spot delta**: Δ_S = e^{-r_f τ} Φ(ωd₊)
//! - **Forward delta**: Δ_F = Φ(ωd₊)  
//! - **Premium-adjusted (pips) spot delta**: Δ_{PA} = Ke^{-r_d τ}/S · Φ(ωd₋)
//!
//! Also provides strike-from-delta solvers (ATM conventions, 25Δ risk reversals, etc.).
//!
//! Reference: Reiswich & Wystup (2010), *FX Volatility Smile Construction*.

use ql_core::errors::{QLError, QLResult};

/// FX delta type/convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DeltaType {
    /// Spot delta (most common for short-dated FX):  Δ = e^{-r_f τ} Φ(ωd₊)
    Spot,
    /// Forward delta: Δ = Φ(ωd₊)
    Forward,
    /// Premium-adjusted spot delta (pips delta): Δ = e^{-r_d τ}(K/S)Φ(ωd₋)
    PremiumAdjustedSpot,
    /// Premium-adjusted forward delta: Δ = (K/F)Φ(ωd₋)
    PremiumAdjustedForward,
}

/// ATM convention for determining the at-the-money strike.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AtmType {
    /// ATM Forward: K = F
    AtmForward,
    /// ATM DNS (delta-neutral straddle): call delta + put delta = 0
    AtmDeltaNeutral,
    /// ATM 50Δ: call delta = 0.5 (in the chosen delta convention)
    Atm50Delta,
}

/// Black-Scholes delta calculator for FX options.
///
/// Given market parameters, computes deltas under various conventions
/// and solves for strikes from delta values.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlackDeltaCalculator {
    /// Option type: +1 for call, −1 for put
    omega: f64,
    /// Spot rate
    spot: f64,
    /// Domestic risk-free rate (continuous)
    rd: f64,
    /// Foreign risk-free rate (continuous)
    rf: f64,
    /// Volatility (Black-Scholes)
    vol: f64,
    /// Time to expiry in years
    tau: f64,
    /// Forward rate: S · exp((rd − rf) · τ)
    forward: f64,
    /// sqrt(τ)
    sqrt_tau: f64,
    /// vol · sqrt(τ)
    vol_sqrt_tau: f64,
    /// Domestic discount factor: exp(−rd·τ)
    df_d: f64,
    /// Foreign discount factor: exp(−rf·τ)
    df_f: f64,
}

impl BlackDeltaCalculator {
    /// Create a new calculator.
    ///
    /// # Arguments
    /// - `is_call` — true for call, false for put
    /// - `spot`    — current FX spot rate
    /// - `rd`      — domestic rate (continuous)
    /// - `rf`      — foreign rate (continuous)
    /// - `vol`     — Black-Scholes implied volatility
    /// - `tau`     — time to expiry in years
    pub fn new(is_call: bool, spot: f64, rd: f64, rf: f64, vol: f64, tau: f64) -> QLResult<Self> {
        if spot <= 0.0 {
            return Err(QLError::InvalidArgument("spot must be positive".into()));
        }
        if vol <= 0.0 {
            return Err(QLError::InvalidArgument("vol must be positive".into()));
        }
        if tau <= 0.0 {
            return Err(QLError::InvalidArgument("tau must be positive".into()));
        }
        let omega = if is_call { 1.0 } else { -1.0 };
        let sqrt_tau = tau.sqrt();
        let vol_sqrt_tau = vol * sqrt_tau;
        let df_d = (-rd * tau).exp();
        let df_f = (-rf * tau).exp();
        let forward = spot * ((rd - rf) * tau).exp();

        Ok(Self {
            omega,
            spot,
            rd,
            rf,
            vol,
            tau,
            forward,
            sqrt_tau,
            vol_sqrt_tau,
            df_d,
            df_f,
        })
    }

    /// Compute d₊ = [ln(F/K) + ½σ²τ] / (σ√τ)
    fn d_plus(&self, strike: f64) -> f64 {
        ((self.forward / strike).ln() + 0.5 * self.vol_sqrt_tau * self.vol_sqrt_tau)
            / self.vol_sqrt_tau
    }

    /// Compute d₋ = d₊ − σ√τ
    fn d_minus(&self, strike: f64) -> f64 {
        self.d_plus(strike) - self.vol_sqrt_tau
    }

    /// Compute the delta of the option for the given strike and delta convention.
    pub fn delta(&self, strike: f64, delta_type: DeltaType) -> f64 {
        let dp = self.d_plus(strike);
        let dm = self.d_minus(strike);
        match delta_type {
            DeltaType::Spot => self.omega * self.df_f * norm_cdf(self.omega * dp),
            DeltaType::Forward => self.omega * norm_cdf(self.omega * dp),
            DeltaType::PremiumAdjustedSpot => {
                self.omega * self.df_d * (strike / self.spot) * norm_cdf(self.omega * dm)
            }
            DeltaType::PremiumAdjustedForward => {
                self.omega * (strike / self.forward) * norm_cdf(self.omega * dm)
            }
        }
    }

    /// Compute the at-the-money strike for the given ATM convention and delta type.
    pub fn atm_strike(&self, atm_type: AtmType, delta_type: DeltaType) -> f64 {
        match atm_type {
            AtmType::AtmForward => self.forward,
            AtmType::Atm50Delta => {
                // Solve: delta(K, delta_type) = ω·0.5
                // For spot delta: Φ(ω·d₊) = 0.5/df_f → d₊ = ω·Φ⁻¹(0.5/df_f)
                // For forward delta: Φ(ω·d₊) = 0.5 → d₊ = 0 → K = F·exp(½σ²τ)
                match delta_type {
                    DeltaType::Forward => {
                        self.forward * (0.5 * self.vol * self.vol * self.tau).exp()
                    }
                    DeltaType::Spot => {
                        let target = 0.5 / self.df_f;
                        let z = inv_norm_cdf(target.min(1.0));
                        let d_plus = self.omega * z;
                        self.forward
                            * (-d_plus * self.vol_sqrt_tau
                                + 0.5 * self.vol_sqrt_tau * self.vol_sqrt_tau)
                                .exp()
                    }
                    _ => {
                        // For premium-adjusted, use Newton solver
                        self.solve_strike_from_delta(self.omega * 0.5, delta_type)
                    }
                }
            }
            AtmType::AtmDeltaNeutral => {
                // DNS: call_delta + put_delta = 0
                // For forward delta: Φ(d₊) + (−1)·Φ(−d₊) = Φ(d₊) − Φ(−d₊) ... 
                // Actually: Δ_call(K) + Δ_put(K) = 0 under same convention
                // Forward: Φ(d₊) − Φ(−d₊) = 0 ⟹ not quite. Put delta = −Φ(−d₊)
                // So call_delta + put_delta = Φ(d₊) − Φ(−d₊) = 2Φ(d₊)−1 = 0 ⟹ d₊=0 → same as forward ATM
                // For spot: df_f·Φ(d₊) − df_f·Φ(−d₊) = 0 → same d₊=0
                // For PA: more complex. Use Newton.
                match delta_type {
                    DeltaType::Spot | DeltaType::Forward => {
                        // d₊ = 0 ⟹ K = F·exp(½σ²τ)
                        self.forward * (0.5 * self.vol * self.vol * self.tau).exp()
                    }
                    _ => {
                        // Newton solve for DNS in premium-adjusted convention
                        let call_calc =
                            BlackDeltaCalculator::new(true, self.spot, self.rd, self.rf, self.vol, self.tau)
                                .unwrap();
                        let put_calc =
                            BlackDeltaCalculator::new(false, self.spot, self.rd, self.rf, self.vol, self.tau)
                                .unwrap();
                        let mut k = self.forward;
                        for _ in 0..100 {
                            let cd = call_calc.delta(k, delta_type);
                            let pd = put_calc.delta(k, delta_type);
                            let obj = cd + pd;
                            if obj.abs() < 1e-12 {
                                break;
                            }
                            // Numerical derivative
                            let dk = k * 1e-6;
                            let cd2 = call_calc.delta(k + dk, delta_type);
                            let pd2 = put_calc.delta(k + dk, delta_type);
                            let deriv = ((cd2 + pd2) - obj) / dk;
                            if deriv.abs() < 1e-15 {
                                break;
                            }
                            k -= obj / deriv;
                            k = k.max(self.forward * 0.01).min(self.forward * 100.0);
                        }
                        k
                    }
                }
            }
        }
    }

    /// Solve for the strike corresponding to a given delta value.
    ///
    /// Uses Newton's method with numerical Jacobian.
    pub fn strike_from_delta(&self, target_delta: f64, delta_type: DeltaType) -> QLResult<f64> {
        let k = self.solve_strike_from_delta(target_delta, delta_type);
        if k <= 0.0 || !k.is_finite() {
            return Err(QLError::InvalidArgument(format!(
                "Could not solve for strike: delta={target_delta}, type={delta_type:?}"
            )));
        }
        Ok(k)
    }

    fn solve_strike_from_delta(&self, target_delta: f64, delta_type: DeltaType) -> f64 {
        // Initial guess from forward-delta inversion:
        // Φ(ω·d₊) = |target_delta| → d₊ = ω·Φ⁻¹(|target_delta|)
        let abs_delta = target_delta.abs().clamp(1e-10, 1.0 - 1e-10);
        let z = inv_norm_cdf(abs_delta);
        let d_plus_guess = self.omega * z;
        let mut k = self.forward
            * (-d_plus_guess * self.vol_sqrt_tau
                + 0.5 * self.vol_sqrt_tau * self.vol_sqrt_tau)
                .exp();

        for _ in 0..200 {
            let d = self.delta(k, delta_type);
            let err = d - target_delta;
            if err.abs() < 1e-12 {
                break;
            }
            let dk = k * 1e-6;
            let d2 = self.delta(k + dk, delta_type);
            let deriv = (d2 - d) / dk;
            if deriv.abs() < 1e-15 {
                break;
            }
            k -= err / deriv;
            k = k.max(self.forward * 1e-4).min(self.forward * 1e4);
        }
        k
    }

    /// Forward rate.
    pub fn forward(&self) -> f64 {
        self.forward
    }
}

// ---------------------------------------------------------------------------
// Normal CDF / inverse helpers (self-contained)
// ---------------------------------------------------------------------------

fn norm_cdf(x: f64) -> f64 {
    let z = x / std::f64::consts::SQRT_2;
    0.5 * (1.0 + erf_approx(z))
}

fn erf_approx(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = t
        * (0.254829592
            + t * (-0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-ax * ax).exp())
}

/// Rational approximation to the inverse normal CDF (Beasley-Springer-Moro).
fn inv_norm_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    // Rational approximation constants
    const A: [f64; 4] = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637];
    const B: [f64; 4] = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833];
    const C: [f64; 9] = [
        0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
        0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
        0.0000321767881768, 0.0000002888167364, 0.0000003960315187,
    ];

    let q = p - 0.5;
    if q.abs() <= 0.42 {
        let r = q * q;
        q * (((A[3] * r + A[2]) * r + A[1]) * r + A[0])
            / ((((B[3] * r + B[2]) * r + B[1]) * r + B[0]) * r + 1.0)
    } else {
        let r = if q < 0.0 { p } else { 1.0 - p };
        let s = (-r.ln()).sqrt();
        let mut x = C[0]
            + s * (C[1]
                + s * (C[2]
                    + s * (C[3]
                        + s * (C[4] + s * (C[5] + s * (C[6] + s * (C[7] + s * C[8])))))));
        if q < 0.0 {
            x = -x;
        }
        x
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn forward_delta_atm_call() {
        // ATM forward call: K=F, d₊ = σ√τ/2
        let calc = BlackDeltaCalculator::new(true, 1.3000, 0.02, 0.01, 0.10, 1.0).unwrap();
        let k_atm = calc.forward();
        let delta = calc.delta(k_atm, DeltaType::Forward);
        // Forward delta of ATM-forward call ≈ Φ(σ√τ/2) ≈ Φ(0.05) ≈ 0.5199
        assert!(delta > 0.50 && delta < 0.55, "forward delta ATM call: {delta}");
    }

    #[test]
    fn spot_delta_boundary() {
        // Deep ITM call → delta ≈ df_f
        let calc = BlackDeltaCalculator::new(true, 1.3000, 0.02, 0.01, 0.10, 1.0).unwrap();
        let deep_itm = calc.delta(0.5, DeltaType::Spot);
        assert!(deep_itm > 0.95, "deep ITM spot delta: {deep_itm}");
    }

    #[test]
    fn put_delta_negative() {
        let calc = BlackDeltaCalculator::new(false, 1.3000, 0.02, 0.01, 0.10, 1.0).unwrap();
        let delta = calc.delta(1.3000, DeltaType::Forward);
        assert!(delta < 0.0, "put delta should be negative: {delta}");
    }

    #[test]
    fn atm_forward_strike() {
        let calc = BlackDeltaCalculator::new(true, 1.3000, 0.02, 0.01, 0.10, 1.0).unwrap();
        let k = calc.atm_strike(AtmType::AtmForward, DeltaType::Forward);
        assert_abs_diff_eq!(k, calc.forward(), epsilon = 1e-10);
    }

    #[test]
    fn strike_from_delta_roundtrip() {
        let calc = BlackDeltaCalculator::new(true, 1.3000, 0.03, 0.01, 0.12, 0.5).unwrap();
        let target_delta = 0.25;
        let k = calc.strike_from_delta(target_delta, DeltaType::Forward).unwrap();
        let recovered = calc.delta(k, DeltaType::Forward);
        assert_abs_diff_eq!(recovered, target_delta, epsilon = 1e-8);
    }

    #[test]
    fn strike_from_delta_spot_roundtrip() {
        let calc = BlackDeltaCalculator::new(true, 1.3000, 0.03, 0.01, 0.12, 0.5).unwrap();
        let target_delta = 0.25;
        let k = calc.strike_from_delta(target_delta, DeltaType::Spot).unwrap();
        let recovered = calc.delta(k, DeltaType::Spot);
        assert_abs_diff_eq!(recovered, target_delta, epsilon = 1e-8);
    }

    #[test]
    fn premium_adjusted_delta() {
        let calc = BlackDeltaCalculator::new(true, 1.3000, 0.02, 0.01, 0.10, 1.0).unwrap();
        let pa = calc.delta(1.3000, DeltaType::PremiumAdjustedSpot);
        let spot = calc.delta(1.3000, DeltaType::Spot);
        // Premium-adjusted delta should be less than spot delta for calls
        assert!(pa < spot, "PA delta ({pa}) should be < spot delta ({spot})");
        assert!(pa > 0.0, "PA call delta should be positive");
    }

    #[test]
    fn atm_dns_forward_equals_50delta() {
        // For non-premium-adjusted conventions, DNS = 50Δ = F·exp(½σ²τ)
        let calc = BlackDeltaCalculator::new(true, 1.3000, 0.02, 0.01, 0.10, 1.0).unwrap();
        let k_dns = calc.atm_strike(AtmType::AtmDeltaNeutral, DeltaType::Forward);
        let k_50d = calc.atm_strike(AtmType::Atm50Delta, DeltaType::Forward);
        assert_abs_diff_eq!(k_dns, k_50d, epsilon = 1e-8);
    }
}
