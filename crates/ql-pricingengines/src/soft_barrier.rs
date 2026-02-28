//! Soft-barrier option engine.
//!
//! A soft barrier option has a payoff that activates smoothly over a barrier range
//! [B_low, B_high] rather than at a single barrier level. This avoids the
//! discontinuity of standard barrier options.
//!
//! Uses Monte Carlo simulation to price the option.
//! Corresponds to QuantLib's `SoftBarrier` pricing with continuous monitoring.

use serde::{Deserialize, Serialize};

/// Soft barrier type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SoftBarrierType {
    /// Down-and-out: knocked out if spot penetrates lower barrier range.
    DownAndOut,
    /// Down-and-in: activated if spot penetrates lower barrier range.
    DownAndIn,
    /// Up-and-out: knocked out if spot penetrates upper barrier range.
    UpAndOut,
    /// Up-and-in: activated if spot penetrates upper barrier range.
    UpAndIn,
}

/// Result from the soft barrier engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftBarrierResult {
    /// Net present value.
    pub npv: f64,
    /// Standard error (from MC).
    pub std_error: f64,
    /// Number of paths used.
    pub num_paths: usize,
}

/// Configuration for the soft barrier option.
#[derive(Debug, Clone)]
pub struct SoftBarrierConfig {
    /// Spot price.
    pub spot: f64,
    /// Strike price.
    pub strike: f64,
    /// Lower boundary of the soft barrier range.
    pub barrier_low: f64,
    /// Upper boundary of the soft barrier range.
    pub barrier_high: f64,
    /// Barrier type.
    pub barrier_type: SoftBarrierType,
    /// Risk-free rate (continuous).
    pub rate: f64,
    /// Dividend yield (continuous).
    pub div_yield: f64,
    /// Volatility.
    pub volatility: f64,
    /// Time to maturity in years.
    pub maturity: f64,
    /// True for call, false for put.
    pub is_call: bool,
    /// Number of MC paths.
    pub num_paths: usize,
    /// Number of time steps per path.
    pub num_steps: usize,
    /// Random seed.
    pub seed: u64,
}

/// Price a soft barrier option via Monte Carlo.
///
/// The soft barrier linearly interpolates the knock-in/out indicator over
/// the barrier range. If the underlying crosses into [barrier_low, barrier_high],
/// the option activation fraction = (penetration depth) / (barrier_high - barrier_low).
pub fn price_soft_barrier(config: &SoftBarrierConfig) -> SoftBarrierResult {
    let dt = config.maturity / config.num_steps as f64;
    let drift = (config.rate - config.div_yield - 0.5 * config.volatility * config.volatility) * dt;
    let vol_sqrt_dt = config.volatility * dt.sqrt();
    let discount = (-config.rate * config.maturity).exp();

    let barrier_width = (config.barrier_high - config.barrier_low).max(1e-10);

    let mut rng = SimpleRng::new(config.seed);
    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for _ in 0..config.num_paths {
        let mut s = config.spot;
        let mut max_penetration: f64 = 0.0;

        for _ in 0..config.num_steps {
            let z = rng.normal();
            s *= (drift + vol_sqrt_dt * z).exp();

            let penetration = match config.barrier_type {
                SoftBarrierType::DownAndOut | SoftBarrierType::DownAndIn => {
                    if s <= config.barrier_low {
                        1.0
                    } else if s < config.barrier_high {
                        (config.barrier_high - s) / barrier_width
                    } else {
                        0.0
                    }
                }
                SoftBarrierType::UpAndOut | SoftBarrierType::UpAndIn => {
                    if s >= config.barrier_high {
                        1.0
                    } else if s > config.barrier_low {
                        (s - config.barrier_low) / barrier_width
                    } else {
                        0.0
                    }
                }
            };
            max_penetration = max_penetration.max(penetration);
        }

        let intrinsic = if config.is_call {
            (s - config.strike).max(0.0)
        } else {
            (config.strike - s).max(0.0)
        };

        let payoff = match config.barrier_type {
            SoftBarrierType::DownAndOut | SoftBarrierType::UpAndOut => {
                intrinsic * (1.0 - max_penetration)
            }
            SoftBarrierType::DownAndIn | SoftBarrierType::UpAndIn => {
                intrinsic * max_penetration
            }
        };

        sum += payoff;
        sum_sq += payoff * payoff;
    }

    let n = config.num_paths as f64;
    let mean = sum / n;
    let variance = (sum_sq / n - mean * mean).max(0.0);
    let std_error = (variance / n).sqrt();

    SoftBarrierResult {
        npv: discount * mean,
        std_error: discount * std_error,
        num_paths: config.num_paths,
    }
}

/// Simple xorshift64 PRNG with Box-Muller for normal deviates.
struct SimpleRng {
    state: u64,
    spare: Option<f64>,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1), spare: None }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        loop {
            let u = 2.0 * self.uniform() - 1.0;
            let v = 2.0 * self.uniform() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let factor = (-2.0 * s.ln() / s).sqrt();
                self.spare = Some(v * factor);
                return u * factor;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_barrier_down_out_call() {
        let config = SoftBarrierConfig {
            spot: 100.0,
            strike: 100.0,
            barrier_low: 75.0,
            barrier_high: 85.0,
            barrier_type: SoftBarrierType::DownAndOut,
            rate: 0.05,
            div_yield: 0.0,
            volatility: 0.20,
            maturity: 1.0,
            is_call: true,
            num_paths: 50_000,
            num_steps: 100,
            seed: 42,
        };
        let res = price_soft_barrier(&config);
        // Price should be positive but less than vanilla call (~10.45)
        assert!(res.npv > 0.0, "npv={}", res.npv);
        assert!(res.npv < 15.0, "npv={}", res.npv);
    }

    #[test]
    fn test_soft_barrier_in_out_parity() {
        // Down-In + Down-Out ≈ Vanilla (approximately)
        let base = SoftBarrierConfig {
            spot: 100.0,
            strike: 100.0,
            barrier_low: 80.0,
            barrier_high: 90.0,
            barrier_type: SoftBarrierType::DownAndOut,
            rate: 0.05,
            div_yield: 0.0,
            volatility: 0.25,
            maturity: 1.0,
            is_call: true,
            num_paths: 100_000,
            num_steps: 100,
            seed: 12345,
        };
        let out_res = price_soft_barrier(&base);

        let in_config = SoftBarrierConfig {
            barrier_type: SoftBarrierType::DownAndIn,
            ..base.clone()
        };
        let in_res = price_soft_barrier(&in_config);

        let combined = out_res.npv + in_res.npv;
        // Combined should approximate vanilla price
        assert!(combined > 5.0 && combined < 20.0,
            "in={}, out={}, combined={}", in_res.npv, out_res.npv, combined);
    }

    #[test]
    fn test_soft_barrier_up_out_put() {
        let config = SoftBarrierConfig {
            spot: 100.0,
            strike: 100.0,
            barrier_low: 115.0,
            barrier_high: 125.0,
            barrier_type: SoftBarrierType::UpAndOut,
            rate: 0.05,
            div_yield: 0.0,
            volatility: 0.20,
            maturity: 1.0,
            is_call: false,
            num_paths: 50_000,
            num_steps: 100,
            seed: 99,
        };
        let res = price_soft_barrier(&config);
        assert!(res.npv > 0.0, "npv={}", res.npv);
    }

    #[test]
    fn test_wide_barrier_converges_to_vanilla() {
        // Very wide barrier range far from spot ≈ vanilla
        let config = SoftBarrierConfig {
            spot: 100.0,
            strike: 100.0,
            barrier_low: 10.0,
            barrier_high: 20.0,
            barrier_type: SoftBarrierType::DownAndOut,
            rate: 0.05,
            div_yield: 0.0,
            volatility: 0.20,
            maturity: 1.0,
            is_call: true,
            num_paths: 50_000,
            num_steps: 100,
            seed: 77,
        };
        let res = price_soft_barrier(&config);
        // With barrier far below, should be close to vanilla (~10.45)
        assert!(res.npv > 8.0 && res.npv < 14.0, "npv={}", res.npv);
    }
}
