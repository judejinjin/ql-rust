//! Calibrate a Heston stochastic volatility model to synthetic option prices.
//!
//! Run with:
//! ```sh
//! cargo run -p ql-rust --example calibrate_heston
//! ```

use ql_math::optimization::EndCriteria;
use ql_models::{CalibratedModel, CalibrationHelper, HestonModel, calibrate};
use ql_pricingengines::heston_price;

/// A calibration helper that prices a European option under the Heston model.
struct HestonCalibHelper {
    spot: f64,
    rate: f64,
    dividend: f64,
    strike: f64,
    time_to_expiry: f64,
    is_call: bool,
    market_price: f64,
}

impl CalibrationHelper for HestonCalibHelper {
    fn market_value(&self) -> f64 {
        self.market_price
    }

    fn model_value_with_params(&self, params: &[f64]) -> f64 {
        let model = HestonModel::new(
            self.spot,
            self.rate,
            self.dividend,
            params[0], params[1], params[2], params[3], params[4],
        );
        heston_price(&model, self.strike, self.time_to_expiry, self.is_call).npv
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Heston Model Calibration");
    println!("════════════════════════\n");

    // ── "True" model parameters (used to generate synthetic market data) ──
    let spot = 100.0;
    let rate = 0.05;
    let dividend = 0.02;

    let true_v0 = 0.04;
    let true_kappa = 2.0;
    let true_theta = 0.04;
    let true_sigma = 0.3;
    let true_rho = -0.5;

    println!("True parameters:");
    println!("  v0    = {:.4}", true_v0);
    println!("  κ     = {:.4}", true_kappa);
    println!("  θ     = {:.4}", true_theta);
    println!("  σ     = {:.4}", true_sigma);
    println!("  ρ     = {:.4}", true_rho);
    println!("  Feller: 2κθ = {:.4} > σ² = {:.4} → {}",
        2.0 * true_kappa * true_theta,
        true_sigma * true_sigma,
        if 2.0 * true_kappa * true_theta > true_sigma * true_sigma { "✓" } else { "✗" }
    );
    println!();

    let true_model = HestonModel::new(
        spot, rate, dividend, true_v0, true_kappa, true_theta, true_sigma, true_rho,
    );

    // ── Generate synthetic market data ───────────────────────────────
    let strikes = [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0];
    let maturities = [0.25, 0.5, 1.0, 2.0];

    println!("Synthetic market (Heston prices):");
    println!("  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}", "Strike", "T=0.25", "T=0.50", "T=1.00", "T=2.00");

    let mut helpers: Vec<Box<dyn CalibrationHelper>> = Vec::new();

    for &k in &strikes {
        print!("  {:>8.1}", k);
        for &t in &maturities {
            let price = heston_price(&true_model, k, t, true).npv;
            print!("  {:>8.4}", price);
            helpers.push(Box::new(HestonCalibHelper {
                spot,
                rate,
                dividend,
                strike: k,
                time_to_expiry: t,
                is_call: true,
                market_price: price,
            }));
        }
        println!();
    }
    println!("\n  Total calibration instruments: {}\n", helpers.len());

    // ── Calibrate from perturbed starting point ──────────────────────
    let mut model = HestonModel::new(
        spot, rate, dividend,
        0.06,  // v0 guess
        1.0,   // kappa guess
        0.06,  // theta guess
        0.5,   // sigma guess
        -0.3,  // rho guess
    );

    println!("Starting guess:");
    println!("  v0    = {:.4}", 0.06);
    println!("  κ     = {:.4}", 1.0);
    println!("  θ     = {:.4}", 0.06);
    println!("  σ     = {:.4}", 0.5);
    println!("  ρ     = {:.4}\n", -0.3);

    let criteria = EndCriteria {
        max_iterations: 5000,
        max_stationary_iterations: 500,
        root_epsilon: 1e-12,
        function_epsilon: 1e-12,
        gradient_epsilon: 1e-12,
    };

    println!("Calibrating (Simplex optimizer)...");
    let sse = calibrate(&mut model, &helpers, &criteria)?;

    let params = model.params_as_vec();
    println!("  Done! SSE = {:.2e}\n", sse);

    // ── Results ──────────────────────────────────────────────────────
    println!("Calibrated parameters:");
    println!("  {:>8}  {:>10}  {:>10}  {:>10}", "Param", "True", "Fitted", "Error");
    println!("  {:>8}  {:>10}  {:>10}  {:>10}", "─────", "────", "──────", "─────");
    println!("  {:>8}  {:>10.6}  {:>10.6}  {:>+10.6}", "v0", true_v0, params[0], params[0] - true_v0);
    println!("  {:>8}  {:>10.6}  {:>10.6}  {:>+10.6}", "κ", true_kappa, params[1], params[1] - true_kappa);
    println!("  {:>8}  {:>10.6}  {:>10.6}  {:>+10.6}", "θ", true_theta, params[2], params[2] - true_theta);
    println!("  {:>8}  {:>10.6}  {:>10.6}  {:>+10.6}", "σ", true_sigma, params[3], params[3] - true_sigma);
    println!("  {:>8}  {:>10.6}  {:>10.6}  {:>+10.6}", "ρ", true_rho, params[4], params[4] - true_rho);
    println!();

    // ── Repricing check ──────────────────────────────────────────────
    let mut max_err: f64 = 0.0;
    for helper in &helpers {
        let model_price = helper.model_value_with_params(&params);
        let err = (model_price - helper.market_value()).abs();
        max_err = max_err.max(err);
    }
    println!("Max repricing error: {:.6}", max_err);

    Ok(())
}
