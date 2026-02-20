#![allow(clippy::too_many_arguments)]
//! LMM multi-step product pricing via Monte Carlo with Longstaff-Schwartz.
//!
//! Provides a generic [`LmmProduct`] trait for defining payoffs on forward-rate
//! paths, and a [`lmm_product_mc`] engine that supports both European cash-flow
//! products and Bermudan / callable exercise decisions using backward
//! Longstaff-Schwartz regression.
//!
//! Built-in products:
//! - [`BermudanSwaption`] — exercise at each coupon date
//! - [`CmsSpreadOption`] — payoff on the spread of two CMS rates
//! - [`CallableRangeAccrual`] — accrues coupon when LIBOR is in a range, issuer-callable

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use ql_models::lmm::{evolve_one_step, LmmConfig, LmmCurveState, LmmResult};

// ===========================================================================
// Trait and types
// ===========================================================================

/// Exercise type for LMM products.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExerciseType {
    /// Holder exercises to maximize value (e.g., Bermudan swaption).
    Bermudan,
    /// Issuer calls to minimize holder value (e.g., callable note).
    Callable,
}

/// Trait for products priced under the LIBOR Market Model via Monte Carlo.
///
/// The engine simulates forward-rate paths using the LMM predictor-corrector
/// scheme, evaluates non-exercise cashflows along each path, then applies
/// backward Longstaff-Schwartz regression at exercise dates (if any).
pub trait LmmProduct: Send + Sync {
    /// Number of forward-rate time steps to simulate.
    fn num_steps(&self) -> usize;

    /// Non-exercise cashflow at this step (e.g., coupon accrual).
    /// Receives the forward-rate state at time T\_step.
    fn cashflow(&self, step: usize, state: &LmmCurveState, config: &LmmConfig) -> f64;

    /// Whether this step is an early-exercise / call date.
    fn is_exercise_date(&self, step: usize) -> bool;

    /// Exercise / call value at this step (value at T\_step).
    fn exercise_value(&self, step: usize, state: &LmmCurveState, config: &LmmConfig) -> f64;

    /// State variables for LSM regression at this step.
    /// The first element is used as the primary regressor.
    fn regression_variables(
        &self,
        step: usize,
        state: &LmmCurveState,
        config: &LmmConfig,
    ) -> Vec<f64>;

    /// Terminal value at maturity (e.g., par redemption for bonds, 0 for options).
    fn terminal_value(&self, state: &LmmCurveState, config: &LmmConfig) -> f64;

    /// Exercise type: holder-optimal (Bermudan) or issuer-optimal (Callable).
    fn exercise_type(&self) -> ExerciseType;
}

// ===========================================================================
// Built-in products
// ===========================================================================

/// Bermudan swaption under the LIBOR Market Model.
///
/// Exercisable at each coupon date from `swap_start` to `swap_end - 1`.
/// On exercise at step k, the holder enters a swap from k to `swap_end`,
/// receiving max(±(S(k) − K), 0) × A(k).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BermudanSwaption {
    /// First swap period index.
    pub swap_start: usize,
    /// Past-the-end swap period index.
    pub swap_end: usize,
    /// Fixed rate (strike).
    pub strike: f64,
    /// Payer (true) or receiver (false).
    pub is_payer: bool,
}

impl LmmProduct for BermudanSwaption {
    fn num_steps(&self) -> usize {
        self.swap_end
    }

    fn cashflow(&self, _step: usize, _state: &LmmCurveState, _config: &LmmConfig) -> f64 {
        0.0
    }

    fn is_exercise_date(&self, step: usize) -> bool {
        step >= self.swap_start && step < self.swap_end
    }

    fn exercise_value(&self, step: usize, state: &LmmCurveState, config: &LmmConfig) -> f64 {
        if step < self.swap_start || step >= self.swap_end {
            return 0.0;
        }
        let sr = state.swap_rate(step, self.swap_end, &config.accruals);
        let sign = if self.is_payer { 1.0 } else { -1.0 };
        let raw = sign * (sr - self.strike);
        if raw <= 0.0 {
            return 0.0;
        }
        // Annuity at T_step
        let mut annuity = 0.0;
        let mut d = 1.0;
        for k in step..self.swap_end {
            d /= 1.0 + config.accruals[k] * state.forwards[k];
            annuity += config.accruals[k] * d;
        }
        raw * annuity
    }

    fn regression_variables(
        &self,
        step: usize,
        state: &LmmCurveState,
        config: &LmmConfig,
    ) -> Vec<f64> {
        let end = self.swap_end.min(config.n_rates);
        let start = step.min(end.saturating_sub(1));
        vec![state.swap_rate(start, end, &config.accruals)]
    }

    fn terminal_value(&self, _state: &LmmCurveState, _config: &LmmConfig) -> f64 {
        0.0
    }

    fn exercise_type(&self) -> ExerciseType {
        ExerciseType::Bermudan
    }
}

/// CMS spread option under the LIBOR Market Model.
///
/// European payoff at a single observation step:
///   call: max(S1 − S2 − K, 0) × notional
///   put:  max(K − (S1 − S2), 0) × notional
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CmsSpreadOption {
    /// Step at which the spread is observed.
    pub observation_step: usize,
    /// CMS1: swap rate from `cms1_start` to `cms1_end`.
    pub cms1_start: usize,
    pub cms1_end: usize,
    /// CMS2: swap rate from `cms2_start` to `cms2_end`.
    pub cms2_start: usize,
    pub cms2_end: usize,
    /// Strike on the spread.
    pub strike: f64,
    /// Call (true) or put (false) on the spread.
    pub is_call: bool,
    /// Notional amount.
    pub notional: f64,
}

impl LmmProduct for CmsSpreadOption {
    fn num_steps(&self) -> usize {
        self.observation_step + 1
    }

    fn cashflow(&self, step: usize, state: &LmmCurveState, config: &LmmConfig) -> f64 {
        if step != self.observation_step {
            return 0.0;
        }
        let s1 = state.swap_rate(self.cms1_start, self.cms1_end, &config.accruals);
        let s2 = state.swap_rate(self.cms2_start, self.cms2_end, &config.accruals);
        let spread = s1 - s2;
        let payoff = if self.is_call {
            (spread - self.strike).max(0.0)
        } else {
            (self.strike - spread).max(0.0)
        };
        payoff * self.notional
    }

    fn is_exercise_date(&self, _step: usize) -> bool {
        false
    }

    fn exercise_value(
        &self,
        _step: usize,
        _state: &LmmCurveState,
        _config: &LmmConfig,
    ) -> f64 {
        0.0
    }

    fn regression_variables(
        &self,
        _step: usize,
        _state: &LmmCurveState,
        _config: &LmmConfig,
    ) -> Vec<f64> {
        vec![]
    }

    fn terminal_value(&self, _state: &LmmCurveState, _config: &LmmConfig) -> f64 {
        0.0
    }

    fn exercise_type(&self) -> ExerciseType {
        ExerciseType::Bermudan
    }
}

/// Callable range accrual note under the LIBOR Market Model.
///
/// Accrues `coupon_rate × τ` at each step when the current LIBOR rate
/// `f_step` is within `[lower_barrier, upper_barrier]`. Callable by the
/// issuer at specified dates, who redeems at `call_price × notional`.
/// Par is returned at maturity.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CallableRangeAccrual {
    /// First coupon step.
    pub start_step: usize,
    /// Past-the-end step (maturity).
    pub end_step: usize,
    /// Lower barrier for range accrual.
    pub lower_barrier: f64,
    /// Upper barrier for range accrual.
    pub upper_barrier: f64,
    /// Coupon rate (annualized).
    pub coupon_rate: f64,
    /// Call price as fraction of notional (typically 1.0).
    pub call_price: f64,
    /// Steps at which the issuer may call.
    pub call_steps: Vec<usize>,
    /// Notional amount.
    pub notional: f64,
}

impl LmmProduct for CallableRangeAccrual {
    fn num_steps(&self) -> usize {
        self.end_step
    }

    fn cashflow(&self, step: usize, state: &LmmCurveState, config: &LmmConfig) -> f64 {
        if step < self.start_step || step >= self.end_step {
            return 0.0;
        }
        let rate_idx = step.min(config.n_rates - 1);
        let rate = state.forwards[rate_idx];
        if rate >= self.lower_barrier && rate <= self.upper_barrier {
            self.coupon_rate * config.accruals[step.min(config.accruals.len() - 1)] * self.notional
        } else {
            0.0
        }
    }

    fn is_exercise_date(&self, step: usize) -> bool {
        self.call_steps.contains(&step)
    }

    fn exercise_value(
        &self,
        _step: usize,
        _state: &LmmCurveState,
        _config: &LmmConfig,
    ) -> f64 {
        self.call_price * self.notional
    }

    fn regression_variables(
        &self,
        step: usize,
        state: &LmmCurveState,
        _config: &LmmConfig,
    ) -> Vec<f64> {
        let rate_idx = step.min(state.forwards.len() - 1);
        vec![state.forwards[rate_idx]]
    }

    fn terminal_value(&self, _state: &LmmCurveState, _config: &LmmConfig) -> f64 {
        self.notional
    }

    fn exercise_type(&self) -> ExerciseType {
        ExerciseType::Callable
    }
}

// ===========================================================================
// Generic MC engine
// ===========================================================================

/// Price an LMM product via Monte Carlo with optional Longstaff-Schwartz.
///
/// For European products (no exercise dates), sums discounted cashflows.
/// For Bermudan / callable products, applies backward LSM regression.
///
/// # Parameters
/// - `config` — LMM configuration (forward rates, vols, correlations)
/// - `product` — the product to price (implements [`LmmProduct`])
/// - `n_paths` — number of Monte Carlo paths
/// - `basis_degree` — polynomial degree for LSM regression (2–4 recommended)
/// - `seed` — RNG seed for reproducibility
pub fn lmm_product_mc(
    config: &LmmConfig,
    product: &dyn LmmProduct,
    n_paths: usize,
    basis_degree: usize,
    seed: u64,
) -> LmmResult {
    let n = config.n_rates;
    let chol = config.cholesky();
    let dt = config.accruals[0];
    let sqrt_dt = dt.sqrt();
    let n_steps = product.num_steps().min(n);

    // Initial-curve discount factors: df[k] = P(0, T_k)
    let mut df = vec![1.0_f64; n + 1];
    for k in 0..n {
        df[k + 1] = df[k] / (1.0 + config.accruals[k] * config.initial_forwards[k]);
    }

    // ---- Forward simulation: store all curve states ----
    let mut all_forwards: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_paths);
    let mut rng = SmallRng::seed_from_u64(seed);

    for _ in 0..n_paths {
        let mut forwards = config.initial_forwards.clone();
        let mut path_fwds = Vec::with_capacity(n_steps + 1);
        path_fwds.push(forwards.clone());

        for step in 0..n_steps {
            let z: Vec<f64> = (0..n).map(|_| rng.sample(StandardNormal)).collect();
            forwards =
                evolve_one_step(config, &forwards, step + 1, dt, sqrt_dt, &chol, &z);
            path_fwds.push(forwards.clone());
        }
        all_forwards.push(path_fwds);
    }

    // ---- Check for exercise dates ----
    let has_exercise = (0..n_steps).any(|s| product.is_exercise_date(s));

    if !has_exercise {
        european_price(config, product, &all_forwards, &df, n_paths, n_steps)
    } else {
        backward_lsm(
            config,
            product,
            &all_forwards,
            &df,
            n_paths,
            n_steps,
            basis_degree,
        )
    }
}

/// European pricing: sum discounted cashflows + terminal value.
fn european_price(
    config: &LmmConfig,
    product: &dyn LmmProduct,
    all_forwards: &[Vec<Vec<f64>>],
    df: &[f64],
    n_paths: usize,
    n_steps: usize,
) -> LmmResult {
    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for fwds in all_forwards.iter().take(n_paths) {
        let mut path_pv = 0.0;
        for step in 0..n_steps {
            let state = LmmCurveState {
                forwards: fwds[step].clone(),
                alive_index: step,
            };
            let cf = product.cashflow(step, &state, config);
            path_pv += cf * df[step];
        }
        let final_state = LmmCurveState {
            forwards: fwds[n_steps].clone(),
            alive_index: n_steps,
        };
        path_pv += product.terminal_value(&final_state, config) * df[n_steps];

        sum += path_pv;
        sum_sq += path_pv * path_pv;
    }

    let mean = sum / n_paths as f64;
    let variance = (sum_sq / n_paths as f64 - mean * mean).max(0.0);
    LmmResult {
        price: mean,
        std_error: (variance / n_paths as f64).sqrt(),
    }
}

/// Backward Longstaff-Schwartz for Bermudan / callable exercise.
fn backward_lsm(
    config: &LmmConfig,
    product: &dyn LmmProduct,
    all_forwards: &[Vec<Vec<f64>>],
    df: &[f64],
    n_paths: usize,
    n_steps: usize,
    basis_degree: usize,
) -> LmmResult {
    let ex_type = product.exercise_type();
    let p_cols = basis_degree + 1;

    // cont_pv0[p] = PV at T_0 of all cashflows from step+1 onward
    let mut cont_pv0 = vec![0.0_f64; n_paths];
    for (p, fwds) in all_forwards.iter().enumerate().take(n_paths) {
        let state = LmmCurveState {
            forwards: fwds[n_steps].clone(),
            alive_index: n_steps,
        };
        cont_pv0[p] = product.terminal_value(&state, config) * df[n_steps];
    }

    let mut basis_buf = Vec::new();

    for step in (0..n_steps).rev() {
        if !product.is_exercise_date(step) {
            // Not an exercise date — just add coupon
            for (p, fwds) in all_forwards.iter().enumerate().take(n_paths) {
                let state = LmmCurveState {
                    forwards: fwds[step].clone(),
                    alive_index: step,
                };
                cont_pv0[p] += product.cashflow(step, &state, config) * df[step];
            }
            continue;
        }

        // ---- Exercise date: compute exercise values and regression ----

        // Coupon PV at this step (earned if not exercised)
        let mut coupon_pv = vec![0.0_f64; n_paths];
        // Total continuation = continuation from step+1 + coupon at this step
        let mut total_cont = vec![0.0_f64; n_paths];
        // Exercise PV at T_0
        let mut exercise_pv = vec![0.0_f64; n_paths];
        // Regression variable
        let mut reg_x = vec![0.0_f64; n_paths];
        let mut itm_indices: Vec<usize> = Vec::new();

        for (p, fwds) in all_forwards.iter().enumerate().take(n_paths) {
            let state = LmmCurveState {
                forwards: fwds[step].clone(),
                alive_index: step,
            };

            coupon_pv[p] = product.cashflow(step, &state, config) * df[step];
            total_cont[p] = cont_pv0[p] + coupon_pv[p];

            let ex_val = product.exercise_value(step, &state, config);
            exercise_pv[p] = ex_val * df[step];

            let vars = product.regression_variables(step, &state, config);
            reg_x[p] = if vars.is_empty() { 0.0 } else { vars[0] };

            let is_candidate = match ex_type {
                ExerciseType::Bermudan => ex_val > 0.0,
                ExerciseType::Callable => true,
            };
            if is_candidate {
                itm_indices.push(p);
            }
        }

        if itm_indices.len() < p_cols + 1 {
            // Not enough paths for regression — just add coupon
            for p in 0..n_paths {
                cont_pv0[p] = total_cont[p];
            }
            continue;
        }

        // Regression: fit E[total_cont | reg_x]
        let itm_x: Vec<f64> = itm_indices.iter().map(|&p| reg_x[p]).collect();
        let itm_y: Vec<f64> = itm_indices.iter().map(|&p| total_cont[p]).collect();

        build_monomial_basis(&itm_x, basis_degree, &mut basis_buf);
        let coeffs = least_squares_fit(&basis_buf, p_cols, &itm_y);

        // Apply exercise decisions
        let mut exercised = vec![false; n_paths];
        for &p in &itm_indices {
            let fitted_cont = evaluate_monomial(reg_x[p], &coeffs);
            let trigger = match ex_type {
                ExerciseType::Bermudan => exercise_pv[p] >= fitted_cont,
                ExerciseType::Callable => fitted_cont >= exercise_pv[p],
            };
            if trigger {
                cont_pv0[p] = exercise_pv[p];
                exercised[p] = true;
            }
        }

        // Non-exercised paths: use total continuation (adds coupon)
        for p in 0..n_paths {
            if !exercised[p] {
                cont_pv0[p] = total_cont[p];
            }
        }
    }

    // Final statistics
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for &v in &cont_pv0 {
        sum += v;
        sum_sq += v * v;
    }

    let mean = sum / n_paths as f64;
    let variance = (sum_sq / n_paths as f64 - mean * mean).max(0.0);
    LmmResult {
        price: mean,
        std_error: (variance / n_paths as f64).sqrt(),
    }
}

// ===========================================================================
// Regression helpers (monomial basis, normal equations + Cholesky)
// ===========================================================================

fn build_monomial_basis(x: &[f64], degree: usize, buf: &mut Vec<f64>) {
    let n = x.len();
    let p = degree + 1;
    buf.clear();
    buf.resize(n * p, 0.0);
    for (i, &xi) in x.iter().enumerate() {
        let row = i * p;
        let mut xp = 1.0;
        for col in 0..p {
            buf[row + col] = xp;
            xp *= xi;
        }
    }
}

fn evaluate_monomial(x: f64, coeffs: &[f64]) -> f64 {
    let mut val = 0.0;
    let mut xp = 1.0;
    for &c in coeffs {
        val += c * xp;
        xp *= x;
    }
    val
}

fn least_squares_fit(basis_matrix: &[f64], p: usize, y: &[f64]) -> Vec<f64> {
    let n = y.len();
    if n == 0 || p == 0 {
        return vec![];
    }

    // A^T A (p × p) and A^T y (p × 1)
    let mut ata = vec![0.0; p * p];
    let mut aty = vec![0.0; p];

    for (i, &yi) in y.iter().enumerate() {
        let row = i * p;
        for j in 0..p {
            let bj = basis_matrix[row + j];
            aty[j] += bj * yi;
            for k in j..p {
                ata[j * p + k] += bj * basis_matrix[row + k];
            }
        }
    }
    for j in 1..p {
        for k in 0..j {
            ata[j * p + k] = ata[k * p + j];
        }
    }
    // Small regularization
    for j in 0..p {
        ata[j * p + j] += 1e-8;
    }

    let l = cholesky_flat(p, &ata);

    // Forward substitution: L z = A^T y
    let mut z = vec![0.0; p];
    for i in 0..p {
        let mut s = aty[i];
        for j in 0..i {
            s -= l[i * p + j] * z[j];
        }
        z[i] = s / l[i * p + i];
    }

    // Back substitution: L^T c = z
    let mut c = vec![0.0; p];
    for i in (0..p).rev() {
        let mut s = z[i];
        for j in (i + 1)..p {
            s -= l[j * p + i] * c[j];
        }
        c[i] = s / l[i * p + i];
    }
    c
}

fn cholesky_flat(n: usize, a: &[f64]) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
            if i == j {
                let v = a[i * n + i] - s;
                l[i * n + j] = if v > 0.0 { v.sqrt() } else { 1e-15 };
            } else {
                l[i * n + j] = (a[i * n + j] - s) / l[j * n + j];
            }
        }
    }
    l
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ql_models::lmm::{lmm_swaption_price, LmmConfig};

    fn make_config() -> LmmConfig {
        LmmConfig::flat(10, 0.05, 0.25, 0.20, 0.5)
    }

    #[test]
    fn bermudan_swaption_positive() {
        let config = make_config();
        let product = BermudanSwaption {
            swap_start: 2,
            swap_end: 10,
            strike: 0.05,
            is_payer: true,
        };
        let result = lmm_product_mc(&config, &product, 5000, 3, 42);
        assert!(
            result.price > 0.0,
            "Bermudan payer swaption should be positive: {}",
            result.price
        );
    }

    #[test]
    fn bermudan_exceeds_european() {
        let config = make_config();
        let n_paths = 10_000;

        // European swaption (from existing LMM engine)
        let euro = lmm_swaption_price(&config, 2, 10, 0.05, n_paths, true, 42);

        // Bermudan swaption (multiple exercise dates ⟹ ≥ European)
        let product = BermudanSwaption {
            swap_start: 2,
            swap_end: 10,
            strike: 0.05,
            is_payer: true,
        };
        let berm = lmm_product_mc(&config, &product, n_paths, 3, 42);

        assert!(
            berm.price >= euro.price - 3.0 * (berm.std_error + euro.std_error),
            "Bermudan {:.6} should >= European {:.6} (within 3 SE)",
            berm.price,
            euro.price
        );
    }

    #[test]
    fn bermudan_payer_receiver_both_positive() {
        let config = make_config();
        let payer = BermudanSwaption {
            swap_start: 2,
            swap_end: 10,
            strike: 0.05,
            is_payer: true,
        };
        let receiver = BermudanSwaption {
            swap_start: 2,
            swap_end: 10,
            strike: 0.05,
            is_payer: false,
        };
        let p_result = lmm_product_mc(&config, &payer, 5000, 3, 42);
        let r_result = lmm_product_mc(&config, &receiver, 5000, 3, 42);
        assert!(p_result.price > 0.0, "Payer Bermudan should be positive");
        assert!(r_result.price > 0.0, "Receiver Bermudan should be positive");
    }

    #[test]
    fn cms_spread_option_positive() {
        let config = make_config();
        let product = CmsSpreadOption {
            observation_step: 2,
            cms1_start: 2,
            cms1_end: 10,
            cms2_start: 2,
            cms2_end: 6,
            strike: 0.0,
            is_call: true,
            notional: 1.0,
        };
        let result = lmm_product_mc(&config, &product, 20_000, 3, 42);
        assert!(
            result.price > 0.0,
            "CMS spread call should be positive: {}",
            result.price
        );
    }

    #[test]
    fn cms_spread_monotone_in_strike() {
        let config = make_config();
        let make_product = |strike| CmsSpreadOption {
            observation_step: 2,
            cms1_start: 2,
            cms1_end: 10,
            cms2_start: 2,
            cms2_end: 6,
            strike,
            is_call: true,
            notional: 1.0,
        };
        let n = 20_000;
        let low = lmm_product_mc(&config, &make_product(0.0), n, 3, 42);
        let high = lmm_product_mc(&config, &make_product(0.02), n, 3, 42);
        assert!(
            low.price >= high.price - 3.0 * (low.std_error + high.std_error),
            "Lower strike CMS spread {:.6} should >= higher strike {:.6}",
            low.price,
            high.price
        );
    }

    #[test]
    fn callable_range_accrual_positive() {
        let config = make_config();
        let product = CallableRangeAccrual {
            start_step: 0,
            end_step: 10,
            lower_barrier: 0.02,
            upper_barrier: 0.08,
            coupon_rate: 0.06,
            call_price: 1.0,
            call_steps: vec![3, 5, 7],
            notional: 1.0,
        };
        let result = lmm_product_mc(&config, &product, 5000, 3, 42);
        assert!(
            result.price > 0.0,
            "Callable range accrual should be positive: {}",
            result.price
        );
    }

    #[test]
    fn callable_less_than_noncallable() {
        let config = make_config();
        let n_paths = 10_000;

        // Non-callable: no call dates
        let noncallable = CallableRangeAccrual {
            start_step: 0,
            end_step: 10,
            lower_barrier: 0.02,
            upper_barrier: 0.08,
            coupon_rate: 0.06,
            call_price: 1.0,
            call_steps: vec![],
            notional: 1.0,
        };
        let nc = lmm_product_mc(&config, &noncallable, n_paths, 3, 42);

        // Callable
        let callable = CallableRangeAccrual {
            start_step: 0,
            end_step: 10,
            lower_barrier: 0.02,
            upper_barrier: 0.08,
            coupon_rate: 0.06,
            call_price: 1.0,
            call_steps: vec![2, 4, 6, 8],
            notional: 1.0,
        };
        let c = lmm_product_mc(&config, &callable, n_paths, 3, 42);

        let tol = 3.0 * (nc.std_error + c.std_error);
        assert!(
            c.price <= nc.price + tol,
            "Callable {:.6} should <= non-callable {:.6} (+{:.4} tol)",
            c.price,
            nc.price,
            tol
        );
    }
}
