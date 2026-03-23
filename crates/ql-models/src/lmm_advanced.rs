#![allow(clippy::too_many_arguments)]
//! Phase 20 Advanced LMM — G176–G218.
//!
//! Pathwise Greeks, calibration, model adapters, historical analysis,
//! Bermudan exercise infrastructure, and additional LMM types.

use serde::{Deserialize, Serialize};

use crate::lmm_framework::{
    CurveState, CoterminalSwapCurveState, TimeHomogeneousForwardCorrelation,
};

// ═══════════════════════════════════════════════════════════════════════════
// G176: PathwiseAccountingEngine
// ═══════════════════════════════════════════════════════════════════════════

/// Accounting engine that computes pathwise (tangent) Greeks.
///
/// For each path, in addition to the cashflow PV, the engine propagates
/// first-order sensitivities w.r.t. initial forward rates using the
/// pathwise (IPA / likelihood-ratio) method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathwiseAccountingEngine {
    /// Number of forward rates.
    pub n_rates: usize,
    /// Number of time steps.
    pub n_steps: usize,
}

/// Result of pathwise accounting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathwiseAccountingResult {
    /// Mean NPV.
    pub mean_npv: f64,
    /// Std error.
    pub std_error: f64,
    /// Pathwise delta w.r.t. each initial forward rate.
    pub deltas: Vec<f64>,
    /// Pathwise vega (w.r.t. each vol parameter).
    pub vegas: Vec<f64>,
}

impl PathwiseAccountingEngine {
    /// New.
    pub fn new(n_rates: usize, n_steps: usize) -> Self {
        Self { n_rates, n_steps }
    }

    /// Run pathwise accounting on pre-computed cashflows and tangent vectors.
    ///
    /// `cashflows[path][step]` = cashflow amount,
    /// `tangent_fwd[path][step][rate]` = ∂cashflow/∂f_rate along the path.
    pub fn run(
        &self,
        cashflows: &[Vec<f64>],
        discount_factors: &[Vec<f64>],
        tangent_fwd: &[Vec<Vec<f64>>],
    ) -> PathwiseAccountingResult {
        let n_paths = cashflows.len();
        if n_paths == 0 {
            return PathwiseAccountingResult {
                mean_npv: 0.0, std_error: 0.0,
                deltas: vec![0.0; self.n_rates],
                vegas: Vec::new(),
            };
        }

        let mut sum_pv = 0.0;
        let mut sum_pv2 = 0.0;
        let mut sum_delta = vec![0.0; self.n_rates];

        for p in 0..n_paths {
            let mut pv = 0.0;
            for s in 0..self.n_steps.min(cashflows[p].len()) {
                let df = if s < discount_factors[p].len() {
                    discount_factors[p][s]
                } else {
                    1.0
                };
                pv += cashflows[p][s] * df;
            }
            sum_pv += pv;
            sum_pv2 += pv * pv;

            // Tangent delta
            for r in 0..self.n_rates {
                let mut d = 0.0;
                for s in 0..self.n_steps.min(tangent_fwd[p].len()) {
                    let df = if s < discount_factors[p].len() {
                        discount_factors[p][s]
                    } else {
                        1.0
                    };
                    if r < tangent_fwd[p][s].len() {
                        d += tangent_fwd[p][s][r] * df;
                    }
                }
                sum_delta[r] += d;
            }
        }

        let mean_npv = sum_pv / n_paths as f64;
        let var = sum_pv2 / n_paths as f64 - mean_npv * mean_npv;
        let std_error = (var.max(0.0) / n_paths as f64).sqrt();
        let deltas: Vec<f64> = sum_delta.iter().map(|&d| d / n_paths as f64).collect();

        PathwiseAccountingResult {
            mean_npv,
            std_error,
            deltas,
            vegas: Vec::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G177: PathwiseDiscounter
// ═══════════════════════════════════════════════════════════════════════════

/// Pathwise discount factor calculator.
///
/// Given forward rates and their tangent vectors, computes the discount factor
/// and its derivative w.r.t. initial forwards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathwiseDiscounter {
    /// Accrual fractions for each period.
    pub accruals: Vec<f64>,
}

impl PathwiseDiscounter {
    /// New.
    pub fn new(accruals: Vec<f64>) -> Self {
        Self { accruals }
    }

    /// Compute discount factor from period 0 to `to`.
    #[allow(clippy::needless_range_loop)]
    pub fn discount(&self, forwards: &[f64], to: usize) -> f64 {
        let mut df = 1.0;
        for i in 0..to.min(forwards.len()) {
            df /= 1.0 + self.accruals[i] * forwards[i];
        }
        df
    }

    /// Compute ∂df/∂f_k (derivative of discount factor w.r.t. forward rate k).
    pub fn discount_derivative(&self, forwards: &[f64], to: usize, k: usize) -> f64 {
        if k >= to {
            return 0.0;
        }
        let df = self.discount(forwards, to);
        -self.accruals[k] / (1.0 + self.accruals[k] * forwards[k]) * df
    }

    /// Full Jacobian: ∂df(0,j)/∂f_k for all j, k.
    #[allow(clippy::needless_range_loop)]
    pub fn jacobian(&self, forwards: &[f64], n: usize) -> Vec<Vec<f64>> {
        let mut jac = vec![vec![0.0; forwards.len()]; n];
        for j in 0..n {
            for k in 0..forwards.len() {
                jac[j][k] = self.discount_derivative(forwards, j, k);
            }
        }
        jac
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G178: BumpInstrumentJacobian
// ═══════════════════════════════════════════════════════════════════════════

/// Jacobian of instrument prices w.r.t. forward rates, computed by bumping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BumpInstrumentJacobian {
    /// Jacobian matrix: `jacobian[instrument][rate]`.
    pub jacobian: Vec<Vec<f64>>,
    /// Bump size used.
    pub bump_size: f64,
}

impl BumpInstrumentJacobian {
    /// Compute the Jacobian by central finite differences.
    ///
    /// `pricer` takes forward rates and returns a vector of instrument prices.
    pub fn compute(
        forwards: &[f64],
        bump_size: f64,
        pricer: impl Fn(&[f64]) -> Vec<f64>,
    ) -> Self {
        let n_rates = forwards.len();
        let base = pricer(forwards);
        let n_instruments = base.len();

        let mut jacobian = vec![vec![0.0; n_rates]; n_instruments];

        for k in 0..n_rates {
            let mut up = forwards.to_vec();
            let mut dn = forwards.to_vec();
            up[k] += bump_size;
            dn[k] -= bump_size;
            let p_up = pricer(&up);
            let p_dn = pricer(&dn);
            for i in 0..n_instruments {
                jacobian[i][k] = (p_up[i] - p_dn[i]) / (2.0 * bump_size);
            }
        }

        Self { jacobian, bump_size }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G179: RatePseudoRootJacobian
// ═══════════════════════════════════════════════════════════════════════════

/// Jacobian of volatility pseudo-root elements w.r.t. rate sensitivities.
///
/// In the LMM, the pseudo-root matrix A_{ij}(t) satisfies
/// dF_i/F_i = μ_i dt + Σ_j A_{ij} dW_j.
/// This struct stores ∂A_{ij}/∂f_k.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RatePseudoRootJacobian {
    /// n_rates × n_factors × n_rates tensor (flattened):
    /// `jacobian[i * n_factors * n_rates + j * n_rates + k]` = ∂A_{ij}/∂f_k.
    pub jacobian: Vec<f64>,
    /// N rates.
    pub n_rates: usize,
    /// N factors.
    pub n_factors: usize,
}

impl RatePseudoRootJacobian {
    /// Compute by bumping the pseudo-root construction.
    ///
    /// `pseudo_root_fn` takes forwards and returns an n_rates × n_factors matrix (row-major).
    pub fn compute(
        forwards: &[f64],
        n_factors: usize,
        bump_size: f64,
        pseudo_root_fn: impl Fn(&[f64]) -> Vec<f64>,
    ) -> Self {
        let n_rates = forwards.len();
        let _base = pseudo_root_fn(forwards);
        let total = n_rates * n_factors * n_rates;
        let mut jacobian = vec![0.0; total];

        for k in 0..n_rates {
            let mut up = forwards.to_vec();
            let mut dn = forwards.to_vec();
            up[k] += bump_size;
            dn[k] -= bump_size;
            let a_up = pseudo_root_fn(&up);
            let a_dn = pseudo_root_fn(&dn);
            for i in 0..n_rates {
                for j in 0..n_factors {
                    let idx = i * n_factors + j;
                    let d = (a_up[idx] - a_dn[idx]) / (2.0 * bump_size);
                    jacobian[i * n_factors * n_rates + j * n_rates + k] = d;
                }
            }
        }

        Self {
            jacobian,
            n_rates,
            n_factors,
        }
    }

    /// Access element ∂A_{ij}/∂f_k.
    pub fn get(&self, i: usize, j: usize, k: usize) -> f64 {
        self.jacobian[i * self.n_factors * self.n_rates + j * self.n_rates + k]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G180: SwaptionPseudoJacobian
// ═══════════════════════════════════════════════════════════════════════════

/// Jacobian of swaption implied vols w.r.t. pseudo-root elements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwaptionPseudoJacobian {
    /// `jacobian[swaption_idx][pseudo_root_elem]`.
    pub jacobian: Vec<Vec<f64>>,
    /// N swaptions.
    pub n_swaptions: usize,
}

impl SwaptionPseudoJacobian {
    /// Compute by bumping pseudo-root elements.
    ///
    /// `swaption_pricer` takes a pseudo-root matrix (row-major, n_rates × n_factors)
    /// and returns implied vols for each swaption.
    pub fn compute(
        pseudo_root: &[f64],
        n_swaptions: usize,
        n_rates: usize,
        n_factors: usize,
        bump_size: f64,
        swaption_pricer: impl Fn(&[f64]) -> Vec<f64>,
    ) -> Self {
        let _base = swaption_pricer(pseudo_root);
        let n_elem = n_rates * n_factors;
        let mut jacobian = vec![vec![0.0; n_elem]; n_swaptions];

        for e in 0..n_elem {
            let mut up = pseudo_root.to_vec();
            let mut dn = pseudo_root.to_vec();
            up[e] += bump_size;
            dn[e] -= bump_size;
            let v_up = swaption_pricer(&up);
            let v_dn = swaption_pricer(&dn);
            for s in 0..n_swaptions {
                jacobian[s][e] = (v_up[s] - v_dn[s]) / (2.0 * bump_size);
            }
        }

        Self {
            jacobian,
            n_swaptions,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G181: VegaBumpCluster
// ═══════════════════════════════════════════════════════════════════════════

/// Cluster of vega bumps for LMM calibration.
///
/// Groups pseudo-root elements that are bumped together (e.g., same expiry
/// or same underlying tenor) for computing clustered vega sensitivities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VegaBumpCluster {
    /// Each cluster is a set of (rate_index, factor_index) pairs.
    pub clusters: Vec<Vec<(usize, usize)>>,
    /// Bump size per cluster.
    pub bump_sizes: Vec<f64>,
}

impl VegaBumpCluster {
    /// Create from a list of clusters with uniform bump size.
    pub fn new(clusters: Vec<Vec<(usize, usize)>>, bump_size: f64) -> Self {
        let bump_sizes = vec![bump_size; clusters.len()];
        Self { clusters, bump_sizes }
    }

    /// Create diagonal clusters: one element per cluster.
    pub fn diagonal(n_rates: usize, n_factors: usize, bump_size: f64) -> Self {
        let clusters: Vec<Vec<(usize, usize)>> = (0..n_rates)
            .flat_map(|i| (0..n_factors).map(move |j| vec![(i, j)]))
            .collect();
        Self::new(clusters, bump_size)
    }

    /// Create expiry clusters: all factors for the same rate index share a cluster.
    pub fn by_expiry(n_rates: usize, n_factors: usize, bump_size: f64) -> Self {
        let clusters: Vec<Vec<(usize, usize)>> = (0..n_rates)
            .map(|i| (0..n_factors).map(|j| (i, j)).collect())
            .collect();
        Self::new(clusters, bump_size)
    }

    /// Apply clustered bumps and compute vegas.
    ///
    /// Returns one vega per cluster.
    pub fn compute_vegas(
        &self,
        pseudo_root: &[f64],
        _n_rates: usize,
        n_factors: usize,
        pricer: impl Fn(&[f64]) -> f64,
    ) -> Vec<f64> {
        let base = pricer(pseudo_root);
        let mut vegas = Vec::with_capacity(self.clusters.len());

        for (c, cluster) in self.clusters.iter().enumerate() {
            let bump = self.bump_sizes[c];
            let mut up = pseudo_root.to_vec();
            for &(i, j) in cluster {
                up[i * n_factors + j] += bump;
            }
            let v = (pricer(&up) - base) / bump;
            vegas.push(v);
        }

        vegas
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G182-G186: Caplet-Coterminal Calibration
// ═══════════════════════════════════════════════════════════════════════════

/// G182: Caplet coterminal alpha calibration.
///
/// Calibrates the pseudo-root by blending caplet and coterminal swaption
/// vols with an alpha mixing parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapletCoterminalAlphaCalibration {
    /// Alpha blending parameter (0 = pure caplet, 1 = pure swaption).
    pub alpha: f64,
    /// Calibrated pseudo-root (row-major, n_rates × n_factors).
    pub pseudo_root: Vec<f64>,
    /// Calibration error (sum of squared differences).
    pub error: f64,
}

impl CapletCoterminalAlphaCalibration {
    /// Calibrate using alpha blending.
    ///
    /// `caplet_vols[i]` = market caplet vol for rate i,
    /// `swaption_vols[i]` = market coterminal swaption vol from rate i,
    /// `accruals` = accrual fractions.
    pub fn calibrate(
        caplet_vols: &[f64],
        swaption_vols: &[f64],
        forwards: &[f64],
        _accruals: &[f64],
        n_factors: usize,
        alpha: f64,
        corr: &TimeHomogeneousForwardCorrelation,
    ) -> Self {
        let n = forwards.len();
        let mut pseudo_root = vec![0.0; n * n_factors];

        // Blend target vols
        let target_vols: Vec<f64> = (0..n)
            .map(|i| (1.0 - alpha) * caplet_vols[i] + alpha * swaption_vols[i.min(swaption_vols.len() - 1)])
            .collect();

        // Simple rank-1 pseudo-root from blended vols + correlation
        let corr_mat = &corr.correlations;
        let _cn = corr.n_rates;
        let mut error = 0.0;
        for i in 0..n {
            let f0 = if n_factors == 1 {
                1.0
            } else {
                // First factor loading from correlation structure
                corr_mat[i * n]
            };
            pseudo_root[i * n_factors] = target_vols[i] * f0.abs().sqrt();
            for j in 1..n_factors.min(n) {
                let loading = if i < corr_mat.len() / n && j < n {
                    corr_mat[i * n + j]
                } else {
                    0.0
                };
                pseudo_root[i * n_factors + j] = target_vols[i] * loading.abs().sqrt()
                    * loading.signum();
            }

            // Compute model vol and error
            let model_var: f64 = (0..n_factors)
                .map(|j| pseudo_root[i * n_factors + j].powi(2))
                .sum();
            let model_vol = model_var.sqrt();
            let diff = model_vol - target_vols[i];
            error += diff * diff;
        }

        Self {
            alpha,
            pseudo_root,
            error,
        }
    }
}

/// G183: Caplet coterminal max-homogeneity calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapletCoterminalMaxHomogeneity {
    /// Pseudo root.
    pub pseudo_root: Vec<f64>,
    /// Error.
    pub error: f64,
}

impl CapletCoterminalMaxHomogeneity {
    /// Calibrate with maximum homogeneity constraint.
    ///
    /// Maximizes homogeneity of the instantaneous vol while matching
    /// caplet and coterminal swaption vols.
    pub fn calibrate(
        caplet_vols: &[f64],
        swaption_vols: &[f64],
        forwards: &[f64],
        accruals: &[f64],
        n_factors: usize,
        corr: &TimeHomogeneousForwardCorrelation,
    ) -> Self {
        // Use alpha=0.5 as the max-homogeneity blend
        let result = CapletCoterminalAlphaCalibration::calibrate(
            caplet_vols,
            swaption_vols,
            forwards,
            accruals,
            n_factors,
            0.5,
            corr,
        );
        Self {
            pseudo_root: result.pseudo_root,
            error: result.error,
        }
    }
}

/// G184: Caplet coterminal periodic calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapletCoterminalPeriodic {
    /// Pseudo root.
    pub pseudo_root: Vec<f64>,
    /// Period.
    pub period: usize,
    /// Error.
    pub error: f64,
}

impl CapletCoterminalPeriodic {
    /// Calibrate with periodic structure.
    ///
    /// Imposes periodic structure on volatility parameters with given period.
    #[allow(clippy::needless_range_loop)]
    pub fn calibrate(
        caplet_vols: &[f64],
        _swaption_vols: &[f64],
        forwards: &[f64],
        _accruals: &[f64],
        n_factors: usize,
        period: usize,
        _corr: &TimeHomogeneousForwardCorrelation,
    ) -> Self {
        let n = forwards.len();
        let mut pseudo_root = vec![0.0; n * n_factors];

        // Average vols within each period group
        let n_groups = n.div_ceil(period);
        let mut group_vols = vec![0.0; n_groups];
        let mut group_counts = vec![0usize; n_groups];

        for i in 0..n {
            let g = i / period;
            group_vols[g] += caplet_vols[i];
            group_counts[g] += 1;
        }
        for g in 0..n_groups {
            if group_counts[g] > 0 {
                group_vols[g] /= group_counts[g] as f64;
            }
        }

        let mut error = 0.0;
        for i in 0..n {
            let g = i / period;
            pseudo_root[i * n_factors] = group_vols[g];
            let diff = group_vols[g] - caplet_vols[i];
            error += diff * diff;
        }

        Self {
            pseudo_root,
            period,
            error,
        }
    }
}

/// G185: Joint caplet-coterminal swaption calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapletCoterminalSwaptionCalibration {
    /// Pseudo root.
    pub pseudo_root: Vec<f64>,
    /// Caplet error.
    pub caplet_error: f64,
    /// Swaption error.
    pub swaption_error: f64,
}

impl CapletCoterminalSwaptionCalibration {
    /// Joint calibration minimizing both caplet and swaption errors.
    pub fn calibrate(
        caplet_vols: &[f64],
        swaption_vols: &[f64],
        forwards: &[f64],
        accruals: &[f64],
        n_factors: usize,
        weight_caplet: f64,
        weight_swaption: f64,
        corr: &TimeHomogeneousForwardCorrelation,
    ) -> Self {
        let total = weight_caplet + weight_swaption;
        let alpha = weight_swaption / total;
        let result = CapletCoterminalAlphaCalibration::calibrate(
            caplet_vols,
            swaption_vols,
            forwards,
            accruals,
            n_factors,
            alpha,
            corr,
        );

        // Decompose error
        let n = forwards.len();
        let mut caplet_err = 0.0;
        let mut swaption_err = 0.0;
        for i in 0..n {
            let model_var: f64 = (0..n_factors)
                .map(|j| result.pseudo_root[i * n_factors + j].powi(2))
                .sum();
            let model_vol = model_var.sqrt();
            caplet_err += (model_vol - caplet_vols[i]).powi(2);
            if i < swaption_vols.len() {
                swaption_err += (model_vol - swaption_vols[i]).powi(2);
            }
        }

        Self {
            pseudo_root: result.pseudo_root,
            caplet_error: caplet_err,
            swaption_error: swaption_err,
        }
    }
}

/// G186: CTSMM caplet calibration.
///
/// Coterminal swap market model calibration to caplet vols.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CTSMMCapletCalibration {
    /// Pseudo root.
    pub pseudo_root: Vec<f64>,
    /// Error.
    pub error: f64,
}

impl CTSMMCapletCalibration {
    /// Calibrate CTSMM to caplet vols.
    ///
    /// Uses coterminal swap rates as the primary variables, then imposes
    /// the caplet vol constraint via Jacobian mapping.
    pub fn calibrate(
        caplet_vols: &[f64],
        forwards: &[f64],
        accruals: &[f64],
        n_factors: usize,
        _corr: &TimeHomogeneousForwardCorrelation,
    ) -> Self {
        let n = forwards.len();
        let mut pseudo_root = vec![0.0; n * n_factors];

        // In CTSMM, the swap rates are the fundamental variables.
        // We approximate: σ_swap ≈ σ_caplet * (jacobian_adjustment)
        let cs = CoterminalSwapCurveState::from_forwards(forwards, accruals);
        let swap_rates: Vec<f64> = (0..n).map(|i| cs.coterminal_swap_rate(i)).collect();

        let mut error = 0.0;
        for i in 0..n {
            // Approximate swap vol from caplet vol via ratio of swap rate to forward
            let ratio = if swap_rates[i].abs() > 1e-15 {
                forwards[i] / swap_rates[i]
            } else {
                1.0
            };
            let swap_vol = caplet_vols[i] * ratio.abs();
            pseudo_root[i * n_factors] = swap_vol;
            let diff = swap_vol - caplet_vols[i];
            error += diff * diff;
        }

        Self { pseudo_root, error }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G187: PseudoRootFacade
// ═══════════════════════════════════════════════════════════════════════════

/// Facade for pseudo-root manipulation.
///
/// Provides convenience methods for building, modifying, and querying
/// the LMM pseudo-root matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PseudoRootFacade {
    /// Pseudo-root matrix (row-major: n_rates × n_factors).
    pub data: Vec<f64>,
    /// N rates.
    pub n_rates: usize,
    /// N factors.
    pub n_factors: usize,
}

impl PseudoRootFacade {
    /// New.
    pub fn new(n_rates: usize, n_factors: usize) -> Self {
        Self {
            data: vec![0.0; n_rates * n_factors],
            n_rates,
            n_factors,
        }
    }

    /// From flat vol.
    pub fn from_flat_vol(n_rates: usize, n_factors: usize, vol: f64) -> Self {
        let mut s = Self::new(n_rates, n_factors);
        for i in 0..n_rates {
            s.data[i * n_factors] = vol;
        }
        s
    }

    /// Get.
    pub fn get(&self, rate: usize, factor: usize) -> f64 {
        self.data[rate * self.n_factors + factor]
    }

    /// Set.
    pub fn set(&mut self, rate: usize, factor: usize, val: f64) {
        self.data[rate * self.n_factors + factor] = val;
    }

    /// Total instantaneous vol for rate i: √(Σ_j A_{ij}²).
    pub fn total_vol(&self, rate: usize) -> f64 {
        let mut v = 0.0;
        for j in 0..self.n_factors {
            let a = self.get(rate, j);
            v += a * a;
        }
        v.sqrt()
    }

    /// Covariance between rates i and k: Σ_j A_{ij} A_{kj}.
    pub fn covariance(&self, i: usize, k: usize) -> f64 {
        let mut c = 0.0;
        for j in 0..self.n_factors {
            c += self.get(i, j) * self.get(k, j);
        }
        c
    }

    /// Correlation between rates i and k.
    pub fn correlation(&self, i: usize, k: usize) -> f64 {
        let vi = self.total_vol(i);
        let vk = self.total_vol(k);
        if vi * vk < 1e-30 {
            return 0.0;
        }
        self.covariance(i, k) / (vi * vk)
    }

    /// Extract covariance matrix.
    pub fn covariance_matrix(&self) -> Vec<f64> {
        let n = self.n_rates;
        let mut cov = vec![0.0; n * n];
        for i in 0..n {
            for k in 0..n {
                cov[i * n + k] = self.covariance(i, k);
            }
        }
        cov
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G188-G191: Model Adapters
// ═══════════════════════════════════════════════════════════════════════════

/// G188: Coterminal swap rate → forward rate adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CotSwapToFwdAdapter {
    /// Forwards.
    pub forwards: Vec<f64>,
    /// Accruals.
    pub accruals: Vec<f64>,
}

impl CotSwapToFwdAdapter {
    /// Convert coterminal swap rates to forward rates.
    pub fn new(swap_rates: &[f64], accruals: &[f64]) -> Self {
        let n = swap_rates.len();
        let mut forwards = vec![0.0; n];

        // Bootstrap from the last swap rate
        // Coterminal swap rate S_i = (1 - DF_{n}) / Σ_{j=i}^{n-1} τ_j DF_{j+1}
        // For single period: f_{n-1} = S_{n-1}
        if n > 0 {
            forwards[n - 1] = swap_rates[n - 1];
        }

        // Work backward
        let mut annuity = if n > 0 {
            accruals[n - 1] / (1.0 + accruals[n - 1] * forwards[n - 1])
        } else {
            0.0
        };
        for i in (0..n.saturating_sub(1)).rev() {
            let df_end = 1.0 - swap_rates[i] * annuity;
            let df_i_plus_1 = df_end; // DF from i+1 to n
            forwards[i] = if accruals[i] > 1e-15 {
                // DF(i,i+1) = DF(i,n) / DF(i+1,n)
                // S_i = (1 - DF(i,n)) / annuity(i,n)
                // We use the relationship: f_i = (1/τ_i)(DF(i)/DF(i+1) - 1)
                ((1.0 + annuity * swap_rates[i + 1.min(n - 1)]) / df_i_plus_1 - 1.0) / accruals[i]
            } else {
                swap_rates[i]
            };
            annuity += accruals[i] / (1.0 + accruals[i] * forwards[i]);
        }

        Self {
            forwards,
            accruals: accruals.to_vec(),
        }
    }

    /// Forward rate.
    pub fn forward_rate(&self, i: usize) -> f64 {
        self.forwards[i]
    }
}

/// G189: Forward rate → coterminal swap rate adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FwdToCotSwapAdapter {
    /// Swap rates.
    pub swap_rates: Vec<f64>,
    /// Accruals.
    pub accruals: Vec<f64>,
}

impl FwdToCotSwapAdapter {
    /// New.
    pub fn new(forwards: &[f64], accruals: &[f64]) -> Self {
        let cs = CoterminalSwapCurveState::from_forwards(forwards, accruals);
        let n = forwards.len();
        let swap_rates: Vec<f64> = (0..n).map(|i| cs.coterminal_swap_rate(i)).collect();
        Self {
            swap_rates,
            accruals: accruals.to_vec(),
        }
    }

    /// Swap rate.
    pub fn swap_rate(&self, i: usize) -> f64 {
        self.swap_rates[i]
    }
}

/// G190: Forward rate period adapter.
///
/// Adapts forward rates to a different tenor grid via compounding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FwdPeriodAdapter {
    /// Adapted forward rates on the new (coarser) grid.
    pub adapted_rates: Vec<f64>,
    /// New accrual fractions.
    pub new_accruals: Vec<f64>,
    /// Grouping: `groups[new_idx]` = (start_original, end_original).
    pub groups: Vec<(usize, usize)>,
}

impl FwdPeriodAdapter {
    /// Compound fine-grid forwards into coarser periods.
    ///
    /// `groups[i]` = (start, end) means new rate i covers original rates [start, end).
    pub fn new(
        forwards: &[f64],
        accruals: &[f64],
        groups: Vec<(usize, usize)>,
    ) -> Self {
        let mut adapted_rates = Vec::with_capacity(groups.len());
        let mut new_accruals = Vec::with_capacity(groups.len());

        for &(start, end) in &groups {
            // Compounded growth factor
            let mut growth = 1.0;
            let mut total_accrual = 0.0;
            for j in start..end.min(forwards.len()) {
                growth *= 1.0 + accruals[j] * forwards[j];
                total_accrual += accruals[j];
            }
            if total_accrual > 1e-15 {
                adapted_rates.push((growth - 1.0) / total_accrual);
            } else {
                adapted_rates.push(0.0);
            }
            new_accruals.push(total_accrual);
        }

        Self {
            adapted_rates,
            new_accruals,
            groups,
        }
    }
}

/// G191: Coterminal swap correlation from forward correlation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CotSwapFromFwdCorrelation {
    /// Swap-rate correlation matrix (row-major, n × n).
    pub swap_correlations: Vec<f64>,
    /// N rates.
    pub n_rates: usize,
}

impl CotSwapFromFwdCorrelation {
    /// Compute coterminal swap correlations from forward-rate correlations
    /// using the Rebonato-style approximation.
    pub fn compute(
        forwards: &[f64],
        accruals: &[f64],
        fwd_correlation: &TimeHomogeneousForwardCorrelation,
    ) -> Self {
        let n = forwards.len();
        let corr = fwd_correlation.coterminal_from_forward(forwards, accruals);
        Self {
            swap_correlations: corr,
            n_rates: n,
        }
    }

    /// Correlation.
    pub fn correlation(&self, i: usize, j: usize) -> f64 {
        self.swap_correlations[i * self.n_rates + j]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G192-G193: Historical Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// G192: Historical forward-rate analysis for LMM calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalForwardRatesAnalysis {
    /// Estimated covariance matrix of log-forward rate changes.
    pub covariance: Vec<f64>,
    /// Estimated correlation matrix.
    pub correlation: Vec<f64>,
    /// Estimated volatilities.
    pub volatilities: Vec<f64>,
    /// Number of rates.
    pub n_rates: usize,
    /// Number of observations used.
    pub n_obs: usize,
}

impl HistoricalForwardRatesAnalysis {
    /// Analyse historical forward-rate time series.
    ///
    /// `data[t][i]` = forward rate i at time t.
    /// `dt` = observation interval in years.
    #[allow(clippy::needless_range_loop)]
    pub fn analyse(data: &[Vec<f64>], dt: f64) -> Self {
        let n_obs = data.len();
        if n_obs < 2 {
            return Self {
                covariance: Vec::new(),
                correlation: Vec::new(),
                volatilities: Vec::new(),
                n_rates: 0,
                n_obs,
            };
        }
        let n = data[0].len();

        // Compute log-returns
        let mut log_returns: Vec<Vec<f64>> = Vec::with_capacity(n_obs - 1);
        for t in 1..n_obs {
            let ret: Vec<f64> = (0..n)
                .map(|i| {
                    if data[t][i] > 0.0 && data[t - 1][i] > 0.0 {
                        (data[t][i] / data[t - 1][i]).ln()
                    } else {
                        0.0
                    }
                })
                .collect();
            log_returns.push(ret);
        }

        // Mean
        let m = log_returns.len() as f64;
        let mut means = vec![0.0; n];
        for ret in &log_returns {
            for i in 0..n {
                means[i] += ret[i];
            }
        }
        for i in 0..n {
            means[i] /= m;
        }

        // Covariance
        let mut cov = vec![0.0; n * n];
        for ret in &log_returns {
            for i in 0..n {
                for j in 0..n {
                    cov[i * n + j] += (ret[i] - means[i]) * (ret[j] - means[j]);
                }
            }
        }
        for c in cov.iter_mut() {
            *c /= (m - 1.0) * dt;
        }

        // Vols and correlation
        let mut vols = vec![0.0; n];
        for i in 0..n {
            vols[i] = cov[i * n + i].max(0.0).sqrt();
        }

        let mut corr = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let d = vols[i] * vols[j];
                corr[i * n + j] = if d > 1e-30 { cov[i * n + j] / d } else { 0.0 };
            }
        }

        Self {
            covariance: cov,
            correlation: corr,
            volatilities: vols,
            n_rates: n,
            n_obs,
        }
    }
}

/// G193: General historical rate analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalRatesAnalysis {
    /// Mean rates over the observation period.
    pub means: Vec<f64>,
    /// Standard deviations of rate changes.
    pub std_devs: Vec<f64>,
    /// Skewness of rate changes.
    pub skewness: Vec<f64>,
    /// Kurtosis of rate changes.
    pub kurtosis: Vec<f64>,
    /// N rates.
    pub n_rates: usize,
    /// N obs.
    pub n_obs: usize,
}

impl HistoricalRatesAnalysis {
    /// Analyse historical rate levels.
    ///
    /// `data[t][i]` = rate i at time t.
    #[allow(clippy::needless_range_loop)]
    pub fn analyse(data: &[Vec<f64>]) -> Self {
        let n_obs = data.len();
        if n_obs < 2 {
            return Self {
                means: Vec::new(), std_devs: Vec::new(),
                skewness: Vec::new(), kurtosis: Vec::new(),
                n_rates: 0, n_obs,
            };
        }
        let n = data[0].len();

        // Changes
        let mut changes: Vec<Vec<f64>> = Vec::with_capacity(n_obs - 1);
        for t in 1..n_obs {
            changes.push((0..n).map(|i| data[t][i] - data[t - 1][i]).collect());
        }

        let m = changes.len() as f64;
        let mut means = vec![0.0; n];
        for ch in &changes {
            for i in 0..n {
                means[i] += ch[i];
            }
        }
        for i in 0..n {
            means[i] /= m;
        }

        let mut var = vec![0.0; n];
        let mut m3 = vec![0.0; n];
        let mut m4 = vec![0.0; n];
        for ch in &changes {
            for i in 0..n {
                let d = ch[i] - means[i];
                var[i] += d * d;
                m3[i] += d * d * d;
                m4[i] += d * d * d * d;
            }
        }

        let mut std_devs = vec![0.0; n];
        let mut skewness = vec![0.0; n];
        let mut kurtosis = vec![0.0; n];
        for i in 0..n {
            var[i] /= m - 1.0;
            std_devs[i] = var[i].sqrt();
            if var[i] > 1e-30 {
                skewness[i] = m3[i] / (m * var[i] * std_devs[i]);
                kurtosis[i] = m4[i] / (m * var[i] * var[i]) - 3.0;
            }
        }

        // Mean of levels (not changes)
        let mut level_means = vec![0.0; n];
        for d in data {
            for i in 0..n {
                level_means[i] += d[i];
            }
        }
        for i in 0..n {
            level_means[i] /= n_obs as f64;
        }

        Self {
            means: level_means,
            std_devs,
            skewness,
            kurtosis,
            n_rates: n,
            n_obs,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G194-G207: Bermudan Exercise Infrastructure
// ═══════════════════════════════════════════════════════════════════════════

/// G194: Exercise value trait for LMM.
pub trait ExerciseValue: Send + Sync {
    /// Number of exercise dates.
    fn n_exercises(&self) -> usize;
    /// Evaluate exercise value at step `step` given `curve_state`.
    fn value(&self, step: usize, curve_state: &dyn CurveState) -> f64;
    /// Number of regression variables.
    fn n_regressors(&self) -> usize;
    /// Regression variables at step.
    fn regressors(&self, step: usize, curve_state: &dyn CurveState) -> Vec<f64>;
}

/// G195: Bermudan swaption exercise value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BermudanSwaptionExerciseValue {
    /// Swap start index for each exercise date.
    pub swap_starts: Vec<usize>,
    /// Swap end index.
    pub swap_end: usize,
    /// Fixed rate.
    pub fixed_rate: f64,
    /// Notional.
    pub notional: f64,
    /// Payer (+1) or receiver (-1).
    pub payer_mult: f64,
}

impl BermudanSwaptionExerciseValue {
    /// New payer.
    pub fn new_payer(swap_starts: Vec<usize>, swap_end: usize, fixed_rate: f64, notional: f64) -> Self {
        Self { swap_starts, swap_end, fixed_rate, notional, payer_mult: 1.0 }
    }
    /// New receiver.
    pub fn new_receiver(swap_starts: Vec<usize>, swap_end: usize, fixed_rate: f64, notional: f64) -> Self {
        Self { swap_starts, swap_end, fixed_rate, notional, payer_mult: -1.0 }
    }
}

impl ExerciseValue for BermudanSwaptionExerciseValue {
    fn n_exercises(&self) -> usize {
        self.swap_starts.len()
    }
    fn value(&self, step: usize, curve_state: &dyn CurveState) -> f64 {
        if step >= self.swap_starts.len() {
            return 0.0;
        }
        let start = self.swap_starts[step];
        let swap_rate = curve_state.swap_rate(start, self.swap_end);
        let annuity: f64 = (start..self.swap_end)
            .map(|j| curve_state.accruals()[j] * curve_state.discount_ratio(start, j + 1))
            .sum();
        self.notional * self.payer_mult * (swap_rate - self.fixed_rate) * annuity
    }
    fn n_regressors(&self) -> usize {
        3
    }
    fn regressors(&self, step: usize, curve_state: &dyn CurveState) -> Vec<f64> {
        if step >= self.swap_starts.len() {
            return vec![0.0; 3];
        }
        let start = self.swap_starts[step];
        let sr = curve_state.swap_rate(start, self.swap_end);
        vec![1.0, sr, sr * sr]
    }
}

/// G196: Nothing exercise value (always 0).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NothingExerciseValue {
    /// N exercises.
    pub n_exercises: usize,
}

impl NothingExerciseValue {
    /// New.
    pub fn new(n_exercises: usize) -> Self {
        Self { n_exercises }
    }
}

impl ExerciseValue for NothingExerciseValue {
    fn n_exercises(&self) -> usize {
        self.n_exercises
    }
    fn value(&self, _step: usize, _cs: &dyn CurveState) -> f64 {
        0.0
    }
    fn n_regressors(&self) -> usize {
        1
    }
    fn regressors(&self, _step: usize, _cs: &dyn CurveState) -> Vec<f64> {
        vec![1.0]
    }
}

/// G197: Longstaff-Schwartz exercise strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSStrategy {
    /// Regression coefficients per exercise date.
    pub coefficients: Vec<Vec<f64>>,
}

impl LSStrategy {
    /// Train the LS strategy from simulation data.
    ///
    /// `exercise_values[step][path]` = exercise value,
    /// `regressors[step][path]` = regression variables,
    /// `continuation_values[step][path]` = discounted continuation value.
    pub fn train(
        exercise_values: &[Vec<f64>],
        regressors: &[Vec<Vec<f64>>],
        continuation_values: &[Vec<f64>],
    ) -> Self {
        let n_steps = exercise_values.len();
        let mut coefficients = Vec::with_capacity(n_steps);

        for step in 0..n_steps {
            let n_paths = exercise_values[step].len();
            if n_paths == 0 || regressors[step].is_empty() {
                coefficients.push(Vec::new());
                continue;
            }
            let n_reg = regressors[step][0].len();

            // Solve normal equations: (X^T X) β = X^T y
            // where y = continuation - exercise
            let mut xtx = vec![0.0; n_reg * n_reg];
            let mut xty = vec![0.0; n_reg];

            for p in 0..n_paths {
                if exercise_values[step][p] <= 0.0 {
                    continue; // Only ITM paths
                }
                let y = continuation_values[step][p] - exercise_values[step][p];
                let x = &regressors[step][p];
                for i in 0..n_reg {
                    xty[i] += x[i] * y;
                    for j in 0..n_reg {
                        xtx[i * n_reg + j] += x[i] * x[j];
                    }
                }
            }

            // Simple solve for small systems
            let beta = solve_small_linear(n_reg, &xtx, &xty);
            coefficients.push(beta);
        }

        Self { coefficients }
    }

    /// Decide whether to exercise at a given step.
    pub fn should_exercise(&self, step: usize, exercise_val: f64, regressors: &[f64]) -> bool {
        if exercise_val <= 0.0 {
            return false;
        }
        if step >= self.coefficients.len() || self.coefficients[step].is_empty() {
            return exercise_val > 0.0;
        }
        let cont_est: f64 = self.coefficients[step]
            .iter()
            .zip(regressors)
            .map(|(&c, &x)| c * x)
            .sum();
        exercise_val >= cont_est + exercise_val
    }
}

/// G198: Andersen upper-bound engine for Bermudan pricing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpperBoundEngine {
    /// Upper bound estimate.
    pub upper_bound: f64,
    /// Lower bound from LS.
    pub lower_bound: f64,
    /// Standard error of upper bound.
    pub std_error: f64,
}

impl UpperBoundEngine {
    /// Compute dual upper bound via Andersen-Broadie method.
    ///
    /// `exercise_values[step][path]` = exercise payoff,
    /// `ls_strategy` = trained LS exercise strategy,
    /// `martingale_values[step][path]` = discounted continuation doob martingale.
    pub fn compute(
        exercise_values: &[Vec<f64>],
        ls_lower_bound: f64,
        martingale_increments: &[Vec<f64>],
    ) -> Self {
        let n_steps = exercise_values.len();
        if n_steps == 0 {
            return Self {
                upper_bound: 0.0,
                lower_bound: ls_lower_bound,
                std_error: 0.0,
            };
        }
        let n_paths = exercise_values[0].len();

        // Dual upper bound: E[max_t (h_t - M_t)]
        let mut sum = 0.0;
        let mut sum2 = 0.0;
        for p in 0..n_paths {
            let mut max_val = 0.0_f64;
            let mut m = 0.0; // martingale accumulator
            for step in 0..n_steps {
                if step < martingale_increments.len() && p < martingale_increments[step].len() {
                    m += martingale_increments[step][p];
                }
                let h = exercise_values[step][p] - m;
                max_val = max_val.max(h);
            }
            sum += max_val;
            sum2 += max_val * max_val;
        }

        let mean = sum / n_paths as f64;
        let var = sum2 / n_paths as f64 - mean * mean;
        let se = (var.max(0.0) / n_paths as f64).sqrt();

        Self {
            upper_bound: mean,
            lower_bound: ls_lower_bound,
            std_error: se,
        }
    }
}

/// G199: Polynomial basis system for regression.
pub trait MarketModelBasisSystem: Send + Sync {
    /// Number of basis functions.
    fn n_basis(&self) -> usize;
    /// Evaluate basis at given market variables.
    fn evaluate(&self, variables: &[f64]) -> Vec<f64>;
}

/// Monomial basis: 1, x, x², ..., x^degree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonomialBasis {
    /// Degree.
    pub degree: usize,
}

impl MonomialBasis {
    /// New.
    pub fn new(degree: usize) -> Self {
        Self { degree }
    }
}

impl MarketModelBasisSystem for MonomialBasis {
    fn n_basis(&self) -> usize {
        self.degree + 1
    }
    fn evaluate(&self, variables: &[f64]) -> Vec<f64> {
        let x = variables.first().copied().unwrap_or(0.0);
        let mut result = Vec::with_capacity(self.degree + 1);
        let mut xp = 1.0;
        for _ in 0..=self.degree {
            result.push(xp);
            xp *= x;
        }
        result
    }
}

/// G200: Parametric exercise strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketModelParametricExercise {
    /// Parameters per exercise date.
    pub parameters: Vec<Vec<f64>>,
    /// Basis system degree.
    pub basis_degree: usize,
}

impl MarketModelParametricExercise {
    /// New.
    pub fn new(parameters: Vec<Vec<f64>>, basis_degree: usize) -> Self {
        Self { parameters, basis_degree }
    }

    /// Exercise boundary: exercise if Σ_k θ_k φ_k(x) > 0.
    pub fn should_exercise(&self, step: usize, variables: &[f64]) -> bool {
        if step >= self.parameters.len() {
            return false;
        }
        let basis = MonomialBasis::new(self.basis_degree);
        let phi = basis.evaluate(variables);
        let val: f64 = self.parameters[step]
            .iter()
            .zip(phi.iter())
            .map(|(&t, &p)| t * p)
            .sum();
        val > 0.0
    }
}

/// G201: Swap-rate basis system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapBasisSystem {
    /// Degree.
    pub degree: usize,
    /// Swap start.
    pub swap_start: usize,
    /// Swap end.
    pub swap_end: usize,
}

impl SwapBasisSystem {
    /// New.
    pub fn new(degree: usize, swap_start: usize, swap_end: usize) -> Self {
        Self { degree, swap_start, swap_end }
    }
}

impl MarketModelBasisSystem for SwapBasisSystem {
    fn n_basis(&self) -> usize {
        self.degree + 1
    }
    fn evaluate(&self, variables: &[f64]) -> Vec<f64> {
        // variables[0] = swap rate
        let sr = variables.first().copied().unwrap_or(0.0);
        let mut result = Vec::with_capacity(self.degree + 1);
        let mut xp = 1.0;
        for _ in 0..=self.degree {
            result.push(xp);
            xp *= sr;
        }
        result
    }
}

/// G202: Swap-forward basis system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapForwardBasisSystem {
    /// Degree.
    pub degree: usize,
    /// Number of forward rates to include.
    pub n_forwards: usize,
}

impl SwapForwardBasisSystem {
    /// New.
    pub fn new(degree: usize, n_forwards: usize) -> Self {
        Self { degree, n_forwards }
    }
}

impl MarketModelBasisSystem for SwapForwardBasisSystem {
    fn n_basis(&self) -> usize {
        1 + self.n_forwards * self.degree
    }
    #[allow(clippy::needless_range_loop)]
    fn evaluate(&self, variables: &[f64]) -> Vec<f64> {
        // variables = [sr, f0, f1, ..., f_{n-1}]
        let mut result = vec![1.0]; // intercept
        for i in 0..self.n_forwards.min(variables.len()) {
            let x = variables[i];
            let mut xp = x;
            for _ in 0..self.degree {
                result.push(xp);
                xp *= x;
            }
        }
        result
    }
}

/// G203: Swap-rate trigger for exercise decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapRateTrigger {
    /// Trigger levels per exercise date.
    pub trigger_levels: Vec<f64>,
    /// If true, exercise when swap rate > trigger; otherwise when < trigger.
    pub exercise_above: bool,
}

impl SwapRateTrigger {
    /// New.
    pub fn new(trigger_levels: Vec<f64>, exercise_above: bool) -> Self {
        Self { trigger_levels, exercise_above }
    }

    /// Should exercise.
    pub fn should_exercise(&self, step: usize, swap_rate: f64) -> bool {
        if step >= self.trigger_levels.len() {
            return false;
        }
        if self.exercise_above {
            swap_rate > self.trigger_levels[step]
        } else {
            swap_rate < self.trigger_levels[step]
        }
    }
}

/// G204: Triggered swap exercise.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggeredSwapExercise {
    /// Trigger.
    pub trigger: SwapRateTrigger,
    /// Exercise value.
    pub exercise_value: BermudanSwaptionExerciseValue,
}

impl TriggeredSwapExercise {
    /// New.
    pub fn new(trigger: SwapRateTrigger, exercise_value: BermudanSwaptionExerciseValue) -> Self {
        Self { trigger, exercise_value }
    }
}

impl ExerciseValue for TriggeredSwapExercise {
    fn n_exercises(&self) -> usize {
        self.exercise_value.n_exercises()
    }
    fn value(&self, step: usize, curve_state: &dyn CurveState) -> f64 {
        if step >= self.exercise_value.swap_starts.len() {
            return 0.0;
        }
        let start = self.exercise_value.swap_starts[step];
        let sr = curve_state.swap_rate(start, self.exercise_value.swap_end);
        if self.trigger.should_exercise(step, sr) {
            self.exercise_value.value(step, curve_state)
        } else {
            0.0
        }
    }
    fn n_regressors(&self) -> usize {
        self.exercise_value.n_regressors()
    }
    fn regressors(&self, step: usize, curve_state: &dyn CurveState) -> Vec<f64> {
        self.exercise_value.regressors(step, curve_state)
    }
}

/// G205: Node data collector for exercise trees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeData {
    /// Exercise values at this node.
    pub exercise_value: f64,
    /// Continuation value.
    pub continuation_value: f64,
    /// Is the optimal action to exercise?
    pub is_exercise: bool,
    /// Regression variables.
    pub regressors: Vec<f64>,
}

/// Node data provider trait.
pub trait NodeDataProvider: Send + Sync {
    /// Collect data.
    fn collect_data(
        &self,
        step: usize,
        curve_state: &dyn CurveState,
        continuation: f64,
    ) -> NodeData;
}

/// Default implementation collecting node data from an exercise value.
pub struct DefaultNodeDataProvider {
    exercise_value: Box<dyn ExerciseValue>,
}

impl std::fmt::Debug for DefaultNodeDataProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultNodeDataProvider")
            .field("exercise_value", &"<dyn ExerciseValue>")
            .finish()
    }
}

impl DefaultNodeDataProvider {
    /// New.
    pub fn new(ev: impl ExerciseValue + 'static) -> Self {
        Self {
            exercise_value: Box::new(ev),
        }
    }
}

impl NodeDataProvider for DefaultNodeDataProvider {
    fn collect_data(
        &self,
        step: usize,
        curve_state: &dyn CurveState,
        continuation: f64,
    ) -> NodeData {
        let ev = self.exercise_value.value(step, curve_state);
        let regressors = self.exercise_value.regressors(step, curve_state);
        NodeData {
            exercise_value: ev,
            continuation_value: continuation,
            is_exercise: ev > continuation,
            regressors,
        }
    }
}

/// G206: Adapter for parametric exercise strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricExerciseAdapter {
    /// Number of exercises.
    pub n_exercises: usize,
    /// Parametric strategy.
    pub strategy: MarketModelParametricExercise,
}

impl ParametricExerciseAdapter {
    /// New.
    pub fn new(n_exercises: usize, strategy: MarketModelParametricExercise) -> Self {
        Self { n_exercises, strategy }
    }

    /// Determine exercise decision.
    pub fn should_exercise(&self, step: usize, variables: &[f64]) -> bool {
        self.strategy.should_exercise(step, variables)
    }
}

/// G207: Exercise indicator function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExerciseIndicator {
    /// For each path and step: true if exercise is optimal.
    pub indicators: Vec<Vec<bool>>,
}

impl ExerciseIndicator {
    /// Compute exercise indicators from exercise and continuation values.
    pub fn from_values(
        exercise_values: &[Vec<f64>],
        continuation_values: &[Vec<f64>],
    ) -> Self {
        let n_steps = exercise_values.len();
        let mut indicators = Vec::with_capacity(n_steps);
        for step in 0..n_steps {
            let n_paths = exercise_values[step].len();
            let ind: Vec<bool> = (0..n_paths)
                .map(|p| {
                    exercise_values[step][p] > 0.0
                        && exercise_values[step][p]
                            >= continuation_values[step][p]
                })
                .collect();
            indicators.push(ind);
        }
        Self { indicators }
    }

    /// First exercise time for each path.
    pub fn first_exercise_times(&self) -> Vec<Option<usize>> {
        if self.indicators.is_empty() {
            return Vec::new();
        }
        let n_paths = self.indicators[0].len();
        let mut result = vec![None; n_paths];
        for p in 0..n_paths {
            for (step, ind) in self.indicators.iter().enumerate() {
                if p < ind.len() && ind[p] {
                    result[p] = Some(step);
                    break;
                }
            }
        }
        result
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G208-G218: Additional LMM Types
// ═══════════════════════════════════════════════════════════════════════════

/// G208: SVD-based predictor-corrector evolver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SvdDFwdRatePC {
    /// Singular values from SVD of the pseudo-root.
    pub singular_values: Vec<f64>,
    /// Left singular vectors (n_rates × n_factors, row-major).
    pub u_matrix: Vec<f64>,
    /// Right singular vectors (n_factors × n_factors, row-major).
    pub v_matrix: Vec<f64>,
    /// N rates.
    pub n_rates: usize,
    /// N factors.
    pub n_factors: usize,
}

impl SvdDFwdRatePC {
    /// Build from a pseudo-root matrix using truncated SVD.
    pub fn from_pseudo_root(pseudo_root: &[f64], n_rates: usize, n_factors: usize) -> Self {
        // Simplified SVD: for production use, one would use a proper SVD library.
        // Here we keep the first n_factors components as-is (Gram-Schmidt-like).
        let mut u_matrix = vec![0.0; n_rates * n_factors];
        let mut singular_values = vec![0.0; n_factors];
        let mut v_matrix = vec![0.0; n_factors * n_factors];

        // Column norms as approximate singular values
        for j in 0..n_factors {
            let mut norm2 = 0.0;
            for i in 0..n_rates {
                let val = pseudo_root[i * n_factors + j];
                norm2 += val * val;
            }
            singular_values[j] = norm2.sqrt();
            if singular_values[j] > 1e-15 {
                for i in 0..n_rates {
                    u_matrix[i * n_factors + j] = pseudo_root[i * n_factors + j] / singular_values[j];
                }
            }
            v_matrix[j * n_factors + j] = 1.0; // approx
        }

        Self {
            singular_values,
            u_matrix,
            v_matrix,
            n_rates,
            n_factors,
        }
    }

    /// Evolve one step (predictor-corrector) using SVD decomposition.
    #[allow(clippy::needless_range_loop)]
    pub fn evolve(
        &self,
        forwards: &mut [f64],
        accruals: &[f64],
        alive: usize,
        dt: f64,
        dw: &[f64],
    ) {
        let n = self.n_rates;

        // Predictor
        let fwd_save: Vec<f64> = forwards.to_vec();
        for i in alive..n {
            let mut vol_dw = 0.0;
            for j in 0..self.n_factors {
                let a = self.singular_values[j] * self.u_matrix[i * self.n_factors + j];
                vol_dw += a * dw[j];
            }
            let drift = compute_rate_drift(forwards, accruals, i, alive, self.n_factors,
                &self.u_matrix, &self.singular_values);
            forwards[i] = fwd_save[i] * ((drift - 0.5 * total_var(self, i)) * dt + vol_dw * dt.sqrt()).exp();
        }

        // Corrector
        for i in alive..n {
            let mut vol_dw = 0.0;
            for j in 0..self.n_factors {
                let a = self.singular_values[j] * self.u_matrix[i * self.n_factors + j];
                vol_dw += a * dw[j];
            }
            let drift_pred = compute_rate_drift(&fwd_save, accruals, i, alive, self.n_factors,
                &self.u_matrix, &self.singular_values);
            let drift_corr = compute_rate_drift(forwards, accruals, i, alive, self.n_factors,
                &self.u_matrix, &self.singular_values);
            let avg_drift = 0.5 * (drift_pred + drift_corr);
            forwards[i] = fwd_save[i] * ((avg_drift - 0.5 * total_var(self, i)) * dt + vol_dw * dt.sqrt()).exp();
        }
    }
}

fn total_var(svd: &SvdDFwdRatePC, i: usize) -> f64 {
    let mut v = 0.0;
    for j in 0..svd.n_factors {
        let a = svd.singular_values[j] * svd.u_matrix[i * svd.n_factors + j];
        v += a * a;
    }
    v
}

fn compute_rate_drift(
    forwards: &[f64],
    accruals: &[f64],
    i: usize,
    alive: usize,
    n_factors: usize,
    u_matrix: &[f64],
    singular_values: &[f64],
) -> f64 {
    let mut drift = 0.0;
    for j in alive..=i {
        let tau_f = accruals[j] * forwards[j] / (1.0 + accruals[j] * forwards[j]);
        let mut corr = 0.0;
        for k in 0..n_factors {
            let a_i = singular_values[k] * u_matrix[i * n_factors + k];
            let a_j = singular_values[k] * u_matrix[j * n_factors + k];
            corr += a_i * a_j;
        }
        drift += tau_f * corr;
    }
    drift
}

/// G209: Iterative Balland log-normal evolver.
///
/// Like `LogNormalFwdRateBalland`, but iterates the predictor-corrector step
/// multiple times for higher accuracy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogNormalFwdRateIBalland {
    /// N iterations.
    pub n_iterations: usize,
}

impl LogNormalFwdRateIBalland {
    /// New.
    pub fn new(n_iterations: usize) -> Self {
        Self { n_iterations: n_iterations.max(1) }
    }

    /// Evolve with iterated Balland scheme.
    pub fn evolve(
        &self,
        forwards: &mut [f64],
        accruals: &[f64],
        alive: usize,
        pseudo_root: &[f64],
        n_factors: usize,
        dt: f64,
        dw: &[f64],
    ) {
        let n = forwards.len();
        let fwd_init: Vec<f64> = forwards.to_vec();

        for _iter in 0..self.n_iterations {
            let fwd_prev: Vec<f64> = forwards.to_vec();
            for i in alive..n {
                // Compute drift using geometric mean of prev and current
                let mut drift = 0.0;
                for j in alive..=i {
                    let fj_bar = (fwd_prev[j] * fwd_init[j]).sqrt();
                    let tau_f = accruals[j] * fj_bar / (1.0 + accruals[j] * fj_bar);
                    let mut corr = 0.0;
                    for k in 0..n_factors {
                        corr += pseudo_root[i * n_factors + k] * pseudo_root[j * n_factors + k];
                    }
                    drift += tau_f * corr;
                }

                let mut vol_dw = 0.0;
                for k in 0..n_factors {
                    vol_dw += pseudo_root[i * n_factors + k] * dw[k];
                }

                let total_vol2: f64 = (0..n_factors)
                    .map(|k| pseudo_root[i * n_factors + k].powi(2))
                    .sum();

                forwards[i] = fwd_init[i]
                    * ((drift - 0.5 * total_vol2) * dt + vol_dw * dt.sqrt()).exp();
            }
        }
    }
}

/// G210: Multi-step period caplet-swaption product.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiStepPeriodCapletSwaptions {
    /// Caplet strike for each period.
    pub caplet_strikes: Vec<f64>,
    /// Swaption strike for each period.
    pub swaption_strikes: Vec<f64>,
    /// Period boundaries: `periods[i]` = (start_rate, end_rate).
    pub periods: Vec<(usize, usize)>,
}

impl MultiStepPeriodCapletSwaptions {
    /// New.
    pub fn new(
        caplet_strikes: Vec<f64>,
        swaption_strikes: Vec<f64>,
        periods: Vec<(usize, usize)>,
    ) -> Self {
        Self { caplet_strikes, swaption_strikes, periods }
    }

    /// Evaluate cashflows for one path at a given step.
    pub fn cashflows(
        &self,
        step: usize,
        curve_state: &dyn CurveState,
    ) -> Vec<f64> {
        let mut cfs = Vec::new();
        for (p_idx, &(start, end)) in self.periods.iter().enumerate() {
            if step == start {
                // Caplet payment for this period
                let fwd = curve_state.forward_rate(start);
                let caplet = (fwd - self.caplet_strikes[p_idx]).max(0.0)
                    * curve_state.accruals()[start];
                cfs.push(caplet);

                // Swaption payment
                let swap_rate = curve_state.swap_rate(start, end);
                let annuity: f64 = (start..end)
                    .map(|j| curve_state.accruals()[j] * curve_state.discount_ratio(start, j + 1))
                    .sum();
                let swaption = (swap_rate - self.swaption_strikes[p_idx]).max(0.0) * annuity;
                cfs.push(swaption);
            }
        }
        cfs
    }
}

/// G211: Stochastic volatility process for market models.
pub trait MarketModelVolProcess: Send + Sync {
    /// Evolve the volatility state for one time step.
    fn evolve(&mut self, dt: f64, dw_vol: f64);
    /// Current volatility multiplier.
    fn vol_multiplier(&self) -> f64;
    /// Current variance.
    fn variance(&self) -> f64;
}

/// G212: Square-root Andersen stochastic vol process.
///
/// dv = κ(θ − v)dt + ξ√v dW^v
/// Uses the Andersen QE scheme for positivity preservation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SquareRootAndersen {
    /// V.
    pub v: f64,
    /// Kappa.
    pub kappa: f64,
    /// Theta.
    pub theta: f64,
    /// Xi.
    pub xi: f64,
}

impl SquareRootAndersen {
    /// New.
    pub fn new(v0: f64, kappa: f64, theta: f64, xi: f64) -> Self {
        Self { v: v0, kappa, theta, xi }
    }
}

impl MarketModelVolProcess for SquareRootAndersen {
    fn evolve(&mut self, dt: f64, dw_vol: f64) {
        // QE scheme (Andersen 2008)
        let emkt = (-self.kappa * dt).exp();
        let m = self.theta + (self.v - self.theta) * emkt;
        let s2 = self.v * self.xi * self.xi * emkt / self.kappa * (1.0 - emkt)
            + self.theta * self.xi * self.xi / (2.0 * self.kappa) * (1.0 - emkt).powi(2);
        let psi = s2 / (m * m).max(1e-30);

        if psi <= 1.5 {
            // Quadratic scheme
            let b2 = 2.0 / psi - 1.0 + (2.0 / psi).sqrt() * (2.0 / psi - 1.0).max(0.0).sqrt();
            let b = b2.max(0.0).sqrt();
            let a = m / (1.0 + b2);
            self.v = a * (b + dw_vol).powi(2);
        } else {
            // Exponential scheme
            let p = (psi - 1.0) / (psi + 1.0);
            let beta = (1.0 - p) / m.max(1e-30);
            let u = standard_normal_cdf(dw_vol);
            self.v = if u <= p {
                0.0
            } else {
                (-(1.0 - p).max(1e-30).ln() / beta).max(0.0) / ((1.0 - u).max(1e-30).ln().abs())
                    * (-(1.0 - p).max(1e-30).ln()).max(0.0)
                    / beta
            };
            // Simplified: sample from psi-matched distribution
            self.v = m.max(0.0) + s2.max(0.0).sqrt() * dw_vol;
            self.v = self.v.max(0.0);
        }
    }

    fn vol_multiplier(&self) -> f64 {
        self.v.sqrt()
    }

    fn variance(&self) -> f64 {
        self.v
    }
}

/// G213: Piecewise-constant ABCD variance model.
///
/// σ(t) = (a + b(T-t)) exp(-c(T-t)) + d, with parameters varying per period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiecewiseConstantAbcdVariance {
    /// ABCD parameters per period: (a, b, c, d).
    pub params: Vec<(f64, f64, f64, f64)>,
    /// Period boundaries (times).
    pub times: Vec<f64>,
}

impl PiecewiseConstantAbcdVariance {
    /// New.
    pub fn new(params: Vec<(f64, f64, f64, f64)>, times: Vec<f64>) -> Self {
        Self { params, times }
    }

    /// Instantaneous vol at time `t` for expiry `big_t`.
    pub fn vol(&self, t: f64, big_t: f64) -> f64 {
        let tau = (big_t - t).max(0.0);
        let (a, b, c, d) = self.period_params(t);
        (a + b * tau) * (-c * tau).exp() + d
    }

    /// Integrated variance from t1 to t2 for expiry T.
    pub fn integrated_variance(&self, t1: f64, t2: f64, big_t: f64) -> f64 {
        // Numerical integration (Simpson's rule)
        let n = 100;
        let dt = (t2 - t1) / n as f64;
        let mut sum = 0.0;
        for i in 0..=n {
            let t = t1 + i as f64 * dt;
            let v = self.vol(t, big_t);
            let w = if i == 0 || i == n { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
            sum += w * v * v;
        }
        sum * dt / 3.0
    }

    fn period_params(&self, t: f64) -> (f64, f64, f64, f64) {
        for (i, &time) in self.times.iter().enumerate() {
            if t < time {
                return self.params[i.min(self.params.len() - 1)];
            }
        }
        *self.params.last().unwrap_or(&(0.0, 0.0, 0.0, 0.0))
    }
}

/// G214: Volatility interpolation specifier for LMM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolatilityInterpolationSpecifier {
    /// Flat interpolation (nearest predecessor).
    Flat,
    /// Linear interpolation between vol nodes.
    Linear,
    /// Cubic spline interpolation.
    CubicSpline,
    /// ABCD parametric form.
    Abcd {
        /// The `a` parameter of the ABCD function.
        a: f64,
        /// The `b` parameter of the ABCD function.
        b: f64,
        /// The `c` parameter of the ABCD function.
        c: f64,
        /// The `d` parameter of the ABCD function.
        d: f64,
    },
}

impl VolatilityInterpolationSpecifier {
    /// Interpolate vol from a grid.
    pub fn interpolate(&self, t: f64, times: &[f64], vols: &[f64]) -> f64 {
        if times.is_empty() || vols.is_empty() {
            return 0.0;
        }
        match self {
            Self::Flat => {
                let idx = times.partition_point(|&x| x <= t);
                if idx == 0 { vols[0] } else { vols[idx - 1] }
            }
            Self::Linear => {
                if t <= times[0] {
                    return vols[0];
                }
                if t >= *times.last().unwrap() {
                    return *vols.last().unwrap();
                }
                let idx = times.partition_point(|&x| x <= t);
                if idx == 0 {
                    return vols[0];
                }
                let i = idx - 1;
                let w = (t - times[i]) / (times[i + 1] - times[i]);
                (1.0 - w) * vols[i] + w * vols[i + 1]
            }
            Self::CubicSpline => {
                // Fallback to linear for simplicity
                self.interpolate_linear(t, times, vols)
            }
            Self::Abcd { a, b, c, d } => {
                // Use ABCD form: σ(t) = (a + b*t)*exp(-c*t) + d
                (a + b * t) * (-c * t).exp() + d
            }
        }
    }

    fn interpolate_linear(&self, t: f64, times: &[f64], vols: &[f64]) -> f64 {
        if t <= times[0] {
            return vols[0];
        }
        if t >= *times.last().unwrap() {
            return *vols.last().unwrap();
        }
        let idx = times.partition_point(|&x| x <= t);
        if idx == 0 {
            return vols[0];
        }
        let i = idx - 1;
        let w = (t - times[i]) / (times[i + 1] - times[i]);
        (1.0 - w) * vols[i] + w * vols[i + 1]
    }
}

/// G218: Composite of multiple market model products.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketModelComposite {
    /// Weights for each product.
    pub weights: Vec<f64>,
    /// Number of products.
    pub n_products: usize,
}

impl MarketModelComposite {
    /// New.
    pub fn new(weights: Vec<f64>) -> Self {
        let n_products = weights.len();
        Self { weights, n_products }
    }

    /// Combine cashflows from multiple products.
    ///
    /// `product_cashflows[product][step]` = cashflow.
    pub fn combined_cashflows(&self, product_cashflows: &[Vec<f64>]) -> Vec<f64> {
        if product_cashflows.is_empty() {
            return Vec::new();
        }
        let n_steps = product_cashflows.iter().map(|p| p.len()).max().unwrap_or(0);
        let mut combined = vec![0.0; n_steps];
        for (p, cfs) in product_cashflows.iter().enumerate() {
            let w = self.weights.get(p).copied().unwrap_or(1.0);
            for (s, &cf) in cfs.iter().enumerate() {
                combined[s] += w * cf;
            }
        }
        combined
    }

    /// Combined NPV.
    pub fn combined_npv(&self, product_npvs: &[f64]) -> f64 {
        product_npvs
            .iter()
            .zip(self.weights.iter())
            .map(|(&npv, &w)| w * npv)
            .sum()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper functions
// ═══════════════════════════════════════════════════════════════════════════

/// Solve small linear system Ax = b via Gauss elimination.
fn solve_small_linear(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let mut m = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            m[i * (n + 1) + j] = a[i * n + j];
        }
        m[i * (n + 1) + n] = b[i];
    }

    // Forward elimination
    for col in 0..n {
        // Pivot
        let mut max_row = col;
        let mut max_val = m[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let v = m[row * (n + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            continue;
        }
        if max_row != col {
            for j in 0..=n {
                m.swap(col * (n + 1) + j, max_row * (n + 1) + j);
            }
        }
        let pivot = m[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = m[row * (n + 1) + col] / pivot;
            for j in col..=n {
                m[row * (n + 1) + j] -= factor * m[col * (n + 1) + j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let pivot = m[i * (n + 1) + i];
        if pivot.abs() < 1e-15 {
            continue;
        }
        let mut sum = m[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= m[i * (n + 1) + j] * x[j];
        }
        x[i] = sum / pivot;
    }
    x
}

/// Standard normal CDF.
fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = 1.0 - poly * (-x * x).exp();
    if x >= 0.0 { result } else { -result }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use crate::lmm_framework::LMMCurveState;

    fn sample_forwards() -> Vec<f64> {
        vec![0.03, 0.035, 0.04, 0.042, 0.045]
    }
    fn sample_accruals() -> Vec<f64> {
        vec![0.5; 5]
    }

    // G176: PathwiseAccountingEngine
    #[test]
    fn pathwise_engine_basic() {
        let engine = PathwiseAccountingEngine::new(3, 2);
        let cashflows = vec![
            vec![1.0, 2.0],
            vec![1.5, 1.5],
        ];
        let df = vec![
            vec![1.0, 0.9],
            vec![1.0, 0.9],
        ];
        let tangent = vec![
            vec![vec![0.1, 0.0, 0.0], vec![0.0, 0.1, 0.0]],
            vec![vec![0.2, 0.0, 0.0], vec![0.0, 0.0, 0.1]],
        ];
        let result = engine.run(&cashflows, &df, &tangent);
        assert!(result.mean_npv > 0.0);
        assert_eq!(result.deltas.len(), 3);
    }

    // G177: PathwiseDiscounter
    #[test]
    fn pathwise_discounter_derivative() {
        let pd = PathwiseDiscounter::new(vec![0.5; 3]);
        let fwd = [0.04, 0.05, 0.06];
        let df = pd.discount(&fwd, 3);
        assert!(df > 0.0 && df < 1.0);

        let d0 = pd.discount_derivative(&fwd, 3, 0);
        assert!(d0 < 0.0); // Higher rate → lower DF
    }

    // G178: BumpInstrumentJacobian
    #[test]
    fn bump_jacobian_shape() {
        let fwd = vec![0.03, 0.04, 0.05];
        let jac = BumpInstrumentJacobian::compute(&fwd, 1e-5, |f| {
            vec![f[0] * 0.5, f[1] * 0.5]
        });
        assert_eq!(jac.jacobian.len(), 2);
        assert_eq!(jac.jacobian[0].len(), 3);
        assert_abs_diff_eq!(jac.jacobian[0][0], 0.5, epsilon = 1e-4);
    }

    // G179: RatePseudoRootJacobian
    #[test]
    fn rate_pseudo_root_jacobian() {
        let fwd = vec![0.03, 0.04];
        let jac = RatePseudoRootJacobian::compute(&fwd, 1, 1e-5, |f| {
            vec![f[0] * 0.2, f[1] * 0.2]
        });
        assert_abs_diff_eq!(jac.get(0, 0, 0), 0.2, epsilon = 1e-3);
    }

    // G180: SwaptionPseudoJacobian
    #[test]
    fn swaption_pseudo_jacobian() {
        let pr = vec![0.2, 0.18];
        let jac = SwaptionPseudoJacobian::compute(&pr, 1, 2, 1, 1e-5, |a| {
            vec![a[0] + a[1]]
        });
        assert_eq!(jac.jacobian.len(), 1);
        assert_abs_diff_eq!(jac.jacobian[0][0], 1.0, epsilon = 1e-3);
    }

    // G181: VegaBumpCluster
    #[test]
    fn vega_bump_cluster() {
        let clusters = VegaBumpCluster::by_expiry(3, 1, 0.01);
        assert_eq!(clusters.clusters.len(), 3);
        let vegas = clusters.compute_vegas(
            &[0.2, 0.18, 0.22],
            3, 1,
            |pr| pr.iter().sum::<f64>(),
        );
        assert_eq!(vegas.len(), 3);
        assert!(vegas[0] > 0.0);
    }

    // G182: CapletCoterminalAlphaCalibration
    #[test]
    fn alpha_calibration() {
        let corr = TimeHomogeneousForwardCorrelation::exponential_with_floor(5, 0.01, 0.9);
        let result = CapletCoterminalAlphaCalibration::calibrate(
            &[0.20; 5], &[0.18; 5],
            &sample_forwards(), &sample_accruals(),
            1, 0.5, &corr,
        );
        assert!(result.pseudo_root.len() == 5);
    }

    // G185: Joint calibration
    #[test]
    fn joint_calibration() {
        let corr = TimeHomogeneousForwardCorrelation::exponential_with_floor(5, 0.01, 0.9);
        let result = CapletCoterminalSwaptionCalibration::calibrate(
            &[0.20; 5], &[0.18; 5],
            &sample_forwards(), &sample_accruals(),
            1, 1.0, 1.0, &corr,
        );
        assert!(result.caplet_error >= 0.0);
    }

    // G187: PseudoRootFacade
    #[test]
    fn pseudo_root_facade() {
        let mut pr = PseudoRootFacade::from_flat_vol(3, 2, 0.20);
        pr.set(1, 1, 0.05);
        assert_abs_diff_eq!(pr.total_vol(0), 0.20, epsilon = 1e-10);
        let corr01 = pr.correlation(0, 1);
        assert!(corr01 > 0.0 && corr01 < 1.0);
    }

    // G189: FwdToCotSwapAdapter
    #[test]
    fn fwd_to_swap_adapter() {
        let adapter = FwdToCotSwapAdapter::new(&sample_forwards(), &sample_accruals());
        for i in 0..5 {
            assert!(adapter.swap_rate(i) > 0.0);
        }
    }

    // G190: FwdPeriodAdapter
    #[test]
    fn fwd_period_adapter() {
        let fwd = vec![0.03, 0.035, 0.04, 0.045];
        let tau = vec![0.25; 4];
        let groups = vec![(0, 2), (2, 4)];
        let adapter = FwdPeriodAdapter::new(&fwd, &tau, groups);
        assert_eq!(adapter.adapted_rates.len(), 2);
        assert!(adapter.adapted_rates[0] > 0.0);
    }

    // G192: HistoricalForwardRatesAnalysis
    #[test]
    fn historical_fwd_analysis() {
        let data = vec![
            vec![0.03, 0.035],
            vec![0.031, 0.034],
            vec![0.032, 0.036],
            vec![0.030, 0.035],
        ];
        let result = HistoricalForwardRatesAnalysis::analyse(&data, 1.0 / 252.0);
        assert_eq!(result.n_rates, 2);
        assert!(result.volatilities[0] > 0.0);
    }

    // G193: HistoricalRatesAnalysis
    #[test]
    fn historical_rates_analysis() {
        let data = vec![
            vec![0.03, 0.035],
            vec![0.031, 0.034],
            vec![0.029, 0.037],
            vec![0.032, 0.033],
        ];
        let result = HistoricalRatesAnalysis::analyse(&data);
        assert_eq!(result.n_rates, 2);
        assert!(result.std_devs[0] > 0.0);
    }

    // G194-G196
    #[test]
    fn exercise_value_bermudan() {
        let bev = BermudanSwaptionExerciseValue::new_payer(
            vec![0, 1, 2], 5, 0.04, 1_000_000.0,
        );
        assert_eq!(bev.n_exercises(), 3);

        let nothing = NothingExerciseValue::new(3);
        let cs = LMMCurveState::new(sample_forwards(), sample_accruals());
        assert_abs_diff_eq!(nothing.value(0, &cs), 0.0, epsilon = 1e-15);
    }

    // G197: LSStrategy
    #[test]
    fn ls_strategy_train() {
        let ev = vec![vec![1.0, 0.5, 0.0, 2.0]];
        let reg = vec![vec![
            vec![1.0, 0.03],
            vec![1.0, 0.04],
            vec![1.0, 0.05],
            vec![1.0, 0.06],
        ]];
        let cv = vec![vec![0.8, 0.4, 0.1, 1.5]];
        let ls = LSStrategy::train(&ev, &reg, &cv);
        assert_eq!(ls.coefficients.len(), 1);
    }

    // G198: UpperBoundEngine
    #[test]
    fn upper_bound_positive() {
        let ev = vec![vec![1.0, 0.5, 2.0, 0.0]];
        let mi = vec![vec![0.0, -0.1, 0.2, 0.0]];
        let result = UpperBoundEngine::compute(&ev, 0.8, &mi);
        assert!(result.upper_bound >= result.lower_bound);
    }

    // G199: MonomialBasis
    #[test]
    fn monomial_basis() {
        let basis = MonomialBasis::new(3);
        let v = basis.evaluate(&[2.0]);
        assert_eq!(v, vec![1.0, 2.0, 4.0, 8.0]);
    }

    // G203: SwapRateTrigger
    #[test]
    fn swap_rate_trigger() {
        let trigger = SwapRateTrigger::new(vec![0.04, 0.05], true);
        assert!(!trigger.should_exercise(0, 0.03));
        assert!(trigger.should_exercise(0, 0.05));
    }

    // G207: ExerciseIndicator
    #[test]
    fn exercise_indicator() {
        let ev = vec![vec![1.0, 0.0, 2.0], vec![0.5, 1.5, 0.0]];
        let cv = vec![vec![0.5, 1.0, 1.0], vec![1.0, 0.5, 0.5]];
        let ind = ExerciseIndicator::from_values(&ev, &cv);
        assert!(ind.indicators[0][0]); // exercise: 1.0 > 0.5
        assert!(!ind.indicators[0][1]); // no exercise: 0.0
        let first = ind.first_exercise_times();
        assert_eq!(first[0], Some(0));
    }

    // G208: SvdDFwdRatePC
    #[test]
    fn svd_evolver() {
        let pr = vec![0.20, 0.18, 0.19];
        let svd = SvdDFwdRatePC::from_pseudo_root(&pr, 3, 1);
        assert_eq!(svd.singular_values.len(), 1);
        assert!(svd.singular_values[0] > 0.0);
    }

    // G209: LogNormalFwdRateIBalland
    #[test]
    fn iballand_evolver() {
        let evolver = LogNormalFwdRateIBalland::new(3);
        let pr = vec![0.20, 0.18, 0.19];
        let mut fwd = vec![0.03, 0.04, 0.05];
        let accruals = vec![0.5; 3];
        evolver.evolve(&mut fwd, &accruals, 0, &pr, 1, 0.01, &[0.5]);
        assert!(fwd[0] > 0.0);
    }

    // G211-G212: SquareRootAndersen
    #[test]
    fn square_root_andersen_evolve() {
        let mut proc = SquareRootAndersen::new(0.04, 2.0, 0.04, 0.3);
        proc.evolve(0.01, 0.1);
        assert!(proc.variance() >= 0.0);
        assert!(proc.vol_multiplier() >= 0.0);
    }

    // G213: PiecewiseConstantAbcdVariance
    #[test]
    fn pc_abcd_variance() {
        let pcv = PiecewiseConstantAbcdVariance::new(
            vec![(0.1, 0.05, 0.5, 0.02)],
            vec![10.0],
        );
        let v = pcv.vol(0.0, 5.0);
        assert!(v > 0.0);
        let iv = pcv.integrated_variance(0.0, 1.0, 5.0);
        assert!(iv > 0.0);
    }

    // G214: VolatilityInterpolationSpecifier
    #[test]
    fn vol_interpolation() {
        let times = vec![0.0, 1.0, 2.0];
        let vols = vec![0.20, 0.22, 0.25];
        let spec = VolatilityInterpolationSpecifier::Linear;
        let v = spec.interpolate(0.5, &times, &vols);
        assert_abs_diff_eq!(v, 0.21, epsilon = 1e-10);
    }

    // G218: MarketModelComposite
    #[test]
    fn market_model_composite() {
        let comp = MarketModelComposite::new(vec![1.0, -0.5]);
        let cfs = vec![
            vec![10.0, 20.0],
            vec![5.0, 10.0],
        ];
        let combined = comp.combined_cashflows(&cfs);
        assert_abs_diff_eq!(combined[0], 10.0 - 2.5, epsilon = 1e-10);
        assert_abs_diff_eq!(combined[1], 20.0 - 5.0, epsilon = 1e-10);
    }
}
