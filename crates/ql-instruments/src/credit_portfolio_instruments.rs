//! Portfolio credit instrument types: Synthetic CDO, Nth-to-Default, CDS Option.
//!
//! Reference: QuantLib credit/ — SyntheticCDO, NthToDefault, CdsOption.

use serde::{Deserialize, Serialize};
use ql_time::Date;

use crate::credit_default_swap::CdsProtectionSide;

// =========================================================================
// Default event / probability key
// =========================================================================

/// Type of default event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DefaultType {
    /// Failure to pay.
    FailureToPay,
    /// Bankruptcy.
    Bankruptcy,
    /// Restructuring.
    Restructuring,
    /// Obligation acceleration.
    ObligationAcceleration,
    /// Cross-default.
    CrossDefault,
    /// Any event.
    Any,
}

/// A recorded default event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultEvent {
    /// Date of the default.
    pub default_date: Date,
    /// Recovery rate realized at settlement.
    pub recovery_rate: f64,
    /// Type of default.
    pub default_type: DefaultType,
    /// Issuer name / identifier.
    pub issuer: String,
}

/// Key identifying a default probability curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultProbabilityKey {
    /// Seniority level.
    pub seniority: u32,
    /// Currency code.
    pub currency: String,
    /// Default type.
    pub default_type: DefaultType,
}

/// Recovery rate model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryRateModel {
    /// Constant recovery.
    Constant(f64),
    /// Stochastic recovery: mean + std dev (normal).
    Stochastic { mean: f64, std_dev: f64 },
    /// Recovery depends on seniority.
    Seniority(Vec<(u32, f64)>),
}

impl RecoveryRateModel {
    /// Expected recovery rate.
    pub fn expected(&self) -> f64 {
        match self {
            Self::Constant(r) => *r,
            Self::Stochastic { mean, .. } => *mean,
            Self::Seniority(v) => {
                if v.is_empty() {
                    0.4
                } else {
                    v.iter().map(|(_, r)| r).sum::<f64>() / v.len() as f64
                }
            }
        }
    }
}

/// Recovery claim type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimType {
    /// Face value claim.
    FaceValue,
    /// Par claim.
    Par,
    /// Custom fraction of notional.
    Custom,
}

// =========================================================================
// Pool definitions
// =========================================================================

/// A credit pool — collection of issuers for portfolio products.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pool {
    /// Issuer names.
    pub names: Vec<String>,
    /// Notional per name.
    pub notionals: Vec<f64>,
    /// Default probability per name.
    pub default_probabilities: Vec<f64>,
    /// Recovery rate per name.
    pub recovery_rates: Vec<f64>,
    /// Seniority per name.
    pub seniorities: Vec<u32>,
}

impl Pool {
    /// Number of names in the pool.
    pub fn size(&self) -> usize {
        self.names.len()
    }

    /// Total notional.
    pub fn total_notional(&self) -> f64 {
        self.notionals.iter().sum()
    }

    /// Average default probability.
    pub fn avg_default_prob(&self) -> f64 {
        if self.default_probabilities.is_empty() {
            return 0.0;
        }
        self.default_probabilities.iter().sum::<f64>() / self.default_probabilities.len() as f64
    }

    /// Average recovery.
    pub fn avg_recovery(&self) -> f64 {
        if self.recovery_rates.is_empty() {
            return 0.4;
        }
        self.recovery_rates.iter().sum::<f64>() / self.recovery_rates.len() as f64
    }
}

/// Homogeneous pool: all names have same default prob and recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomogeneousPoolDef {
    /// Number of names.
    pub n_names: usize,
    /// Notional per name.
    pub notional_per_name: f64,
    /// Common default probability.
    pub default_probability: f64,
    /// Common recovery rate.
    pub recovery_rate: f64,
}

impl HomogeneousPoolDef {
    /// Convert to a Pool.
    pub fn to_pool(&self) -> Pool {
        Pool {
            names: (0..self.n_names).map(|i| format!("Name_{}", i)).collect(),
            notionals: vec![self.notional_per_name; self.n_names],
            default_probabilities: vec![self.default_probability; self.n_names],
            recovery_rates: vec![self.recovery_rate; self.n_names],
            seniorities: vec![1; self.n_names],
        }
    }

    pub fn total_notional(&self) -> f64 {
        self.n_names as f64 * self.notional_per_name
    }
}

/// Inhomogeneous pool: each name has distinct parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InhomogeneousPoolDef {
    pub pool: Pool,
}

impl InhomogeneousPoolDef {
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }

    pub fn total_notional(&self) -> f64 {
        self.pool.total_notional()
    }
}

// =========================================================================
// Synthetic CDO
// =========================================================================

/// A Synthetic CDO tranche instrument.
///
/// Reference: QuantLib `SyntheticCDO`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticCDO {
    /// Protection side.
    pub side: CdsProtectionSide,
    /// Attachment point (as fraction of portfolio notional).
    pub attachment: f64,
    /// Detachment point.
    pub detachment: f64,
    /// Running spread (coupon).
    pub running_spread: f64,
    /// Upfront payment (as fraction of tranche notional, positive = buyer pays).
    pub upfront: f64,
    /// Portfolio notional.
    pub portfolio_notional: f64,
    /// Maturity date.
    pub maturity: Date,
    /// Premium payment dates.
    pub premium_dates: Vec<Date>,
    /// Day count fractions for each premium period.
    pub accrual_fractions: Vec<f64>,
    /// Whether accrued premium is paid on default.
    pub settles_accrual: bool,
    /// Whether protection starts from day one.
    pub protection_start: bool,
}

impl SyntheticCDO {
    /// Tranche notional = (detachment − attachment) × portfolio notional.
    pub fn tranche_notional(&self) -> f64 {
        (self.detachment - self.attachment) * self.portfolio_notional
    }

    /// Tranche width.
    pub fn width(&self) -> f64 {
        self.detachment - self.attachment
    }
}

/// CDO pricing result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticCDOResult {
    /// NPV of the tranche.
    pub npv: f64,
    /// Fair running spread.
    pub fair_spread: f64,
    /// Fair upfront payment.
    pub fair_upfront: f64,
    /// Protection leg PV.
    pub protection_leg: f64,
    /// Premium leg PV per unit spread.
    pub premium_leg: f64,
    /// Expected tranche loss.
    pub expected_tranche_loss: f64,
    /// Tranche delta with respect to correlation.
    pub correlation_delta: f64,
}

// =========================================================================
// Nth-to-Default
// =========================================================================

/// Nth-to-Default basket instrument.
///
/// Reference: QuantLib `NthToDefault`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NthToDefault {
    /// Protection side.
    pub side: CdsProtectionSide,
    /// Which default triggers payment (1 = first-to-default).
    pub nth: usize,
    /// Running spread.
    pub running_spread: f64,
    /// Portfolio notional (total across all names).
    pub notional: f64,
    /// Maturity date.
    pub maturity: Date,
    /// Premium payment dates.
    pub premium_dates: Vec<Date>,
    /// Day count fractions.
    pub accrual_fractions: Vec<f64>,
    /// Whether accrued is paid on default.
    pub settles_accrual: bool,
}

/// Nth-to-Default pricing result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NthToDefaultResult {
    /// NPV.
    pub npv: f64,
    /// Fair spread.
    pub fair_spread: f64,
    /// Protection leg PV.
    pub protection_leg: f64,
    /// Premium leg PV per unit spread.
    pub premium_leg: f64,
    /// Expected number of defaults.
    pub expected_defaults: f64,
}

// =========================================================================
// CDS Option
// =========================================================================

/// CDS option type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CdsOptionType {
    /// Right to buy protection at the strike spread.
    Payer,
    /// Right to sell protection at the strike spread.
    Receiver,
}

/// An option on a Credit Default Swap.
///
/// Reference: QuantLib `CdsOption`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdsOption {
    /// Option type (payer or receiver).
    pub option_type: CdsOptionType,
    /// Strike spread.
    pub strike_spread: f64,
    /// Option expiry date.
    pub expiry: Date,
    /// Underlying CDS notional.
    pub notional: f64,
    /// Underlying CDS maturity.
    pub underlying_maturity: Date,
    /// Underlying recovery rate assumption.
    pub recovery_rate: f64,
    /// Whether the option is knock-out on default.
    pub knockout: bool,
}

/// CDS Option pricing result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdsOptionResult {
    /// Option NPV.
    pub npv: f64,
    /// Implied volatility (if computed from market price).
    pub implied_vol: Option<f64>,
    /// Forward spread used.
    pub forward_spread: f64,
    /// RPV01 (risky annuity).
    pub rpv01: f64,
    /// Vega (dNPV/dvol).
    pub vega: f64,
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::Month;

    #[test]
    fn test_homogeneous_pool() {
        let pool_def = HomogeneousPoolDef {
            n_names: 125,
            notional_per_name: 1_000_000.0,
            default_probability: 0.02,
            recovery_rate: 0.40,
        };
        let pool = pool_def.to_pool();
        assert_eq!(pool.size(), 125);
        assert!((pool.total_notional() - 125_000_000.0).abs() < 1.0);
        assert!((pool.avg_default_prob() - 0.02).abs() < 1e-10);
        assert!((pool.avg_recovery() - 0.40).abs() < 1e-10);
    }

    #[test]
    fn test_inhomogeneous_pool() {
        let pool = Pool {
            names: vec!["A".into(), "B".into(), "C".into()],
            notionals: vec![1e6, 2e6, 3e6],
            default_probabilities: vec![0.01, 0.03, 0.05],
            recovery_rates: vec![0.4, 0.3, 0.5],
            seniorities: vec![1, 1, 2],
        };
        assert_eq!(pool.size(), 3);
        assert!((pool.total_notional() - 6e6).abs() < 1.0);
        assert!((pool.avg_default_prob() - 0.03).abs() < 1e-10);
    }

    #[test]
    fn test_synthetic_cdo() {
        let cdo = SyntheticCDO {
            side: CdsProtectionSide::Buyer,
            attachment: 0.03,
            detachment: 0.07,
            running_spread: 0.01,
            upfront: 0.0,
            portfolio_notional: 1e9,
            maturity: Date::from_ymd(2030, Month::June, 20),
            premium_dates: vec![],
            accrual_fractions: vec![],
            settles_accrual: true,
            protection_start: true,
        };
        assert!((cdo.tranche_notional() - 40_000_000.0).abs() < 1.0);
        assert!((cdo.width() - 0.04).abs() < 1e-10);
    }

    #[test]
    fn test_nth_to_default() {
        let ntd = NthToDefault {
            side: CdsProtectionSide::Buyer,
            nth: 1,
            running_spread: 0.005,
            notional: 5_000_000.0,
            maturity: Date::from_ymd(2029, Month::December, 20),
            premium_dates: vec![],
            accrual_fractions: vec![],
            settles_accrual: true,
        };
        assert_eq!(ntd.nth, 1);
    }

    #[test]
    fn test_cds_option() {
        let opt = CdsOption {
            option_type: CdsOptionType::Payer,
            strike_spread: 0.01,
            expiry: Date::from_ymd(2025, Month::June, 20),
            notional: 10_000_000.0,
            underlying_maturity: Date::from_ymd(2030, Month::June, 20),
            recovery_rate: 0.40,
            knockout: true,
        };
        assert_eq!(opt.option_type, CdsOptionType::Payer);
        assert!(opt.knockout);
    }

    #[test]
    fn test_recovery_rate_model() {
        let constant = RecoveryRateModel::Constant(0.40);
        assert!((constant.expected() - 0.40).abs() < 1e-10);

        let stochastic = RecoveryRateModel::Stochastic { mean: 0.35, std_dev: 0.10 };
        assert!((stochastic.expected() - 0.35).abs() < 1e-10);

        let seniority = RecoveryRateModel::Seniority(vec![(1, 0.40), (2, 0.25)]);
        assert!((seniority.expected() - 0.325).abs() < 1e-10);
    }

    #[test]
    fn test_default_event() {
        let ev = DefaultEvent {
            default_date: Date::from_ymd(2024, Month::March, 15),
            recovery_rate: 0.35,
            default_type: DefaultType::Bankruptcy,
            issuer: "Acme Corp".into(),
        };
        assert_eq!(ev.default_type, DefaultType::Bankruptcy);
    }
}
