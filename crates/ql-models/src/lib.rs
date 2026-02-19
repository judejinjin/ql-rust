//! # ql-models
//!
//! Calibrated model framework and implementations for derivative pricing.
//!
//! ## Overview
//!
//! ### Framework
//! - [`CalibratedModel`] — trait for models with calibratable parameters
//! - [`CalibrationHelper`] — market instrument used as calibration target
//! - [`Parameter`] / [`Constraint`] — parameter with bounds and transforms
//! - [`calibrate`] — generic Levenberg-Marquardt calibration driver
//!
//! ### Stochastic Volatility
//! - [`HestonModel`] — Heston (1993) stochastic variance model
//! - [`BatesModel`] — Bates = Heston + Merton jumps
//!
//! ### Short-Rate Models
//! - [`HullWhiteModel`] — Hull-White (extended Vasicek)
//! - [`VasicekModel`] — Vasicek mean-reverting model
//! - [`CIRModel`] — Cox-Ingersoll-Ross model
//! - [`BlackKarasinskiModel`] — log-normal short rate
//! - [`G2Model`] — two-factor Gaussian short-rate model
//!
//! ### LIBOR Market Model
//! - [`lmm`] — LMM framework for forward rates (caps, swaptions)

pub mod parameter;
pub mod calibrated_model;
pub mod heston_model;
pub mod hull_white_model;
pub mod bates_model;
pub mod vasicek;
pub mod cir;
pub mod black_karasinski;
pub mod g2_model;
pub mod lmm;

// Re-exports
pub use parameter::{Parameter, Constraint, NoConstraint, PositiveConstraint, BoundaryConstraint, CompositeConstraint};
pub use calibrated_model::{CalibratedModel, CalibrationHelper, ShortRateModel, calibrate};
pub use heston_model::HestonModel;
pub use hull_white_model::HullWhiteModel;
pub use bates_model::BatesModel;
pub use vasicek::VasicekModel;
pub use cir::CIRModel;
pub use black_karasinski::BlackKarasinskiModel;
pub use g2_model::G2Model;
pub use lmm::{LmmConfig, LmmCurveState, LmmResult, lmm_cap_price, lmm_swaption_price};
