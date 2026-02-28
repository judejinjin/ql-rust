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
pub mod lmm_extensions;
pub mod gsr;
pub mod ptd_heston_model;
pub mod gjr_garch_model;
pub mod variance_gamma_model;
pub mod basis_model;
pub mod vol_estimators;
pub mod models_extended;

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
pub use lmm::{LmmConfig, LmmCurveState, LmmResult, lmm_cap_price, lmm_swaption_price, evolve_one_step};
pub use lmm_extensions::{
    spot_measure_drift, evolve_one_step_spot_measure,
    lmm_ratchet_cap_price, lmm_cms_rate,
    lmm_bermudan_swaption_price, RebonatVolSurface,
};
pub use gsr::{Gsr1d, MarkovFunctional};
pub use ptd_heston_model::{PtdHestonModel, PtdHestonParamSlice};
pub use gjr_garch_model::GjrGarchModel;
pub use variance_gamma_model::VarianceGammaModel;
pub use vol_estimators::{
    GarchParams, garch_fit, garch_forecast,
    garman_klass_vol, rogers_satchell_vol, yang_zhang_vol,
};
pub use basis_model::{
    FraOisBasisModel, CrossCurrencyBasisModel, CcyPair,
    BasisSwapResult, price_basis_swap,
};
pub use models_extended::{
    HestonSlvFdmModel, HestonSlvMcModel, ExtendedCoxIngersollRoss,
    CapHelper, SwaptionHelper, SwaptionVolType, ConstantVolEstimator,
};
