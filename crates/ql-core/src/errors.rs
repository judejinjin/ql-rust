//! Error types for the ql-rust library.
//!
//! All public APIs return [`QLResult<T>`] instead of panicking.

use thiserror::Error;

/// Central error type for the entire ql-rust library.
#[derive(Error, Debug)]
pub enum QLError {
    /// Instrument has no pricing engine set.
    #[error("null pricing engine")]
    NullEngine,

    /// Attempted to dereference an empty (unlinked) handle.
    #[error("empty handle cannot be dereferenced")]
    EmptyHandle,

    /// A requested result field was not computed by the engine.
    #[error("{field} not provided")]
    MissingResult {
        /// The name of the missing result field.
        field: &'static str,
    },

    /// A date falls outside the range covered by a term structure.
    #[error("date {0} is outside curve range")]
    DateOutOfRange(String),

    /// A quantity that must be non-negative was negative.
    #[error("negative {quantity}: {value}")]
    NegativeValue {
        /// What was negative (e.g., "volatility", "rate").
        quantity: &'static str,
        /// The offending value.
        value: f64,
    },

    /// Model calibration did not converge.
    #[error("calibration failed: {0}")]
    CalibrationFailure(String),

    /// Root-finding solver exceeded maximum iterations.
    #[error("root not found after {0} iterations")]
    RootNotFound(usize),

    /// An argument to a function was invalid.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// A numeric input was zero when a non-zero value was required (e.g. time, maturity).
    #[error("zero {quantity} not allowed")]
    ZeroInput {
        /// What was zero (e.g., "time_to_expiry", "maturity").
        quantity: &'static str,
    },

    /// A computed or supplied value is NaN or infinite.
    #[error("non-finite value in {context}: {value}")]
    NonFinite {
        /// Where the non-finite value was detected (e.g., "Black-Scholes d1").
        context: &'static str,
        /// The offending value.
        value: f64,
    },

    /// Object not found in persistence store.
    #[error("object not found")]
    NotFound,

    /// Catch-all for other errors.
    #[error("{0}")]
    Other(String),
}

/// Convenience Result alias used throughout ql-rust.
pub type QLResult<T> = Result<T, QLError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_messages() {
        let e = QLError::NullEngine;
        assert_eq!(e.to_string(), "null pricing engine");

        let e = QLError::EmptyHandle;
        assert_eq!(e.to_string(), "empty handle cannot be dereferenced");

        let e = QLError::MissingResult { field: "NPV" };
        assert_eq!(e.to_string(), "NPV not provided");

        let e = QLError::NegativeValue {
            quantity: "volatility",
            value: -0.05,
        };
        assert_eq!(e.to_string(), "negative volatility: -0.05");

        let e = QLError::RootNotFound(100);
        assert_eq!(e.to_string(), "root not found after 100 iterations");

        let e = QLError::ZeroInput { quantity: "time_to_expiry" };
        assert_eq!(e.to_string(), "zero time_to_expiry not allowed");

        let e = QLError::NonFinite { context: "d1", value: f64::NAN };
        assert!(e.to_string().contains("non-finite value in d1"));
    }

    #[test]
    fn ql_result_ok_and_err() {
        let ok: QLResult<f64> = Ok(42.0);
        assert!(ok.is_ok());

        let err: QLResult<f64> = Err(QLError::NotFound);
        assert!(err.is_err());
    }
}
