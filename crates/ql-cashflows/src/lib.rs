//! # ql-cashflows
//!
//! Cash flow traits and implementations: fixed/floating coupons, legs, and coupon pricers.

pub mod cashflow;
pub mod coupon;
pub mod simple_cashflow;
pub mod fixed_rate_coupon;
pub mod ibor_coupon;
pub mod overnight_coupon;
pub mod coupon_pricer;
pub mod leg;
pub mod cashflow_analytics;

// Re-exports
pub use cashflow::{CashFlow, Leg};
pub use coupon::Coupon;
pub use simple_cashflow::SimpleCashFlow;
pub use fixed_rate_coupon::FixedRateCoupon;
pub use ibor_coupon::IborCoupon;
pub use overnight_coupon::OvernightIndexedCoupon;
pub use coupon_pricer::{FloatingRateCouponPricer, BlackIborCouponPricer};
pub use leg::{fixed_leg, ibor_leg, add_notional_exchange};
pub use cashflow_analytics::{npv, bps, accrued_amount, duration};
