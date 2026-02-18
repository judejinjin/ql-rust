//! # ql-cashflows
//!
//! Cash flow traits and implementations: fixed/floating coupons, legs, coupon pricers,
//! CMS coupons, digital/capped-floored coupons, sub-period and range accrual coupons,
//! and extended cash flow analytics.

pub mod cashflow;
pub mod coupon;
pub mod simple_cashflow;
pub mod fixed_rate_coupon;
pub mod ibor_coupon;
pub mod overnight_coupon;
pub mod coupon_pricer;
pub mod leg;
pub mod cashflow_analytics;
pub mod cms_coupon;
pub mod digital_coupon;
pub mod cashflow_analytics_extended;

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
pub use cms_coupon::{CmsCoupon, cms_convexity_adjustment, cms_caplet_price};
pub use digital_coupon::{
    DigitalCoupon, CapFlooredCoupon, RangeAccrualCoupon,
    SubPeriodCoupon, SubPeriodType,
};
pub use cashflow_analytics_extended::{
    convexity, modified_duration, dv01, z_spread, atm_rate,
    time_bucketed_cashflows, TimeBucket,
};
