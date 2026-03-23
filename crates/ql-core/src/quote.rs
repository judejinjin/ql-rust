//! Market quote trait and simple implementation.
//!
//! A [`Quote`] represents a market-observable value (e.g., a spot price,
//! interest rate, or implied volatility). When the value changes, all
//! registered observers are notified.

use std::sync::{Arc, RwLock};

use crate::errors::{QLError, QLResult};
use crate::observable::{Observable, ObservableState};

/// A market-observable value.
pub trait Quote: Observable + Send + Sync {
    /// The current value, or an error if not available.
    fn value(&self) -> QLResult<f64>;

    /// Whether a valid value is currently available.
    fn is_valid(&self) -> bool;
}

/// A simple, mutable market quote.
///
/// Stores a single `f64` value and notifies observers when it changes.
pub struct SimpleQuote {
    value: RwLock<Option<f64>>,
    state: RwLock<ObservableState>,
}

impl SimpleQuote {
    /// Create a new `SimpleQuote` with the given initial value.
    pub fn new(value: f64) -> Self {
        Self {
            value: RwLock::new(Some(value)),
            state: RwLock::new(ObservableState::new()),
        }
    }

    /// Create a `SimpleQuote` with no initial value.
    pub fn empty() -> Self {
        Self {
            value: RwLock::new(None),
            state: RwLock::new(ObservableState::new()),
        }
    }

    /// Set a new value and notify observers.
    ///
    /// Returns the previous value (if any).
    pub fn set_value(&self, new_value: f64) -> Option<f64> {
        let old = {
            let mut v = self.value.write().unwrap_or_else(|p| p.into_inner());
            let old = *v;
            *v = Some(new_value);
            old
        };
        self.notify_observers();
        old
    }

    /// Clear the value (make it invalid) and notify observers.
    pub fn reset(&self) {
        *self.value.write().unwrap_or_else(|p| p.into_inner()) = None;
        self.notify_observers();
    }
}

impl Observable for SimpleQuote {
    fn observable_state(&self) -> &RwLock<ObservableState> {
        &self.state
    }
}

impl Quote for SimpleQuote {
    fn value(&self) -> QLResult<f64> {
        self.value
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .ok_or(QLError::MissingResult { field: "quote value" })
    }

    fn is_valid(&self) -> bool {
        self.value.read().unwrap_or_else(|p| p.into_inner()).is_some()
    }
}

// ===========================================================================
// DerivedQuote — applies a transformation to a source quote
// ===========================================================================

/// A quote that applies a transformation function to a source quote.
///
/// # Example
/// ```
/// # use ql_core::quote::{SimpleQuote, DerivedQuote, Quote};
/// # use std::sync::Arc;
/// let base = Arc::new(SimpleQuote::new(100.0));
/// let derived = DerivedQuote::new(base, |x| x * 1.05);
/// assert!((derived.value().unwrap() - 105.0).abs() < 1e-12);
/// ```
pub struct DerivedQuote<F: Fn(f64) -> f64 + Send + Sync> {
    source: Arc<dyn Quote>,
    transform: F,
    state: RwLock<ObservableState>,
}

impl<F: Fn(f64) -> f64 + Send + Sync> DerivedQuote<F> {
    /// New.
    pub fn new(source: Arc<dyn Quote>, transform: F) -> Self {
        Self {
            source,
            transform,
            state: RwLock::new(ObservableState::new()),
        }
    }
}

impl<F: Fn(f64) -> f64 + Send + Sync> Observable for DerivedQuote<F> {
    fn observable_state(&self) -> &RwLock<ObservableState> {
        &self.state
    }
}

impl<F: Fn(f64) -> f64 + Send + Sync> Quote for DerivedQuote<F> {
    fn value(&self) -> QLResult<f64> {
        self.source.value().map(|v| (self.transform)(v))
    }

    fn is_valid(&self) -> bool {
        self.source.is_valid()
    }
}

// ===========================================================================
// CompositeQuote — combines two quotes with a binary function
// ===========================================================================

/// A quote that combines two source quotes with a binary transformation.
///
/// # Example
/// ```
/// # use ql_core::quote::{SimpleQuote, CompositeQuote, Quote};
/// # use std::sync::Arc;
/// let a = Arc::new(SimpleQuote::new(3.0));
/// let b = Arc::new(SimpleQuote::new(4.0));
/// let spread = CompositeQuote::new(a, b, |x, y| x - y);
/// assert!((spread.value().unwrap() - (-1.0)).abs() < 1e-12);
/// ```
pub struct CompositeQuote<F: Fn(f64, f64) -> f64 + Send + Sync> {
    lhs: Arc<dyn Quote>,
    rhs: Arc<dyn Quote>,
    combine: F,
    state: RwLock<ObservableState>,
}

impl<F: Fn(f64, f64) -> f64 + Send + Sync> CompositeQuote<F> {
    /// New.
    pub fn new(lhs: Arc<dyn Quote>, rhs: Arc<dyn Quote>, combine: F) -> Self {
        Self {
            lhs,
            rhs,
            combine,
            state: RwLock::new(ObservableState::new()),
        }
    }
}

impl<F: Fn(f64, f64) -> f64 + Send + Sync> Observable for CompositeQuote<F> {
    fn observable_state(&self) -> &RwLock<ObservableState> {
        &self.state
    }
}

impl<F: Fn(f64, f64) -> f64 + Send + Sync> Quote for CompositeQuote<F> {
    fn value(&self) -> QLResult<f64> {
        let l = self.lhs.value()?;
        let r = self.rhs.value()?;
        Ok((self.combine)(l, r))
    }

    fn is_valid(&self) -> bool {
        self.lhs.is_valid() && self.rhs.is_valid()
    }
}

// ===========================================================================
// DeltaVolQuote — implied vol quoted in delta space
// ===========================================================================

/// Type of delta quotation.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DeltaType {
    /// Spot delta (Δ_s = N(d1)).
    Spot,
    /// Forward delta (Δ_f = N(d2)).
    Forward,
    /// Delta neutral straddle.
    Atm,
}

/// Implied volatility quoted in delta space.
///
/// Stores (delta, vol, maturity, delta_type).
pub struct DeltaVolQuote {
    delta: f64,
    vol: Arc<dyn Quote>,
    maturity: f64,
    delta_type: DeltaType,
    state: RwLock<ObservableState>,
}

impl DeltaVolQuote {
    /// New.
    pub fn new(delta: f64, vol: Arc<dyn Quote>, maturity: f64, delta_type: DeltaType) -> Self {
        Self {
            delta,
            vol,
            maturity,
            delta_type,
            state: RwLock::new(ObservableState::new()),
        }
    }

    /// Delta.
    pub fn delta(&self) -> f64 { self.delta }
    /// Maturity.
    pub fn maturity(&self) -> f64 { self.maturity }
    /// Delta type.
    pub fn delta_type(&self) -> DeltaType { self.delta_type }
}

impl Observable for DeltaVolQuote {
    fn observable_state(&self) -> &RwLock<ObservableState> { &self.state }
}

impl Quote for DeltaVolQuote {
    fn value(&self) -> QLResult<f64> { self.vol.value() }
    fn is_valid(&self) -> bool { self.vol.is_valid() }
}

// ===========================================================================
// EurodollarFuturesQuote — converts futures price to rate
// ===========================================================================

/// A quote wrapping a Eurodollar futures price (100 - rate*100).
///
/// `value()` returns the *rate* = (100 - price) / 100.
pub struct EurodollarFuturesQuote {
    price: Arc<dyn Quote>,
    state: RwLock<ObservableState>,
}

impl EurodollarFuturesQuote {
    /// New.
    pub fn new(price: Arc<dyn Quote>) -> Self {
        Self { price, state: RwLock::new(ObservableState::new()) }
    }

    /// The raw futures price (100 - rate*100).
    pub fn futures_price(&self) -> QLResult<f64> { self.price.value() }
}

impl Observable for EurodollarFuturesQuote {
    fn observable_state(&self) -> &RwLock<ObservableState> { &self.state }
}

impl Quote for EurodollarFuturesQuote {
    fn value(&self) -> QLResult<f64> {
        self.price.value().map(|p| (100.0 - p) / 100.0)
    }
    fn is_valid(&self) -> bool { self.price.is_valid() }
}

// ===========================================================================
// FuturesConvAdjustmentQuote — futures quote + convexity adjustment
// ===========================================================================

/// A futures-based quote with a convexity adjustment added to the implied rate.
///
/// `value()` = futures_rate + convexity_adjustment.
pub struct FuturesConvAdjustmentQuote {
    futures: Arc<dyn Quote>,
    /// Additive convexity adjustment (e.g. from HullWhite model).
    convexity_adj: Arc<dyn Quote>,
    state: RwLock<ObservableState>,
}

impl FuturesConvAdjustmentQuote {
    /// New.
    pub fn new(futures: Arc<dyn Quote>, convexity_adj: Arc<dyn Quote>) -> Self {
        Self { futures, convexity_adj, state: RwLock::new(ObservableState::new()) }
    }
}

impl Observable for FuturesConvAdjustmentQuote {
    fn observable_state(&self) -> &RwLock<ObservableState> { &self.state }
}

impl Quote for FuturesConvAdjustmentQuote {
    fn value(&self) -> QLResult<f64> {
        let fut_rate = self.futures.value().map(|p| (100.0 - p) / 100.0)?;
        let adj = self.convexity_adj.value()?;
        Ok(fut_rate + adj)
    }
    fn is_valid(&self) -> bool {
        self.futures.is_valid() && self.convexity_adj.is_valid()
    }
}

// ===========================================================================
// ForwardValueQuote — discounted forward value of a quote
// ===========================================================================

/// A quote representing the forward value: `spot * growth_factor / discount_factor`.
///
/// Used for projecting a spot quote to a future date.
pub struct ForwardValueQuote {
    spot: Arc<dyn Quote>,
    /// Growth factor (e.g. e^{q*T}) — often the dividend-adjusted growth.
    growth: Arc<dyn Quote>,
    /// Discount factor to the forward date.
    discount: Arc<dyn Quote>,
    state: RwLock<ObservableState>,
}

impl ForwardValueQuote {
    /// New.
    pub fn new(spot: Arc<dyn Quote>, growth: Arc<dyn Quote>, discount: Arc<dyn Quote>) -> Self {
        Self { spot, growth, discount, state: RwLock::new(ObservableState::new()) }
    }
}

impl Observable for ForwardValueQuote {
    fn observable_state(&self) -> &RwLock<ObservableState> { &self.state }
}

impl Quote for ForwardValueQuote {
    fn value(&self) -> QLResult<f64> {
        let s = self.spot.value()?;
        let g = self.growth.value()?;
        let d = self.discount.value()?;
        if d.abs() < 1e-15 {
            return Err(QLError::InvalidArgument("discount factor is zero".to_string()));
        }
        Ok(s * g / d)
    }
    fn is_valid(&self) -> bool {
        self.spot.is_valid() && self.growth.is_valid() && self.discount.is_valid()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observable::Observer;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    struct CountingObserver(Arc<AtomicU32>);
    impl CountingObserver {
        fn new() -> (Arc<AtomicU32>, Self) {
            let counter = Arc::new(AtomicU32::new(0));
            (counter.clone(), Self(counter))
        }
    }
    impl Observer for CountingObserver {
        fn update(&self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn simple_quote_new() {
        let q = SimpleQuote::new(100.0);
        assert!(q.is_valid());
        assert_eq!(q.value().unwrap(), 100.0);
    }

    #[test]
    fn simple_quote_empty() {
        let q = SimpleQuote::empty();
        assert!(!q.is_valid());
        assert!(q.value().is_err());
    }

    #[test]
    fn simple_quote_set_value() {
        let q = SimpleQuote::new(100.0);
        let old = q.set_value(105.0);
        assert_eq!(old, Some(100.0));
        assert_eq!(q.value().unwrap(), 105.0);
    }

    #[test]
    fn simple_quote_notifies_observers() {
        let q = SimpleQuote::new(100.0);
        let (counter, counting_obs) = CountingObserver::new();
        let obs: Arc<dyn Observer> = Arc::new(counting_obs);
        q.register_observer(&obs);

        q.set_value(101.0);
        q.set_value(102.0);

        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn simple_quote_reset() {
        let q = SimpleQuote::new(100.0);
        q.reset();
        assert!(!q.is_valid());
    }
}
