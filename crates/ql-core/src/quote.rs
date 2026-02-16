//! Market quote trait and simple implementation.
//!
//! A [`Quote`] represents a market-observable value (e.g., a spot price,
//! interest rate, or implied volatility). When the value changes, all
//! registered observers are notified.

use std::sync::RwLock;

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
            let mut v = self.value.write().unwrap();
            let old = *v;
            *v = Some(new_value);
            old
        };
        self.notify_observers();
        old
    }

    /// Clear the value (make it invalid) and notify observers.
    pub fn reset(&self) {
        *self.value.write().unwrap() = None;
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
            .unwrap()
            .ok_or(QLError::MissingResult { field: "quote value" })
    }

    fn is_valid(&self) -> bool {
        self.value.read().unwrap().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observable::Observer;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    struct CountingObserver(AtomicU32);
    impl CountingObserver {
        fn new() -> Self {
            Self(AtomicU32::new(0))
        }
        fn count(&self) -> u32 {
            self.0.load(Ordering::SeqCst)
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
        let obs: Arc<dyn Observer> = Arc::new(CountingObserver::new());
        q.register_observer(&obs);

        q.set_value(101.0);
        q.set_value(102.0);

        let counting = unsafe { &*(Arc::as_ptr(&obs) as *const CountingObserver) };
        assert_eq!(counting.count(), 2);
    }

    #[test]
    fn simple_quote_reset() {
        let q = SimpleQuote::new(100.0);
        q.reset();
        assert!(!q.is_valid());
    }
}
