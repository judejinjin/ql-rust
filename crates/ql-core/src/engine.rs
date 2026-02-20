//! Pricing engine trait and reactive instrument wrapper.
//!
//! This module provides the **reactive dispatch** infrastructure:
//!
//! - [`PricingEngine`]: trait that engines implement to price an instrument
//! - [`LazyInstrument`]: a reactive wrapper that caches results and
//!   auto-invalidates when observed inputs (quotes, curves) change
//!
//! ## Architecture
//!
//! ```text
//! SimpleQuote ──notify──▶ Handle ──notify──▶ LazyInstrument.update()
//!                                                   │
//!                                            cache.invalidate()
//!                                                   │
//!                                            (next .npv() call recomputes)
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use ql_core::engine::{PricingEngine, LazyInstrument};
//! use ql_core::quote::SimpleQuote;
//! use ql_core::handle::{Handle, RelinkableHandle};
//!
//! // 1. Create market observables
//! let spot_quote = Arc::new(SimpleQuote::new(100.0));
//!
//! // 2. Create engine (holds handles to observables)
//! let engine = MyEngine::new(spot_quote.clone());
//!
//! // 3. Wrap instrument + engine in LazyInstrument
//! let lazy = Arc::new(LazyInstrument::new(my_option, Box::new(engine)));
//!
//! // 4. Register as observer on the quote
//! spot_quote.register_observer(&(lazy.clone() as Arc<dyn Observer>));
//!
//! // 5. First access computes the price
//! let result = lazy.npv().unwrap();
//!
//! // 6. Change the quote → cache invalidated → next npv() recomputes
//! spot_quote.set_value(105.0);
//! let new_result = lazy.npv().unwrap(); // recalculated
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::errors::{QLError, QLResult};
use crate::observable::{Observable, Observer, SimpleObservable};

// ═══════════════════════════════════════════════════════════════
// PricingEngine trait
// ═══════════════════════════════════════════════════════════════

/// A pricing engine that computes results for an instrument of type `I`.
///
/// Engines are stateless calculators — they read market data from handles
/// they hold internally, perform the calculation, and return a result.
pub trait PricingEngine<I>: Send + Sync {
    /// The result type produced by this engine.
    type Result: Clone + Send;

    /// Compute the pricing result for the given instrument.
    fn calculate(&self, instrument: &I) -> QLResult<Self::Result>;
}

// ═══════════════════════════════════════════════════════════════
// LazyInstrument — reactive wrapper
// ═══════════════════════════════════════════════════════════════

/// A reactive instrument that combines an instrument with a pricing engine.
///
/// When observed inputs change, the cached result is invalidated. The next
/// call to [`npv()`](Self::npv) triggers a fresh calculation.
///
/// `LazyInstrument` implements [`Observer`] so it can be registered on
/// handles, quotes, and other observables. It also implements [`Observable`]
/// so downstream dependents (e.g., portfolio aggregators) can observe it.
///
/// ## Freeze / Unfreeze
///
/// During calibration loops, call [`freeze()`](Self::freeze) to suppress
/// recalculation, then [`unfreeze()`](Self::unfreeze) when done.
pub struct LazyInstrument<I, R>
where
    I: Send + Sync,
    R: Clone + Send,
{
    instrument: I,
    engine: Mutex<Box<dyn PricingEngine<I, Result = R>>>,
    calculated: AtomicBool,
    frozen: AtomicBool,
    result: Mutex<Option<R>>,
    observable: SimpleObservable,
}

impl<I, R> LazyInstrument<I, R>
where
    I: Send + Sync,
    R: Clone + Send,
{
    /// Create a new reactive instrument.
    pub fn new(instrument: I, engine: Box<dyn PricingEngine<I, Result = R>>) -> Self {
        Self {
            instrument,
            engine: Mutex::new(engine),
            calculated: AtomicBool::new(false),
            frozen: AtomicBool::new(false),
            result: Mutex::new(None),
            observable: SimpleObservable::new(),
        }
    }

    /// Get the cached result, recomputing if necessary.
    pub fn npv(&self) -> QLResult<R> {
        if !self.calculated.load(Ordering::SeqCst) {
            let engine = self.engine.lock().unwrap_or_else(|p| p.into_inner());
            // Double-check after acquiring the lock (another thread may have
            // calculated in the meantime).
            if !self.calculated.load(Ordering::SeqCst) {
                let res = engine.calculate(&self.instrument)?;
                *self.result.lock().unwrap_or_else(|p| p.into_inner()) = Some(res);
                self.calculated.store(true, Ordering::SeqCst);
            }
        }
        self.result
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
            .ok_or(QLError::MissingResult { field: "npv" })
    }

    /// Whether the cached result is valid (not stale).
    pub fn is_calculated(&self) -> bool {
        self.calculated.load(Ordering::SeqCst)
    }

    /// Invalidate the cached result.
    pub fn invalidate(&self) {
        if !self.frozen.load(Ordering::SeqCst) {
            self.calculated.store(false, Ordering::SeqCst);
        }
    }

    /// Freeze — suppress invalidation during calibration loops.
    pub fn freeze(&self) {
        self.frozen.store(true, Ordering::SeqCst);
    }

    /// Unfreeze — re-enable invalidation, and mark dirty.
    pub fn unfreeze(&self) {
        self.frozen.store(false, Ordering::SeqCst);
        self.calculated.store(false, Ordering::SeqCst);
    }

    /// Whether the cache is currently frozen.
    pub fn is_frozen(&self) -> bool {
        self.frozen.load(Ordering::SeqCst)
    }

    /// Access the underlying instrument.
    pub fn instrument(&self) -> &I {
        &self.instrument
    }

    /// Replace the pricing engine (invalidates cached result).
    pub fn set_engine(&self, engine: Box<dyn PricingEngine<I, Result = R>>) {
        *self.engine.lock().unwrap_or_else(|p| p.into_inner()) = engine;
        self.invalidate();
        self.observable.notify_observers();
    }
}

// When an observed input changes, invalidate and propagate.
impl<I, R> Observer for LazyInstrument<I, R>
where
    I: Send + Sync,
    R: Clone + Send,
{
    fn update(&self) {
        self.invalidate();
        self.observable.notify_observers();
    }
}

// LazyInstrument is itself observable by downstream dependents.
impl<I, R> Observable for LazyInstrument<I, R>
where
    I: Send + Sync,
    R: Clone + Send,
{
    fn observable_state(&self) -> &std::sync::RwLock<crate::observable::ObservableState> {
        self.observable.observable_state()
    }
}

// ═══════════════════════════════════════════════════════════════
// ClosureEngine — convenience engine from a closure
// ═══════════════════════════════════════════════════════════════

/// A pricing engine built from a closure.
///
/// Useful for ad-hoc calculations and testing.
pub struct ClosureEngine<I, R, F>
where
    F: Fn(&I) -> QLResult<R> + Send + Sync,
{
    f: F,
    _phantom: std::marker::PhantomData<(I, R)>,
}

impl<I, R, F> ClosureEngine<I, R, F>
where
    F: Fn(&I) -> QLResult<R> + Send + Sync,
{
    /// Create an engine from a closure.
    pub fn new(f: F) -> Self {
        Self {
            f,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<I, R, F> PricingEngine<I> for ClosureEngine<I, R, F>
where
    I: Send + Sync,
    R: Clone + Send + Sync,
    F: Fn(&I) -> QLResult<R> + Send + Sync,
{
    type Result = R;

    fn calculate(&self, instrument: &I) -> QLResult<R> {
        (self.f)(instrument)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observable::Observable;
    use crate::quote::{Quote, SimpleQuote};
    use std::sync::atomic::{AtomicU32, Ordering as AtOrd};
    use std::sync::Arc;

    // ── Helpers ──────────────────────────────────────────────

    #[derive(Debug, Clone)]
    struct SimpleOption {
        strike: f64,
    }

    #[derive(Debug, Clone)]
    struct SimpleResult {
        npv: f64,
    }

    /// A toy engine that reads spot from a SimpleQuote.
    struct SpotEngine {
        spot: Arc<SimpleQuote>,
    }

    impl PricingEngine<SimpleOption> for SpotEngine {
        type Result = SimpleResult;

        fn calculate(&self, opt: &SimpleOption) -> QLResult<SimpleResult> {
            let s = self.spot.value()?;
            // toy: intrinsic value of a call
            Ok(SimpleResult {
                npv: (s - opt.strike).max(0.0),
            })
        }
    }

    // ── Tests ────────────────────────────────────────────────

    #[test]
    fn lazy_instrument_basic() {
        let spot = Arc::new(SimpleQuote::new(105.0));
        let engine = SpotEngine {
            spot: spot.clone(),
        };
        let lazy = LazyInstrument::new(
            SimpleOption { strike: 100.0 },
            Box::new(engine),
        );

        assert!(!lazy.is_calculated());
        let res = lazy.npv().unwrap();
        assert_eq!(res.npv, 5.0);
        assert!(lazy.is_calculated());
    }

    #[test]
    fn lazy_instrument_cache_reuse() {
        let call_count = Arc::new(AtomicU32::new(0));
        let cc = call_count.clone();

        let engine = ClosureEngine::new(move |opt: &SimpleOption| {
            cc.fetch_add(1, AtOrd::SeqCst);
            Ok(SimpleResult { npv: 100.0 - opt.strike })
        });
        let lazy = LazyInstrument::new(
            SimpleOption { strike: 95.0 },
            Box::new(engine),
        );

        let _ = lazy.npv().unwrap();
        let _ = lazy.npv().unwrap();
        let _ = lazy.npv().unwrap();
        // Should only compute once
        assert_eq!(call_count.load(AtOrd::SeqCst), 1);
    }

    #[test]
    fn lazy_instrument_invalidate_recomputes() {
        let call_count = Arc::new(AtomicU32::new(0));
        let cc = call_count.clone();
        let value = Arc::new(std::sync::RwLock::new(10.0));
        let val = value.clone();

        let engine = ClosureEngine::new(move |_opt: &SimpleOption| {
            cc.fetch_add(1, AtOrd::SeqCst);
            let v = *val.read().unwrap();
            Ok(SimpleResult { npv: v })
        });
        let lazy = LazyInstrument::new(
            SimpleOption { strike: 100.0 },
            Box::new(engine),
        );

        let r1 = lazy.npv().unwrap();
        assert_eq!(r1.npv, 10.0);

        // Change the value and invalidate
        *value.write().unwrap() = 20.0;
        lazy.invalidate();
        assert!(!lazy.is_calculated());

        let r2 = lazy.npv().unwrap();
        assert_eq!(r2.npv, 20.0);
        assert_eq!(call_count.load(AtOrd::SeqCst), 2);
    }

    #[test]
    fn lazy_instrument_observer_chain() {
        // SimpleQuote → (notify) → LazyInstrument → (recalc on npv)
        let spot = Arc::new(SimpleQuote::new(110.0));
        let spot2 = spot.clone();

        let engine = ClosureEngine::new(move |opt: &SimpleOption| {
            let s = spot2.value()?;
            Ok(SimpleResult {
                npv: (s - opt.strike).max(0.0),
            })
        });

        let lazy: Arc<LazyInstrument<SimpleOption, SimpleResult>> = Arc::new(
            LazyInstrument::new(SimpleOption { strike: 100.0 }, Box::new(engine)),
        );

        // Register LazyInstrument as observer of the spot quote
        spot.register_observer(&(lazy.clone() as Arc<dyn Observer>));

        // Compute initial price
        let r1 = lazy.npv().unwrap();
        assert_eq!(r1.npv, 10.0);
        assert!(lazy.is_calculated());

        // Change spot → observer fires → cache invalidated
        spot.set_value(120.0);
        assert!(!lazy.is_calculated());

        // Next npv() recomputes
        let r2 = lazy.npv().unwrap();
        assert_eq!(r2.npv, 20.0);
    }

    #[test]
    fn lazy_instrument_freeze_suppresses_invalidation() {
        let spot = Arc::new(SimpleQuote::new(110.0));
        let spot2 = spot.clone();

        let engine = ClosureEngine::new(move |opt: &SimpleOption| {
            let s = spot2.value()?;
            Ok(SimpleResult {
                npv: (s - opt.strike).max(0.0),
            })
        });

        let lazy: Arc<LazyInstrument<SimpleOption, SimpleResult>> = Arc::new(
            LazyInstrument::new(SimpleOption { strike: 100.0 }, Box::new(engine)),
        );
        spot.register_observer(&(lazy.clone() as Arc<dyn Observer>));

        let r1 = lazy.npv().unwrap();
        assert_eq!(r1.npv, 10.0);

        // Freeze → invalidation suppressed
        lazy.freeze();
        spot.set_value(200.0);
        assert!(lazy.is_calculated()); // still valid!

        let r_frozen = lazy.npv().unwrap();
        assert_eq!(r_frozen.npv, 10.0); // still old value

        // Unfreeze → dirty + recompute
        lazy.unfreeze();
        assert!(!lazy.is_calculated());
        let r3 = lazy.npv().unwrap();
        assert_eq!(r3.npv, 100.0);
    }

    #[test]
    fn lazy_instrument_set_engine() {
        let engine1 = ClosureEngine::new(|_: &SimpleOption| {
            Ok(SimpleResult { npv: 42.0 })
        });
        let engine2 = ClosureEngine::new(|_: &SimpleOption| {
            Ok(SimpleResult { npv: 99.0 })
        });
        let lazy = LazyInstrument::new(
            SimpleOption { strike: 100.0 },
            Box::new(engine1),
        );

        assert_eq!(lazy.npv().unwrap().npv, 42.0);
        lazy.set_engine(Box::new(engine2));
        assert_eq!(lazy.npv().unwrap().npv, 99.0);
    }

    #[test]
    fn lazy_instrument_downstream_observer() {
        // LazyInstrument is itself observable — downstream gets notified
        let spot = Arc::new(SimpleQuote::new(100.0));
        let spot2 = spot.clone();

        let engine = ClosureEngine::new(move |_: &SimpleOption| {
            let s = spot2.value()?;
            Ok(SimpleResult { npv: s })
        });

        let lazy: Arc<LazyInstrument<SimpleOption, SimpleResult>> = Arc::new(
            LazyInstrument::new(SimpleOption { strike: 100.0 }, Box::new(engine)),
        );
        spot.register_observer(&(lazy.clone() as Arc<dyn Observer>));

        // Downstream observer
        let downstream_count = Arc::new(AtomicU32::new(0));
        let dc = downstream_count.clone();
        struct DownstreamObs(Arc<AtomicU32>);
        impl Observer for DownstreamObs {
            fn update(&self) {
                self.0.fetch_add(1, AtOrd::SeqCst);
            }
        }
        let downstream: Arc<dyn Observer> = Arc::new(DownstreamObs(dc));
        lazy.register_observer(&downstream);

        // Change spot → LazyInstrument notifies downstream
        spot.set_value(105.0);
        assert_eq!(downstream_count.load(AtOrd::SeqCst), 1);

        spot.set_value(110.0);
        assert_eq!(downstream_count.load(AtOrd::SeqCst), 2);
    }

    #[test]
    fn closure_engine_works() {
        let engine = ClosureEngine::new(|opt: &SimpleOption| {
            Ok(SimpleResult {
                npv: opt.strike * 0.1,
            })
        });
        let result = engine.calculate(&SimpleOption { strike: 100.0 }).unwrap();
        assert_eq!(result.npv, 10.0);
    }
}
