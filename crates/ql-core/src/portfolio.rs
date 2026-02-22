//! Reactive portfolio aggregation.
//!
//! A [`ReactivePortfolio`] holds a collection of [`NpvProvider`]s (instruments
//! wrapped in [`LazyInstrument`]). When any entry's cached result is
//! invalidated, the portfolio's cached total NPV is also invalidated and
//! downstream observers are notified.
//!
//! ## Wiring the graph
//!
//! ```text
//! SimpleQuote ──notify──▶ LazyInstrument.update()
//!                                │
//!                          cache.invalidate()
//!                          observable.notify()
//!                                │
//!                       ReactivePortfolio.update()
//!                                │
//!                          total_npv cache invalid
//!                          observable.notify()
//!                                │
//!                         downstream dashboards
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use ql_core::engine::{ClosureEngine, LazyInstrument};
//! use ql_core::portfolio::{HasNpv, NpvProvider, ReactivePortfolio, wire_entry};
//! use ql_core::quote::SimpleQuote;
//!
//! #[derive(Clone)] struct Params { strike: f64 }
//! #[derive(Clone)] struct Result { npv: f64 }
//! impl HasNpv for Result { fn npv_value(&self) -> f64 { self.npv } }
//!
//! let spot   = Arc::new(SimpleQuote::new(100.0));
//! let spot2  = spot.clone();
//!
//! let engine = ClosureEngine::new(move |p: &Params| {
//!     let s = spot2.value()?;
//!     Ok(Result { npv: (s - p.strike).max(0.0) })
//! });
//!
//! let instr: Arc<LazyInstrument<Params, Result>> = Arc::new(
//!     LazyInstrument::new(Params { strike: 95.0 }, Box::new(engine)),
//! );
//!
//! // Register instr as observer of the quote
//! use ql_core::observable::Observer;
//! spot.register_observer(&(instr.clone() as Arc<dyn Observer>));
//!
//! let book = Arc::new(ReactivePortfolio::new("book-1"));
//! wire_entry(&book, instr as Arc<dyn NpvProvider>);
//!
//! assert_eq!(book.total_npv().unwrap(), 5.0);
//! spot.set_value(110.0);
//! assert!(!book.is_valid());
//! assert_eq!(book.total_npv().unwrap(), 15.0);
//! ```

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::engine::LazyInstrument;
use crate::errors::QLResult;
use crate::observable::{Observable, Observer, SimpleObservable, ObservableState};

// ═══════════════════════════════════════════════════════════════
// HasNpv — extract a scalar NPV from a result type
// ═══════════════════════════════════════════════════════════════

/// Any pricing result that exposes a scalar net present value.
///
/// Implement this for your engine result struct to obtain a blanket
/// [`NpvProvider`] implementation on the corresponding [`LazyInstrument`].
///
/// ```rust
/// use ql_core::portfolio::HasNpv;
///
/// #[derive(Clone)]
/// struct OptionResult { npv: f64, delta: f64 }
///
/// impl HasNpv for OptionResult {
///     fn npv_value(&self) -> f64 { self.npv }
/// }
/// ```
pub trait HasNpv {
    /// The scalar NPV of this result.
    fn npv_value(&self) -> f64;
}

// ═══════════════════════════════════════════════════════════════
// NpvProvider trait
// ═══════════════════════════════════════════════════════════════

/// An instrument that:
/// - Can compute a scalar NPV (possibly from a cache).
/// - Participates in the [`Observer`] / [`Observable`] graph so upstream
///   changes automatically invalidate it and notify downstream dependents.
///
/// Blanket-implemented for [`LazyInstrument<I, R>`] when `R: HasNpv`.
pub trait NpvProvider: Observer + Observable + Send + Sync {
    /// Compute (or return cached) net present value.
    fn npv(&self) -> QLResult<f64>;

    /// Whether the cached NPV is currently valid.
    fn is_calculated(&self) -> bool;

    /// Invalidate the cached NPV.
    fn invalidate(&self);
}

// Blanket impl: any LazyInstrument whose result implements HasNpv is automatically
// an NpvProvider.
impl<I, R> NpvProvider for LazyInstrument<I, R>
where
    I: 'static + Send + Sync,
    R: 'static + Clone + Send + HasNpv,
{
    fn npv(&self) -> QLResult<f64> {
        // Call the inherent method via UFCS to avoid name clash with this trait's npv().
        LazyInstrument::npv(self).map(|r| r.npv_value())
    }

    fn is_calculated(&self) -> bool {
        LazyInstrument::is_calculated(self)
    }

    fn invalidate(&self) {
        LazyInstrument::invalidate(self);
    }
}

// ═══════════════════════════════════════════════════════════════
// ReactivePortfolio
// ═══════════════════════════════════════════════════════════════

/// A reactive portfolio that aggregates [`NpvProvider`] NPVs.
///
/// The portfolio is itself an [`Observer`] (of its entries) and an
/// [`Observable`] (for downstream systems such as risk dashboards). Whenever
/// any entry is invalidated, the portfolio's cached total NPV is invalidated
/// and downstream observers are notified.
pub struct ReactivePortfolio {
    name:       String,
    entries:    Mutex<Vec<Arc<dyn NpvProvider>>>,
    cached_npv: Mutex<Option<f64>>,
    valid:      AtomicBool,
    observable: SimpleObservable,
}

impl ReactivePortfolio {
    /// Create a new, empty portfolio.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name:       name.into(),
            entries:    Mutex::new(Vec::new()),
            cached_npv: Mutex::new(None),
            valid:      AtomicBool::new(false),
            observable: SimpleObservable::new(),
        }
    }

    /// Portfolio name / book identifier.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Add an [`NpvProvider`] entry without wiring the observer link.
    ///
    /// Use [`wire_entry`] to add *and* register the portfolio as an observer
    /// of the entry in one call.
    pub fn add(&self, entry: Arc<dyn NpvProvider>) {
        self.entries.lock().unwrap_or_else(|p| p.into_inner()).push(entry);
        self.valid.store(false, Ordering::SeqCst);
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap_or_else(|p| p.into_inner()).len()
    }

    /// Whether the portfolio has no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Compute (or return cached) total NPV across all entries.
    ///
    /// Returns an error if any entry's calculation fails.
    pub fn total_npv(&self) -> QLResult<f64> {
        if !self.valid.load(Ordering::Acquire) {
            let entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
            let mut total = 0.0_f64;
            for entry in entries.iter() {
                total += entry.npv()?;
            }
            *self.cached_npv.lock().unwrap_or_else(|p| p.into_inner()) = Some(total);
            self.valid.store(true, Ordering::Release);
        }
        self.cached_npv
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .ok_or(crate::errors::QLError::MissingResult { field: "portfolio npv" })
    }

    /// Per-entry NPVs (computes each entry on demand).
    pub fn entry_npvs(&self) -> Vec<QLResult<f64>> {
        let entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        entries.iter().map(|e| e.npv()).collect()
    }

    /// Whether the cached total NPV is currently valid.
    pub fn is_valid(&self) -> bool {
        self.valid.load(Ordering::Acquire)
    }
}

/// Observe changes in the portfolio's entries.
impl Observer for ReactivePortfolio {
    fn update(&self) {
        // Invalidate only if currently valid, to avoid redundant notifications.
        if self.valid.compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
            self.observable.notify_observers();
        }
    }
}

/// Allow downstream systems to observe the portfolio.
impl Observable for ReactivePortfolio {
    fn observable_state(&self) -> &std::sync::RwLock<ObservableState> {
        self.observable.observable_state()
    }
}

// ═══════════════════════════════════════════════════════════════
// wire_entry — convenience wiring function
// ═══════════════════════════════════════════════════════════════

/// Add `entry` to `portfolio` **and** register `portfolio` as an observer
/// of `entry` in one step.
///
/// After calling this function, any invalidation of `entry` will automatically
/// propagate to `portfolio` (and further to any of *its* observers).
///
/// ## Example
///
/// ```rust,ignore
/// wire_entry(&book, instrument.clone() as Arc<dyn NpvProvider>);
/// ```
pub fn wire_entry(portfolio: &Arc<ReactivePortfolio>, entry: Arc<dyn NpvProvider>) {
    // Register portfolio as observer of entry *before* adding so that any
    // immediately-fired notification is handled correctly.
    let observer: Arc<dyn Observer> = portfolio.clone();
    entry.register_observer(&observer);
    portfolio.add(entry);
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ClosureEngine;
    use crate::quote::{Quote, SimpleQuote};
    use std::sync::atomic::{AtomicU32, Ordering as AtOrd};

    // ── helpers ───────────────────────────────────────────────

    #[derive(Clone, Debug)]
    struct Params {
        strike: f64,
    }

    #[derive(Clone, Debug)]
    struct OptionResult {
        npv: f64,
    }

    impl HasNpv for OptionResult {
        fn npv_value(&self) -> f64 {
            self.npv
        }
    }

    fn make_instrument(
        spot: Arc<SimpleQuote>,
        strike: f64,
    ) -> Arc<LazyInstrument<Params, OptionResult>> {
        let s = spot.clone();
        let engine = ClosureEngine::new(move |p: &Params| {
            let sv = s.value()?;
            Ok(OptionResult { npv: (sv - p.strike).max(0.0) })
        });
        Arc::new(LazyInstrument::new(Params { strike }, Box::new(engine)))
    }

    // ── NpvProvider blanket impl ──────────────────────────────

    #[test]
    fn npv_provider_basic() {
        let spot = Arc::new(SimpleQuote::new(110.0));
        let instr = make_instrument(spot, 100.0);
        assert_eq!(NpvProvider::npv(instr.as_ref()).unwrap(), 10.0);
    }

    #[test]
    fn npv_provider_is_calculated_invalidate() {
        let spot = Arc::new(SimpleQuote::new(110.0));
        let instr = make_instrument(spot, 100.0);

        assert!(!NpvProvider::is_calculated(instr.as_ref()));
        let _ = NpvProvider::npv(instr.as_ref());
        assert!(NpvProvider::is_calculated(instr.as_ref()));

        NpvProvider::invalidate(instr.as_ref());
        assert!(!NpvProvider::is_calculated(instr.as_ref()));
    }

    // ── ReactivePortfolio ─────────────────────────────────────

    #[test]
    fn portfolio_total_npv() {
        let spot = Arc::new(SimpleQuote::new(110.0));
        let i1 = make_instrument(spot.clone(), 100.0); // npv = 10
        let i2 = make_instrument(spot.clone(), 105.0); // npv = 5

        let book = Arc::new(ReactivePortfolio::new("test"));
        wire_entry(&book, i1 as Arc<dyn NpvProvider>);
        wire_entry(&book, i2 as Arc<dyn NpvProvider>);

        assert_eq!(book.total_npv().unwrap(), 15.0);
        assert!(book.is_valid());
    }

    #[test]
    fn portfolio_cache_reused() {
        let spot = Arc::new(SimpleQuote::new(110.0));
        let i1 = make_instrument(spot.clone(), 100.0);

        let book = Arc::new(ReactivePortfolio::new("test"));
        wire_entry(&book, i1 as Arc<dyn NpvProvider>);

        let v1 = book.total_npv().unwrap();
        let v2 = book.total_npv().unwrap(); // should not recompute
        assert_eq!(v1, v2);
        assert_eq!(v1, 10.0);
    }

    #[test]
    fn portfolio_invalidates_on_entry_change() {
        let spot = Arc::new(SimpleQuote::new(110.0));
        let instr = make_instrument(spot.clone(), 100.0);

        // Wire: spot → instr (observer), instr → book (observer)
        spot.register_observer(&(instr.clone() as Arc<dyn Observer>));
        let book = Arc::new(ReactivePortfolio::new("test"));
        wire_entry(&book, instr.clone() as Arc<dyn NpvProvider>);

        assert_eq!(book.total_npv().unwrap(), 10.0);
        assert!(book.is_valid());

        // Change spot → instr notified → book notified
        spot.set_value(120.0);

        assert!(!book.is_valid());
        assert_eq!(book.total_npv().unwrap(), 20.0);
    }

    #[test]
    fn portfolio_notifies_downstream() {
        let spot = Arc::new(SimpleQuote::new(110.0));
        let instr = make_instrument(spot.clone(), 100.0);
        spot.register_observer(&(instr.clone() as Arc<dyn Observer>));

        let book = Arc::new(ReactivePortfolio::new("test"));
        wire_entry(&book, instr.clone() as Arc<dyn NpvProvider>);

        // Downstream observer on the portfolio
        let count = Arc::new(AtomicU32::new(0));
        let cnt = count.clone();
        struct DownObs(Arc<AtomicU32>);
        impl Observer for DownObs {
            fn update(&self) { self.0.fetch_add(1, AtOrd::SeqCst); }
        }
        let downstream: Arc<dyn Observer> = Arc::new(DownObs(cnt));
        book.register_observer(&downstream);

        // Prime the cache, then change spot twice
        let _ = book.total_npv();

        spot.set_value(115.0);
        assert_eq!(count.load(AtOrd::SeqCst), 1);

        // Re-prime and change again
        let _ = book.total_npv();
        spot.set_value(120.0);
        assert_eq!(count.load(AtOrd::SeqCst), 2);
    }

    #[test]
    fn portfolio_entry_npvs() {
        let spot = Arc::new(SimpleQuote::new(110.0));
        let i1 = make_instrument(spot.clone(), 100.0);
        let i2 = make_instrument(spot.clone(), 108.0);

        let book = Arc::new(ReactivePortfolio::new("test"));
        wire_entry(&book, i1 as Arc<dyn NpvProvider>);
        wire_entry(&book, i2 as Arc<dyn NpvProvider>);

        let npvs: Vec<f64> = book.entry_npvs().into_iter().map(|r| r.unwrap()).collect();
        assert_eq!(npvs, vec![10.0, 2.0]);
    }

    #[test]
    fn portfolio_len_and_empty() {
        let spot = Arc::new(SimpleQuote::new(100.0));
        let book = Arc::new(ReactivePortfolio::new("test"));
        assert!(book.is_empty());

        let i = make_instrument(spot, 90.0);
        wire_entry(&book, i as Arc<dyn NpvProvider>);
        assert_eq!(book.len(), 1);
        assert!(!book.is_empty());
    }
}
