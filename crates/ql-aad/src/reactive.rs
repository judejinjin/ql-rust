//! Bridge between `ql-aad` portfolio-level AAD and `ql-core`'s
//! [`ReactivePortfolio`](ql_core::portfolio::ReactivePortfolio) with
//! observer/observable graph.
//!
//! When a market quote changes, the reactive system reprices affected
//! instruments. With AAD, it also efficiently recomputes **all Greeks**
//! without separate bumping.
//!
//! # Architecture
//!
//! ```text
//! SimpleQuote (ql-core)
//!       │  notify
//!       ▼
//! AadInstrument (ql-aad::reactive)
//!   ├── caches NPV + PortfolioGreeks
//!   ├── invalidated on update()
//!   └── implements NpvProvider + GreeksProvider
//!       │  notify
//!       ▼
//! AadReactivePortfolio (ql-aad::reactive)
//!   ├── aggregates all entries' Greeks
//!   ├── single-entry or multi-entry modes
//!   └── total_greeks() → aggregate KRDs + rate sensitivities
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use ql_core::quote::SimpleQuote;
//! use ql_aad::reactive::{AadInstrument, AadReactivePortfolio};
//! use ql_aad::cashflows::fixed_rate_bond_cashflows;
//!
//! let times = vec![1.0, 2.0, 5.0, 10.0];
//! let rates = vec![0.03, 0.032, 0.035, 0.04];
//!
//! let bond = Arc::new(AadInstrument::new_bond(
//!     0.04, 100.0, 5, 1.0, &times, &rates,
//! ));
//!
//! let portfolio = Arc::new(AadReactivePortfolio::new("my-book"));
//! portfolio.add_entry(bond);
//!
//! let greeks = portfolio.total_greeks().unwrap();
//! assert!(greeks.total_npv.is_finite());
//! assert!(greeks.key_rate_durations.len() == 4);
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use ql_core::errors::QLResult;
use ql_core::observable::{Observable, ObservableState, Observer, SimpleObservable};
use ql_core::portfolio::NpvProvider;

use crate::cashflows::{fixed_rate_bond_cashflows, npv, Cashflow};
use crate::curves::DiscountCurveAD;
use crate::number::Number;
use crate::portfolio::PortfolioGreeks;
use crate::tape::{adjoint_tl, with_tape, AReal};

// ===========================================================================
// GreeksProvider trait
// ===========================================================================

/// Extension of [`NpvProvider`] that also produces AAD-computed Greeks.
///
/// Implementors cache both NPV and Greeks; a single computation produces
/// both via reverse-mode automatic differentiation.
pub trait GreeksProvider: NpvProvider {
    /// Compute (or return cached) full Greeks for this instrument.
    fn greeks(&self) -> QLResult<PortfolioGreeks>;
}

// ===========================================================================
// AadInstrument — a single reactive AAD-enabled instrument
// ===========================================================================

/// A reactive, AAD-enabled instrument that caches NPV + Greeks.
///
/// When notified of a market change (via [`Observer::update`]), the cached
/// results are invalidated. The next call to [`npv()`](NpvProvider::npv) or
/// [`greeks()`](GreeksProvider::greeks) recomputes everything in a single
/// tape pass.
pub struct AadInstrument {
    cashflows: Vec<Cashflow>,
    weight: f64,
    pillar_times: Vec<f64>,
    pillar_rates: Mutex<Vec<f64>>,
    cached: Mutex<Option<PortfolioGreeks>>,
    valid: AtomicBool,
    observable: SimpleObservable,
}

impl AadInstrument {
    /// Create an AAD instrument from custom cashflows.
    pub fn new(
        cashflows: Vec<Cashflow>,
        weight: f64,
        pillar_times: &[f64],
        pillar_rates: &[f64],
    ) -> Self {
        Self {
            cashflows,
            weight,
            pillar_times: pillar_times.to_vec(),
            pillar_rates: Mutex::new(pillar_rates.to_vec()),
            cached: Mutex::new(None),
            valid: AtomicBool::new(false),
            observable: SimpleObservable::new(),
        }
    }

    /// Create an AAD instrument for a fixed-rate bond.
    pub fn new_bond(
        coupon_rate: f64,
        face: f64,
        maturity_years: usize,
        weight: f64,
        pillar_times: &[f64],
        pillar_rates: &[f64],
    ) -> Self {
        let cfs = fixed_rate_bond_cashflows(coupon_rate, face, maturity_years);
        Self::new(cfs, weight, pillar_times, pillar_rates)
    }

    /// Update the pillar rates (e.g. after a market move).
    ///
    /// This invalidates the cached result and notifies observers.
    pub fn set_rates(&self, new_rates: &[f64]) {
        assert_eq!(new_rates.len(), self.pillar_times.len());
        *self.pillar_rates.lock().unwrap_or_else(|p| p.into_inner()) = new_rates.to_vec();
        self.invalidate();
        self.observable.notify_observers();
    }

    /// Recompute NPV + Greeks via a single tape pass.
    fn compute_inner(&self) -> PortfolioGreeks {
        let rates = self.pillar_rates.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let n = rates.len();
        let weight = self.weight;
        let cashflows = &self.cashflows;
        let pillar_times = &self.pillar_times;

        let (total_npv, instrument_npv, sensitivities) = with_tape(|tape| {
            // Create tape inputs for pillar rates
            let rate_inputs: Vec<AReal> = rates.iter().map(|&r| tape.input(r)).collect();

            // Build discount curve from zero rates
            let curve = DiscountCurveAD::from_zero_rates(&pillar_times, &rate_inputs);

            // Compute NPV of cashflows (Cashflow has f64 fields; npv() lifts via from_f64)
            let instr_npv = npv(cashflows, &curve);
            let weighted = instr_npv * Number::from_f64(weight);

            // Reverse pass
            let adj = adjoint_tl(weighted);

            let sens: Vec<f64> = (0..n).map(|i| adj[i]).collect();
            (weighted.val, instr_npv.val, sens)
        });

        let key_rate_durations: Vec<f64> = sensitivities
            .iter()
            .map(|&s| s * 0.0001)
            .collect();

        PortfolioGreeks {
            total_npv,
            instrument_npvs: vec![instrument_npv],
            key_rate_durations,
            rate_sensitivities: sensitivities,
            pillar_times: pillar_times.to_vec(),
            num_instruments: 1,
        }
    }

    /// Ensure the cache is populated, returning a clone of the result.
    fn ensure_computed(&self) -> PortfolioGreeks {
        if !self.valid.load(Ordering::Acquire) {
            let result = self.compute_inner();
            *self.cached.lock().unwrap_or_else(|p| p.into_inner()) = Some(result);
            self.valid.store(true, Ordering::Release);
        }
        self.cached
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
            .expect("cached result should be populated")
    }
}

impl Observer for AadInstrument {
    fn update(&self) {
        if self
            .valid
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            self.observable.notify_observers();
        }
    }
}

impl Observable for AadInstrument {
    fn observable_state(&self) -> &RwLock<ObservableState> {
        self.observable.observable_state()
    }
}

impl NpvProvider for AadInstrument {
    fn npv(&self) -> QLResult<f64> {
        Ok(self.ensure_computed().total_npv)
    }

    fn is_calculated(&self) -> bool {
        self.valid.load(Ordering::Acquire)
    }

    fn invalidate(&self) {
        self.valid.store(false, Ordering::SeqCst);
        *self.cached.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }
}

impl GreeksProvider for AadInstrument {
    fn greeks(&self) -> QLResult<PortfolioGreeks> {
        Ok(self.ensure_computed())
    }
}

// ===========================================================================
// AadReactivePortfolio — portfolio with aggregated AAD Greeks
// ===========================================================================

/// A reactive portfolio that aggregates AAD Greeks across entries.
///
/// Each entry must implement [`GreeksProvider`]. When any entry is
/// invalidated, the portfolio's cached aggregates are also invalidated
/// and downstream observers are notified.
pub struct AadReactivePortfolio {
    name: String,
    entries: Mutex<Vec<Arc<dyn GreeksProvider>>>,
    cached_greeks: Mutex<Option<PortfolioGreeks>>,
    valid: AtomicBool,
    observable: SimpleObservable,
}

impl AadReactivePortfolio {
    /// Create a new, empty AAD reactive portfolio.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            entries: Mutex::new(Vec::new()),
            cached_greeks: Mutex::new(None),
            valid: AtomicBool::new(false),
            observable: SimpleObservable::new(),
        }
    }

    /// Portfolio name / book identifier.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Add an entry and wire the observer graph.
    pub fn add_entry(self: &Arc<Self>, entry: Arc<dyn GreeksProvider>) {
        let observer: Arc<dyn Observer> = self.clone();
        entry.register_observer(&observer);
        self.entries
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(entry);
        self.valid.store(false, Ordering::SeqCst);
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .len()
    }

    /// Whether the portfolio has no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the cached aggregate is currently valid.
    pub fn is_valid(&self) -> bool {
        self.valid.load(Ordering::Acquire)
    }

    /// Compute (or return cached) total NPV across all entries.
    pub fn total_npv(&self) -> QLResult<f64> {
        Ok(self.total_greeks()?.total_npv)
    }

    /// Compute (or return cached) aggregate Greeks across all entries.
    ///
    /// Key-rate durations and rate sensitivities are summed across entries.
    pub fn total_greeks(&self) -> QLResult<PortfolioGreeks> {
        if !self.valid.load(Ordering::Acquire) {
            let entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
            if entries.is_empty() {
                return Err(ql_core::errors::QLError::MissingResult {
                    field: "no entries in portfolio",
                });
            }

            // Get the first entry's Greeks to initialise dimensions
            let first = entries[0].greeks()?;
            let n_pillars = first.pillar_times.len();

            let mut total_npv = first.total_npv;
            let mut all_instrument_npvs = first.instrument_npvs.clone();
            let mut total_krds = first.key_rate_durations.clone();
            let mut total_sens = first.rate_sensitivities.clone();
            let mut num_instruments = first.num_instruments;

            for entry in entries.iter().skip(1) {
                let g = entry.greeks()?;
                total_npv += g.total_npv;
                all_instrument_npvs.extend(g.instrument_npvs);
                num_instruments += g.num_instruments;
                for i in 0..n_pillars.min(g.key_rate_durations.len()) {
                    total_krds[i] += g.key_rate_durations[i];
                    total_sens[i] += g.rate_sensitivities[i];
                }
            }

            let result = PortfolioGreeks {
                total_npv,
                instrument_npvs: all_instrument_npvs,
                key_rate_durations: total_krds,
                rate_sensitivities: total_sens,
                pillar_times: first.pillar_times,
                num_instruments,
            };

            *self
                .cached_greeks
                .lock()
                .unwrap_or_else(|p| p.into_inner()) = Some(result);
            self.valid.store(true, Ordering::Release);
        }

        self.cached_greeks
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
            .ok_or(ql_core::errors::QLError::MissingResult {
                field: "portfolio greeks",
            })
    }

    /// Per-entry Greeks (computes each entry on demand).
    pub fn entry_greeks(&self) -> Vec<QLResult<PortfolioGreeks>> {
        let entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        entries.iter().map(|e| e.greeks()).collect()
    }
}

impl Observer for AadReactivePortfolio {
    fn update(&self) {
        if self
            .valid
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            self.observable.notify_observers();
        }
    }
}

impl Observable for AadReactivePortfolio {
    fn observable_state(&self) -> &RwLock<ObservableState> {
        self.observable.observable_state()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aad_instrument_basic() {
        let times = vec![1.0, 2.0, 5.0, 10.0];
        let rates = vec![0.03, 0.032, 0.035, 0.04];

        let instr = AadInstrument::new_bond(0.04, 100.0, 5, 1.0, &times, &rates);
        let greeks = instr.greeks().unwrap();

        assert!(greeks.total_npv > 0.0, "NPV should be positive: {}", greeks.total_npv);
        assert_eq!(greeks.key_rate_durations.len(), 4);
        assert_eq!(greeks.num_instruments, 1);
    }

    #[test]
    fn aad_instrument_caching() {
        let times = vec![1.0, 2.0, 5.0, 10.0];
        let rates = vec![0.03, 0.032, 0.035, 0.04];

        let instr = AadInstrument::new_bond(0.04, 100.0, 5, 1.0, &times, &rates);
        assert!(!instr.is_calculated());

        let g1 = instr.greeks().unwrap();
        assert!(instr.is_calculated());

        // Second call should return cached result
        let g2 = instr.greeks().unwrap();
        assert!((g1.total_npv - g2.total_npv).abs() < 1e-15);
    }

    #[test]
    fn aad_instrument_invalidation() {
        let times = vec![1.0, 2.0, 5.0, 10.0];
        let rates = vec![0.03, 0.032, 0.035, 0.04];

        let instr = AadInstrument::new_bond(0.04, 100.0, 5, 1.0, &times, &rates);
        let g1 = instr.greeks().unwrap();
        assert!(instr.is_calculated());

        // Change rates → recompute
        instr.set_rates(&[0.04, 0.042, 0.045, 0.05]);
        assert!(!instr.is_calculated());

        let g2 = instr.greeks().unwrap();
        // Higher rates → lower NPV
        assert!(g2.total_npv < g1.total_npv,
            "NPV after rate increase ({:.2}) should be < ({:.2})", g2.total_npv, g1.total_npv);
    }

    #[test]
    fn aad_instrument_npv_provider() {
        let times = vec![1.0, 2.0, 5.0, 10.0];
        let rates = vec![0.03, 0.032, 0.035, 0.04];

        let instr = AadInstrument::new_bond(0.04, 100.0, 5, 1.0, &times, &rates);
        let npv_val = instr.npv().unwrap();
        let greeks = instr.greeks().unwrap();
        assert!((npv_val - greeks.total_npv).abs() < 1e-15);
    }

    #[test]
    fn reactive_portfolio_single_entry() {
        let times = vec![1.0, 2.0, 5.0, 10.0];
        let rates = vec![0.03, 0.032, 0.035, 0.04];

        let instr = Arc::new(AadInstrument::new_bond(0.04, 100.0, 5, 1.0, &times, &rates));
        let port = Arc::new(AadReactivePortfolio::new("test"));
        port.add_entry(instr);

        let greeks = port.total_greeks().unwrap();
        assert!(greeks.total_npv > 0.0);
        assert_eq!(greeks.key_rate_durations.len(), 4);
        assert_eq!(greeks.num_instruments, 1);
    }

    #[test]
    fn reactive_portfolio_multi_entry() {
        let times = vec![1.0, 2.0, 5.0, 10.0];
        let rates = vec![0.03, 0.032, 0.035, 0.04];

        let bond1 = Arc::new(AadInstrument::new_bond(0.04, 100.0, 5, 1.0, &times, &rates));
        let bond2 = Arc::new(AadInstrument::new_bond(0.05, 100.0, 10, -0.5, &times, &rates));

        let port = Arc::new(AadReactivePortfolio::new("test"));
        port.add_entry(bond1.clone());
        port.add_entry(bond2.clone());

        let greeks = port.total_greeks().unwrap();
        assert_eq!(greeks.num_instruments, 2);

        // Sum of weighted NPVs
        let g1 = bond1.greeks().unwrap();
        let g2 = bond2.greeks().unwrap();
        let expected_npv = g1.total_npv + g2.total_npv;
        assert!((greeks.total_npv - expected_npv).abs() < 1e-10,
            "aggregate NPV={:.4}, expected={:.4}", greeks.total_npv, expected_npv);
    }

    #[test]
    fn reactive_portfolio_invalidation() {
        let times = vec![1.0, 2.0, 5.0, 10.0];
        let rates = vec![0.03, 0.032, 0.035, 0.04];

        let instr = Arc::new(AadInstrument::new_bond(0.04, 100.0, 5, 1.0, &times, &rates));
        let port = Arc::new(AadReactivePortfolio::new("test"));
        port.add_entry(instr.clone());

        let g1 = port.total_greeks().unwrap();
        assert!(port.is_valid());

        // Change rates on the instrument → portfolio invalidated
        instr.set_rates(&[0.04, 0.042, 0.045, 0.05]);
        assert!(!port.is_valid());

        let g2 = port.total_greeks().unwrap();
        assert!(g2.total_npv < g1.total_npv);
    }

    #[test]
    fn reactive_portfolio_krds_aggregate() {
        let times = vec![1.0, 2.0, 5.0, 10.0];
        let rates = vec![0.03, 0.032, 0.035, 0.04];

        let bond1 = Arc::new(AadInstrument::new_bond(0.04, 100.0, 5, 1.0, &times, &rates));
        let bond2 = Arc::new(AadInstrument::new_bond(0.05, 100.0, 10, 1.0, &times, &rates));

        let port = Arc::new(AadReactivePortfolio::new("test"));
        port.add_entry(bond1.clone());
        port.add_entry(bond2.clone());

        let total = port.total_greeks().unwrap();
        let g1 = bond1.greeks().unwrap();
        let g2 = bond2.greeks().unwrap();

        // KRDs should be summed
        for i in 0..4 {
            let expected = g1.key_rate_durations[i] + g2.key_rate_durations[i];
            assert!((total.key_rate_durations[i] - expected).abs() < 1e-10,
                "KRD[{}]: total={:.6}, expected={:.6}", i,
                total.key_rate_durations[i], expected);
        }
    }

    #[test]
    fn reactive_portfolio_sensitivity_signs() {
        let times = vec![1.0, 5.0, 10.0];
        let rates = vec![0.03, 0.035, 0.04];

        let bond = Arc::new(AadInstrument::new_bond(0.04, 100.0, 5, 1.0, &times, &rates));
        let port = Arc::new(AadReactivePortfolio::new("test"));
        port.add_entry(bond);

        let greeks = port.total_greeks().unwrap();
        // Bond price decreases when rates increase → sensitivities should be negative
        for (i, s) in greeks.rate_sensitivities.iter().enumerate() {
            // Some pillars may have zero sensitivity if no cashflows depend on them
            assert!(*s <= 0.001, "rate sensitivity[{}] = {} should be <= 0", i, s);
        }
    }
}
