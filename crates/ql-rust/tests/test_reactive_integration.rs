//! Integration tests for the reactive pricing infrastructure.
//!
//! Tests the full observer chain:
//!   `SimpleQuote → LazyInstrument → ReactivePortfolio → downstream`
//!
//! And the market data feed chain:
//!   `MarketDataFeed → FeedDrivenQuote → observer chain`

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use ql_core::engine::{ClosureEngine, LazyInstrument};
use ql_core::errors::QLResult;
use ql_core::market_data::{FeedDrivenQuote, FeedEvent, FeedField, InMemoryFeed, MarketDataFeed};
use ql_core::observable::{Observable, Observer};
use ql_core::portfolio::{wire_entry, HasNpv, NpvProvider, ReactivePortfolio};
use ql_core::quote::{Quote, SimpleQuote};

// ── Shared helpers ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct Instrument {
    strike: f64,
}

#[derive(Clone, Debug)]
struct Result {
    npv: f64,
}

impl HasNpv for Result {
    fn npv_value(&self) -> f64 {
        self.npv
    }
}

fn make_lazy(
    spot: Arc<SimpleQuote>,
    strike: f64,
) -> Arc<LazyInstrument<Instrument, Result>> {
    let s = Arc::clone(&spot);
    let engine = ClosureEngine::new(move |instr: &Instrument| {
        let sv = s.value()?;
        Ok(Result { npv: (sv - instr.strike).max(0.0) })
    });
    Arc::new(LazyInstrument::new(Instrument { strike }, Box::new(engine)))
}

// Register `instr` as observer of `quote` and return `instr`.
fn register(
    quote: &Arc<SimpleQuote>,
    instr: Arc<LazyInstrument<Instrument, Result>>,
) -> Arc<LazyInstrument<Instrument, Result>> {
    quote.register_observer(&(instr.clone() as Arc<dyn Observer>));
    instr
}

// ── Observer chain tests ──────────────────────────────────────────────────────

#[test]
fn quote_change_invalidates_instrument() {
    let spot  = Arc::new(SimpleQuote::new(110.0));
    let instr = register(&spot, make_lazy(spot.clone(), 100.0));

    assert_eq!(NpvProvider::npv(instr.as_ref()).unwrap(), 10.0);
    assert!(instr.is_calculated());

    spot.set_value(120.0);
    assert!(!instr.is_calculated()); // dirtied by observer

    assert_eq!(NpvProvider::npv(instr.as_ref()).unwrap(), 20.0);
}

#[test]
fn instrument_cache_reuse_until_change() {
    let call_count = Arc::new(AtomicU32::new(0));
    let cc = call_count.clone();
    let spot = Arc::new(SimpleQuote::new(110.0));

    let engine = ClosureEngine::new(move |instr: &Instrument| {
        cc.fetch_add(1, Ordering::SeqCst);
        let sv = 110.0_f64; // fixed for counting purposes
        Ok(Result { npv: (sv - instr.strike).max(0.0) })
    });
    let lazy = Arc::new(LazyInstrument::new(Instrument { strike: 100.0 }, Box::new(engine)));
    spot.register_observer(&(lazy.clone() as Arc<dyn Observer>));

    // Three npv() calls — should only compute once
    let _ = NpvProvider::npv(lazy.as_ref());
    let _ = NpvProvider::npv(lazy.as_ref());
    let _ = NpvProvider::npv(lazy.as_ref());
    assert_eq!(call_count.load(Ordering::SeqCst), 1);

    // Change spot → dirty → recompute
    spot.set_value(115.0);
    let _ = NpvProvider::npv(lazy.as_ref());
    assert_eq!(call_count.load(Ordering::SeqCst), 2);
}

#[test]
fn freeze_prevents_invalidation() {
    let spot  = Arc::new(SimpleQuote::new(110.0));
    let instr = register(&spot, make_lazy(spot.clone(), 100.0));

    let v1 = NpvProvider::npv(instr.as_ref()).unwrap();
    instr.freeze();

    spot.set_value(200.0); // would normally invalidate
    assert!(instr.is_calculated());
    assert_eq!(NpvProvider::npv(instr.as_ref()).unwrap(), v1); // stale cache

    instr.unfreeze();
    assert!(!instr.is_calculated()); // automatically dirty on unfreeze
    let v2 = NpvProvider::npv(instr.as_ref()).unwrap();
    assert_eq!(v2, 100.0); // now reflects spot=200
}

// ── Portfolio tests ───────────────────────────────────────────────────────────

#[test]
fn portfolio_aggregates_multiple_instruments() {
    let spot = Arc::new(SimpleQuote::new(110.0));
    let i1   = register(&spot, make_lazy(spot.clone(), 100.0)); // npv=10
    let i2   = register(&spot, make_lazy(spot.clone(), 105.0)); // npv=5
    let i3   = register(&spot, make_lazy(spot.clone(), 115.0)); // npv=0

    let book = Arc::new(ReactivePortfolio::new("tests"));
    wire_entry(&book, i1 as Arc<dyn NpvProvider>);
    wire_entry(&book, i2 as Arc<dyn NpvProvider>);
    wire_entry(&book, i3 as Arc<dyn NpvProvider>);

    assert_eq!(book.total_npv().unwrap(), 15.0);
}

#[test]
fn portfolio_goes_invalid_when_entry_changes() {
    let spot = Arc::new(SimpleQuote::new(110.0));
    let i1   = register(&spot, make_lazy(spot.clone(), 100.0));

    let book = Arc::new(ReactivePortfolio::new("tests"));
    wire_entry(&book, i1 as Arc<dyn NpvProvider>);

    assert_eq!(book.total_npv().unwrap(), 10.0);
    assert!(book.is_valid());

    spot.set_value(125.0);
    assert!(!book.is_valid());
    assert_eq!(book.total_npv().unwrap(), 25.0);
}

#[test]
fn portfolio_notifies_downstream_observer() {
    let spot  = Arc::new(SimpleQuote::new(110.0));
    let instr = register(&spot, make_lazy(spot.clone(), 100.0));

    let book = Arc::new(ReactivePortfolio::new("tests"));
    wire_entry(&book, instr as Arc<dyn NpvProvider>);
    let _ = book.total_npv(); // prime the cache

    let count: Arc<AtomicU32> = Arc::new(AtomicU32::new(0));
    let cnt = count.clone();
    struct Counter(Arc<AtomicU32>);
    impl Observer for Counter {
        fn update(&self) { self.0.fetch_add(1, Ordering::SeqCst); }
    }
    let _obs: Arc<dyn Observer> = Arc::new(Counter(cnt));
    book.register_observer(&_obs);

    spot.set_value(115.0);
    assert_eq!(count.load(Ordering::SeqCst), 1);

    let _ = book.total_npv(); // re-prime
    spot.set_value(120.0);
    assert_eq!(count.load(Ordering::SeqCst), 2);
}

#[test]
fn portfolio_entry_npvs_correct() {
    let spot = Arc::new(SimpleQuote::new(110.0));
    let i1   = register(&spot, make_lazy(spot.clone(), 100.0));
    let i2   = register(&spot, make_lazy(spot.clone(), 108.0));

    let book = Arc::new(ReactivePortfolio::new("tests"));
    wire_entry(&book, i1 as Arc<dyn NpvProvider>);
    wire_entry(&book, i2 as Arc<dyn NpvProvider>);

    let npvs: Vec<f64> = book
        .entry_npvs()
        .into_iter()
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(npvs, vec![10.0, 2.0]);
    assert_eq!(book.len(), 2);
}

// ── Market data feed tests ────────────────────────────────────────────────────

#[test]
fn feed_event_delivered_to_subscriber() {
    let feed  = InMemoryFeed::new("test");
    let count = Arc::new(AtomicU32::new(0));
    let cnt   = count.clone();

    feed.subscribe("AAPL", Arc::new(move |_| { cnt.fetch_add(1, Ordering::SeqCst); }));

    feed.publish(FeedEvent::new("AAPL", 170.0));
    feed.publish(FeedEvent::new("AAPL", 171.0));

    assert_eq!(count.load(Ordering::SeqCst), 2);
}

#[test]
fn feed_no_cross_ticker_delivery() {
    let feed  = InMemoryFeed::new("test");
    let count = Arc::new(AtomicU32::new(0));
    let cnt   = count.clone();

    feed.subscribe("AAPL", Arc::new(move |_| { cnt.fetch_add(1, Ordering::SeqCst); }));
    feed.publish(FeedEvent::new("MSFT", 300.0));

    assert_eq!(count.load(Ordering::SeqCst), 0);
}

#[test]
fn feed_unsubscribe_stops_delivery() {
    let feed  = InMemoryFeed::new("test");
    let count = Arc::new(AtomicU32::new(0));
    let cnt   = count.clone();

    let sub_id = feed.subscribe("SPY", Arc::new(move |_| { cnt.fetch_add(1, Ordering::SeqCst); }));
    feed.publish(FeedEvent::new("SPY", 450.0));
    feed.unsubscribe(sub_id);
    feed.publish(FeedEvent::new("SPY", 451.0)); // must not fire

    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn feed_driven_quote_updates_and_notifies() {
    let feed = Arc::new(InMemoryFeed::new("sim"));
    let fdq  = FeedDrivenQuote::new("AAPL", Arc::clone(&feed) as _, FeedField::Last);

    // Attach a counting observer to the quote
    let count = Arc::new(AtomicU32::new(0));
    let cnt   = count.clone();
    struct Counter(Arc<AtomicU32>);
    impl Observer for Counter {
        fn update(&self) { self.0.fetch_add(1, Ordering::SeqCst); }
    }
    let _obs: Arc<dyn Observer> = Arc::new(Counter(cnt));
    fdq.quote().register_observer(&_obs);

    feed.publish(FeedEvent::new("AAPL", 175.0));
    assert_eq!(fdq.quote().value().unwrap(), 175.0);
    assert_eq!(count.load(Ordering::SeqCst), 1);

    feed.publish(FeedEvent::new("AAPL", 178.5));
    assert_eq!(fdq.quote().value().unwrap(), 178.5);
    assert_eq!(count.load(Ordering::SeqCst), 2);
}

// ── Full end-to-end chain: feed → quote → instrument → portfolio ──────────────

#[test]
fn feed_drives_instrument_through_portfolio() {
    let feed = Arc::new(InMemoryFeed::new("sim"));
    let fdq  = FeedDrivenQuote::new("SPY", Arc::clone(&feed) as _, FeedField::Last);

    let spot_ref = Arc::clone(fdq.quote());
    let engine = ClosureEngine::new(move |instr: &Instrument| {
        let sv = spot_ref.value()?;
        Ok(Result { npv: (sv - instr.strike).max(0.0) })
    });
    let lazy: Arc<LazyInstrument<Instrument, Result>> = Arc::new(
        LazyInstrument::new(Instrument { strike: 100.0 }, Box::new(engine)),
    );
    // Wire: fdq.quote() → lazy
    fdq.quote().register_observer(&(lazy.clone() as Arc<dyn Observer>));

    let book = Arc::new(ReactivePortfolio::new("spy-calls"));
    wire_entry(&book, lazy as Arc<dyn NpvProvider>);

    // Simulate incoming ticks
    let cases = [(100.0, 0.0), (105.0, 5.0), (110.0, 10.0), (95.0, 0.0)];
    for (price, expected_npv) in cases {
        feed.publish(FeedEvent::new("SPY", price));
        let achieved = book.total_npv().unwrap();
        assert!(
            (achieved - expected_npv).abs() < 1e-12,
            "SPY={price}: expected {expected_npv}, got {achieved}"
        );
    }
}

#[test]
fn feed_driven_quote_auto_unsubscribes_on_drop() {
    let feed = Arc::new(InMemoryFeed::new("sim"));
    assert_eq!(feed.subscription_count(), 0);

    {
        let _fdq = FeedDrivenQuote::new("AAPL", Arc::clone(&feed) as _, FeedField::Last);
        assert_eq!(feed.subscription_count(), 1);
    } // dropped here

    assert_eq!(feed.subscription_count(), 0);
}

#[test]
fn feed_mid_price_used_correctly() {
    let feed = Arc::new(InMemoryFeed::new("sim"));
    let fdq  = FeedDrivenQuote::new("EUR/USD", Arc::clone(&feed) as _, FeedField::Mid);

    feed.publish(FeedEvent::with_bbo("EUR/USD", 1.0850, 1.0852));
    let v = fdq.quote().value().unwrap();
    assert!((v - 1.0851).abs() < 1e-10, "mid mismatch: {v}");
}

// ── Long observer chain ───────────────────────────────────────────────────────

#[test]
fn three_level_observer_chain() {
    // spot → instr → book → aggregator
    let spot  = Arc::new(SimpleQuote::new(110.0));
    let instr = register(&spot, make_lazy(spot.clone(), 100.0));

    let book: Arc<ReactivePortfolio> = Arc::new(ReactivePortfolio::new("level-2"));
    wire_entry(&book, instr as Arc<dyn NpvProvider>);

    // Outer aggregator observes the book
    let outer_invalidated = Arc::new(AtomicU32::new(0));
    let oi = outer_invalidated.clone();
    struct Aggregator(Arc<AtomicU32>);
    impl Observer for Aggregator {
        fn update(&self) { self.0.fetch_add(1, Ordering::SeqCst); }
    }
    let _agg: Arc<dyn Observer> = Arc::new(Aggregator(oi));
    book.register_observer(&_agg);

    // Prime the cache
    let _ = book.total_npv();

    // Push a change — should propagate through all three levels
    spot.set_value(115.0);
    assert_eq!(outer_invalidated.load(Ordering::SeqCst), 1);

    let _ = book.total_npv(); // re-prime
    spot.set_value(120.0);
    assert_eq!(outer_invalidated.load(Ordering::SeqCst), 2);
}
