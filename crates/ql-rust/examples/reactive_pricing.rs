//! Reactive pricing — observer chain and portfolio aggregation.
//!
//! Demonstrates the full **Observer → LazyInstrument → ReactivePortfolio →
//! downstream** wiring pattern, plus a [`MarketDataFeed`] driving live quotes.
//!
//! Run with:
//! ```sh
//! cargo run -p ql-rust --example reactive_pricing
//! ```

use std::sync::Arc;

use ql_core::engine::{ClosureEngine, LazyInstrument};
use ql_core::market_data::{FeedDrivenQuote, FeedEvent, FeedField, InMemoryFeed, MarketDataFeed};
use ql_core::observable::{Observable, Observer};
use ql_core::portfolio::{wire_entry, HasNpv, NpvProvider, ReactivePortfolio};
use ql_core::quote::{Quote, SimpleQuote};
use ql_instruments::VanillaOption;
use ql_pricingengines::price_european;
use ql_time::{Date, Month};

// ─── Result type ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct EuResult { npv: f64, delta: f64, vega: f64 }

impl HasNpv for EuResult {
    fn npv_value(&self) -> f64 { self.npv }
}

// ─── Instrument parameters ────────────────────────────────────────────────────

#[derive(Clone)]
struct EuParams {
    option: VanillaOption,
    expiry: f64,
    vol:    f64,
    rate:   f64,
    div:    f64,
}

// ─── Factory ─────────────────────────────────────────────────────────────────

fn make_call(
    spot_quote: &Arc<SimpleQuote>,
    strike:     f64,
) -> Arc<LazyInstrument<EuParams, EuResult>> {
    let today  = Date::from_ymd(2026, Month::February, 22);
    let option = VanillaOption::european_call(strike, today + 365);
    let params = EuParams { option, expiry: 1.0, vol: 0.20, rate: 0.05, div: 0.02 };

    let sq = Arc::clone(spot_quote);
    let engine = ClosureEngine::new(move |p: &EuParams| {
        let spot = sq.value()?;
        let res  = price_european(&p.option, spot, p.rate, p.div, p.vol, p.expiry);
        Ok(EuResult { npv: res.npv, delta: res.delta, vega: res.vega })
    });
    Arc::new(LazyInstrument::new(params, Box::new(engine)))
}

// helper: trait-qualified to avoid inherent-method ambiguity
fn npv_f64(i: &(impl NpvProvider + ?Sized)) -> f64 {
    NpvProvider::npv(i).unwrap()
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Reactive Pricing — Observer / LazyInstrument / Portfolio");
    println!("═══════════════════════════════════════════════════════════");

    // 1. Market observable
    let spot_quote = Arc::new(SimpleQuote::new(100.0));

    // 2. Three lazy call instruments at different strikes
    let call_95  = make_call(&spot_quote, 95.0);
    let call_100 = make_call(&spot_quote, 100.0);
    let call_105 = make_call(&spot_quote, 105.0);

    // 3. Register each as observer of the spot quote (use Observable::register_observer
    //    explicitly so Rust picks the trait method, not an inherent method)
    Observable::register_observer(&*spot_quote, &(call_95.clone()  as Arc<dyn Observer>));
    Observable::register_observer(&*spot_quote, &(call_100.clone() as Arc<dyn Observer>));
    Observable::register_observer(&*spot_quote, &(call_105.clone() as Arc<dyn Observer>));

    // 4. Reactive portfolio: wire_entry adds entry + wires observer link
    let book = Arc::new(ReactivePortfolio::new("call-strip"));
    wire_entry(&book, call_95.clone()  as Arc<dyn NpvProvider>);
    wire_entry(&book, call_100.clone() as Arc<dyn NpvProvider>);
    wire_entry(&book, call_105.clone() as Arc<dyn NpvProvider>);

    // 5. Initial pricing
    println!("\n[Initial] Spot = {:.2}", spot_quote.value().unwrap());
    println!("  Call(K= 95) NPV = {:.6}", npv_f64(call_95.as_ref()));
    println!("  Call(K=100) NPV = {:.6}", npv_f64(call_100.as_ref()));
    println!("  Call(K=105) NPV = {:.6}", npv_f64(call_105.as_ref()));
    println!("  Portfolio total = {:.6}", book.total_npv().unwrap());
    println!("  Portfolio valid = {}", book.is_valid());

    // 6. Bump spot — entire chain auto-invalidates
    println!("\n[Bump +5] Changing spot 100 → 105");
    spot_quote.set_value(105.0);
    println!("  Call(K=100) calculated? {}", call_100.is_calculated());
    println!("  Portfolio   valid?      {}", book.is_valid());
    println!("  Call(K=100) NPV = {:.6}  (lazy recompute)", npv_f64(call_100.as_ref()));
    println!("  Portfolio total = {:.6}  (lazy recompute)", book.total_npv().unwrap());

    // 7. Freeze a single instrument during a calibration loop
    println!("\n[Freeze] Freezing call_100 to suppress re-pricing during loop");
    let npv_before = npv_f64(call_100.as_ref());
    call_100.freeze();
    spot_quote.set_value(200.0);
    println!("  After spot=200: call_100 calculated? {} (frozen)", call_100.is_calculated());
    println!("  NPV before freeze = {:.6}", npv_before);
    println!("  NPV while frozen  = {:.6}  (stale)", npv_f64(call_100.as_ref()));
    call_100.unfreeze();
    spot_quote.set_value(105.0); // reset
    println!("  After unfreeze: call_100 calculated? {}", call_100.is_calculated());
    println!("  NPV after unfreeze (spot=105) = {:.6}", npv_f64(call_100.as_ref()));

    // 8. Wire a live market-data feed into the chain
    println!("\n[Feed] Wiring InMemoryFeed → FeedDrivenQuote → LazyInstrument");
    let feed      = Arc::new(InMemoryFeed::new("sim"));
    let fdq       = FeedDrivenQuote::new("SPY", Arc::clone(&feed) as _, FeedField::Last);
    let feed_call = make_call(fdq.quote(), 100.0);
    Observable::register_observer(fdq.quote().as_ref(), &(feed_call.clone() as Arc<dyn Observer>));

    println!("  Active tickers: {:?}", feed.active_tickers());
    for price in [98.0_f64, 101.0, 104.0, 107.0, 110.0] {
        feed.publish(FeedEvent::new("SPY", price));
        println!("  SPY tick {:.1} → Call(K=100) NPV = {:.6}", price, npv_f64(feed_call.as_ref()));
    }

    println!("\n[Done] All reactive chains working correctly.");
}

