//! Real-time market data feed abstraction.
//!
//! Defines the [`MarketDataFeed`] trait that feed adapters (Bloomberg,
//! Refinitiv, CSV file replay, etc.) implement, plus an [`InMemoryFeed`]
//! mock for testing and simulation, and [`FeedDrivenQuote`] which bridges
//! an incoming feed event into the observer graph via a [`SimpleQuote`].
//!
//! ## Architecture
//!
//! ```text
//! MarketDataFeed ──publish(FeedEvent)──▶ FeedCallback
//!                                              │
//!                                       quote.set_value(v)
//!                                              │
//!                                        Observer chain
//!                                        (LazyInstrument, ReactivePortfolio, …)
//! ```
//!
//! ## Quick start
//!
//! ```rust
//! use std::sync::Arc;
//! use ql_core::market_data::{InMemoryFeed, FeedDrivenQuote, FeedField, FeedEvent,
//!                            MarketDataFeed};
//! use ql_core::quote::Quote;
//!
//! let feed = Arc::new(InMemoryFeed::new("sim"));
//! let fdq  = FeedDrivenQuote::new("AAPL", Arc::clone(&feed) as _, FeedField::Last);
//!
//! feed.publish(FeedEvent::new("AAPL", 175.0));
//! assert_eq!(fdq.quote().value().unwrap(), 175.0);
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::errors::{QLError, QLResult};
use crate::quote::SimpleQuote;

// ═══════════════════════════════════════════════════════════════
// SubscriptionId
// ═══════════════════════════════════════════════════════════════

/// Opaque subscription handle returned by [`MarketDataFeed::subscribe`].
///
/// Pass this to [`MarketDataFeed::unsubscribe`] to cancel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubscriptionId(u64);

// ═══════════════════════════════════════════════════════════════
// FeedField
// ═══════════════════════════════════════════════════════════════

/// Which price field to extract from a [`FeedEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FeedField {
    Bid,
    Ask,
    /// Average of bid and ask; falls back to `Last` if either is absent.
    #[default]
    Mid,
    Last,
}

// ═══════════════════════════════════════════════════════════════
// FeedEvent
// ═══════════════════════════════════════════════════════════════

/// A single market-data tick delivered by a [`MarketDataFeed`].
#[derive(Debug, Clone)]
pub struct FeedEvent {
    /// Instrument identifier (e.g., `"AAPL"`, `"EUR/USD"`)
    pub ticker: String,
    /// Unix epoch time in milliseconds (0 if unavailable).
    pub timestamp_ms: u64,
    /// Best bid price, if available.
    pub bid: Option<f64>,
    /// Best ask price, if available.
    pub ask: Option<f64>,
    /// Last traded price, if available.
    pub last: Option<f64>,
}

impl FeedEvent {
    /// Create a tick with a `last` price and no bid/ask.
    pub fn new(ticker: impl Into<String>, last: f64) -> Self {
        Self {
            ticker: ticker.into(),
            timestamp_ms: Self::now_ms(),
            bid: None,
            ask: None,
            last: Some(last),
        }
    }

    /// Create a tick with a bid/ask spread (last = mid).
    pub fn with_bbo(ticker: impl Into<String>, bid: f64, ask: f64) -> Self {
        Self {
            ticker: ticker.into(),
            timestamp_ms: Self::now_ms(),
            bid: Some(bid),
            ask: Some(ask),
            last: Some((bid + ask) * 0.5),
        }
    }

    /// Create a tick with all fields specified.
    pub fn full(
        ticker: impl Into<String>,
        bid: f64,
        ask: f64,
        last: f64,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            ticker: ticker.into(),
            timestamp_ms,
            bid: Some(bid),
            ask: Some(ask),
            last: Some(last),
        }
    }

    /// Extract the value for a given [`FeedField`].
    pub fn value(&self, field: FeedField) -> QLResult<f64> {
        match field {
            FeedField::Bid  => self.bid .ok_or(QLError::MissingResult { field: "bid"  }),
            FeedField::Ask  => self.ask .ok_or(QLError::MissingResult { field: "ask"  }),
            FeedField::Last => self.last.ok_or(QLError::MissingResult { field: "last" }),
            FeedField::Mid  => {
                match (self.bid, self.ask) {
                    (Some(b), Some(a)) => Ok((b + a) * 0.5),
                    _                  => self.last.ok_or(QLError::MissingResult { field: "mid" }),
                }
            }
        }
    }

    /// Convenience: mid price (same as `value(FeedField::Mid)`).
    pub fn mid(&self) -> QLResult<f64> {
        self.value(FeedField::Mid)
    }

    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

// ═══════════════════════════════════════════════════════════════
// MarketDataFeed trait
// ═══════════════════════════════════════════════════════════════

/// A callback invoked whenever a [`FeedEvent`] arrives for a subscribed ticker.
pub type FeedCallback = Arc<dyn Fn(FeedEvent) + Send + Sync>;

/// A source of real-time or replayed market-data events.
///
/// Feed adapters (Bloomberg, Refinitiv, CSV replay, synthetic) implement this
/// trait. The library ships with [`InMemoryFeed`] for testing.
///
/// ## Contract
///
/// - `subscribe` must be safe to call from multiple threads simultaneously.
/// - `publish` delivers events synchronously to all matching subscribers.
/// - Dead subscriptions (after `unsubscribe`) must never fire.
pub trait MarketDataFeed: Send + Sync {
    /// Human-readable feed name (e.g., `"Bloomberg"`, `"InMemory"`).
    fn name(&self) -> &str;

    /// Subscribe to events for `ticker`.
    ///
    /// Returns a [`SubscriptionId`] that can be passed to [`unsubscribe`].
    fn subscribe(&self, ticker: &str, cb: FeedCallback) -> SubscriptionId;

    /// Cancel a subscription. Subsequent publishes will not invoke the callback.
    fn unsubscribe(&self, id: SubscriptionId);

    /// Synchronously deliver `event` to all subscribers of `event.ticker`.
    fn publish(&self, event: FeedEvent);

    /// All tickers currently having at least one active subscriber.
    fn active_tickers(&self) -> Vec<String>;

    /// Number of active subscriptions across all tickers.
    fn subscription_count(&self) -> usize;
}

// ═══════════════════════════════════════════════════════════════
// InMemoryFeed
// ═══════════════════════════════════════════════════════════════

struct FeedState {
    next_id: u64,
    subs: HashMap<String, Vec<(SubscriptionId, FeedCallback)>>,
}

/// An in-memory [`MarketDataFeed`] for testing and simulation.
///
/// Events are delivered synchronously when [`publish`](Self::publish) is
/// called. Multiple threads may publish and subscribe concurrently.
///
/// ## Example
///
/// ```rust
/// use std::sync::Arc;
/// use ql_core::market_data::{InMemoryFeed, MarketDataFeed, FeedEvent};
/// use std::sync::atomic::{AtomicU32, Ordering};
///
/// let feed = InMemoryFeed::new("test");
/// let count = Arc::new(AtomicU32::new(0));
/// let cnt = count.clone();
///
/// feed.subscribe("SPY", Arc::new(move |_e| { cnt.fetch_add(1, Ordering::SeqCst); }));
/// feed.publish(FeedEvent::new("SPY", 450.0));
/// feed.publish(FeedEvent::new("SPY", 451.0));
///
/// assert_eq!(count.load(Ordering::SeqCst), 2);
/// ```
pub struct InMemoryFeed {
    name: String,
    state: RwLock<FeedState>,
}

impl InMemoryFeed {
    /// Create a new empty feed with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            state: RwLock::new(FeedState {
                next_id: 0,
                subs: HashMap::new(),
            }),
        }
    }
}

impl MarketDataFeed for InMemoryFeed {
    fn name(&self) -> &str {
        &self.name
    }

    fn subscribe(&self, ticker: &str, cb: FeedCallback) -> SubscriptionId {
        let mut state = self.state.write().unwrap_or_else(|p| p.into_inner());
        let id = SubscriptionId(state.next_id);
        state.next_id += 1;
        state.subs.entry(ticker.to_owned()).or_default().push((id, cb));
        id
    }

    fn unsubscribe(&self, id: SubscriptionId) {
        let mut state = self.state.write().unwrap_or_else(|p| p.into_inner());
        for subs in state.subs.values_mut() {
            subs.retain(|(sid, _)| *sid != id);
        }
    }

    fn publish(&self, event: FeedEvent) {
        // Collect callbacks under the read lock, then invoke outside the lock
        // to prevent holding the lock while calling user code.
        let callbacks: Vec<FeedCallback> = {
            let state = self.state.read().unwrap_or_else(|p| p.into_inner());
            state
                .subs
                .get(&event.ticker)
                .map(|v| v.iter().map(|(_, cb)| Arc::clone(cb)).collect())
                .unwrap_or_default()
        };
        for cb in &callbacks {
            cb(event.clone());
        }
    }

    fn active_tickers(&self) -> Vec<String> {
        let state = self.state.read().unwrap_or_else(|p| p.into_inner());
        state
            .subs
            .iter()
            .filter(|(_, v)| !v.is_empty())
            .map(|(k, _)| k.clone())
            .collect()
    }

    fn subscription_count(&self) -> usize {
        let state = self.state.read().unwrap_or_else(|p| p.into_inner());
        state.subs.values().map(|v| v.len()).sum()
    }
}

// ═══════════════════════════════════════════════════════════════
// FeedDrivenQuote
// ═══════════════════════════════════════════════════════════════

/// A [`SimpleQuote`] that auto-updates from a [`MarketDataFeed`].
///
/// When the feed publishes an event for the subscribed ticker, the inner
/// [`SimpleQuote`] is updated (calling `set_value`), which in turn notifies
/// any [`Observer`]s registered on it (instruments, term structures, etc.).
///
/// The feed subscription is cancelled automatically when this object is
/// dropped.
///
/// ## Example
///
/// ```rust
/// use std::sync::Arc;
/// use ql_core::market_data::{InMemoryFeed, FeedDrivenQuote, FeedField, FeedEvent,
///                            MarketDataFeed};
/// use ql_core::quote::Quote;
///
/// let feed = Arc::new(InMemoryFeed::new("sim"));
/// let fdq  = FeedDrivenQuote::new("AAPL", Arc::clone(&feed) as _, FeedField::Last);
///
/// // Before any event, the quote is empty
/// assert!(!fdq.quote().is_valid());
///
/// // Publish an event — the quote is updated
/// feed.publish(FeedEvent::new("AAPL", 175.0));
/// assert_eq!(fdq.quote().value().unwrap(), 175.0);
///
/// // Another event
/// feed.publish(FeedEvent::new("AAPL", 178.5));
/// assert_eq!(fdq.quote().value().unwrap(), 178.5);
/// ```
///
/// [`Observer`]: crate::observable::Observer
pub struct FeedDrivenQuote {
    quote:  Arc<SimpleQuote>,
    sub_id: SubscriptionId,
    feed:   Arc<dyn MarketDataFeed>,
}

impl FeedDrivenQuote {
    /// Subscribe to `ticker` on `feed`, extracting `field` from each event.
    pub fn new(
        ticker: impl AsRef<str>,
        feed:   Arc<dyn MarketDataFeed>,
        field:  FeedField,
    ) -> Self {
        let quote = Arc::new(SimpleQuote::empty());
        let quote_ref = Arc::clone(&quote);
        let cb: FeedCallback = Arc::new(move |event: FeedEvent| {
            if let Ok(v) = event.value(field) {
                quote_ref.set_value(v);
            }
        });
        let sub_id = feed.subscribe(ticker.as_ref(), cb);
        Self { quote, sub_id, feed }
    }

    /// The underlying [`SimpleQuote`].  Register observers on this to wire
    /// the feed into the reactive pricing graph.
    pub fn quote(&self) -> &Arc<SimpleQuote> {
        &self.quote
    }
}

impl Drop for FeedDrivenQuote {
    fn drop(&mut self) {
        self.feed.unsubscribe(self.sub_id);
    }
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observable::Observer;
    use crate::quote::Quote;
    use std::sync::atomic::{AtomicU32, Ordering};

    // ── FeedEvent ─────────────────────────────────────────────

    #[test]
    fn feed_event_last() {
        let e = FeedEvent::new("SPY", 450.0);
        assert_eq!(e.value(FeedField::Last).unwrap(), 450.0);
        assert!(e.value(FeedField::Bid).is_err());
    }

    #[test]
    fn feed_event_bbo() {
        let e = FeedEvent::with_bbo("EUR/USD", 1.0850, 1.0852);
        assert_eq!(e.value(FeedField::Bid).unwrap(), 1.0850);
        assert_eq!(e.value(FeedField::Ask).unwrap(), 1.0852);
        let mid = e.mid().unwrap();
        assert!((mid - 1.0851).abs() < 1e-10);
    }

    #[test]
    fn feed_event_mid_fallback() {
        // No bid/ask, only last — mid falls back to last
        let e = FeedEvent::new("GOOG", 150.0);
        assert_eq!(e.value(FeedField::Mid).unwrap(), 150.0);
    }

    // ── InMemoryFeed ──────────────────────────────────────────

    #[test]
    fn in_memory_feed_subscribe_publish() {
        let feed = InMemoryFeed::new("test");
        let count = Arc::new(AtomicU32::new(0));
        let cnt = count.clone();

        feed.subscribe("AAPL", Arc::new(move |_| { cnt.fetch_add(1, Ordering::SeqCst); }));

        feed.publish(FeedEvent::new("AAPL", 170.0));
        feed.publish(FeedEvent::new("AAPL", 171.0));

        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn in_memory_feed_no_cross_ticker() {
        let feed = InMemoryFeed::new("test");
        let count = Arc::new(AtomicU32::new(0));
        let cnt = count.clone();

        feed.subscribe("AAPL", Arc::new(move |_| { cnt.fetch_add(1, Ordering::SeqCst); }));
        feed.publish(FeedEvent::new("MSFT", 300.0)); // different ticker

        assert_eq!(count.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn in_memory_feed_unsubscribe() {
        let feed = InMemoryFeed::new("test");
        let count = Arc::new(AtomicU32::new(0));
        let cnt = count.clone();

        let id = feed.subscribe("TSLA", Arc::new(move |_| { cnt.fetch_add(1, Ordering::SeqCst); }));
        feed.publish(FeedEvent::new("TSLA", 200.0));
        feed.unsubscribe(id);
        feed.publish(FeedEvent::new("TSLA", 201.0)); // should NOT fire

        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn in_memory_feed_multiple_subscribers() {
        let feed = InMemoryFeed::new("test");
        let c1 = Arc::new(AtomicU32::new(0));
        let c2 = Arc::new(AtomicU32::new(0));
        let cc1 = c1.clone();
        let cc2 = c2.clone();

        feed.subscribe("SPY", Arc::new(move |_| { cc1.fetch_add(1, Ordering::SeqCst); }));
        feed.subscribe("SPY", Arc::new(move |_| { cc2.fetch_add(1, Ordering::SeqCst); }));

        feed.publish(FeedEvent::new("SPY", 450.0));

        assert_eq!(c1.load(Ordering::SeqCst), 1);
        assert_eq!(c2.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn in_memory_feed_active_tickers() {
        let feed = InMemoryFeed::new("test");
        feed.subscribe("AAPL", Arc::new(|_| {}));
        feed.subscribe("MSFT", Arc::new(|_| {}));
        let mut tickers = feed.active_tickers();
        tickers.sort();
        assert_eq!(tickers, vec!["AAPL", "MSFT"]);
    }

    #[test]
    fn in_memory_feed_subscription_count() {
        let feed = InMemoryFeed::new("test");
        assert_eq!(feed.subscription_count(), 0);
        feed.subscribe("A", Arc::new(|_| {}));
        feed.subscribe("A", Arc::new(|_| {}));
        feed.subscribe("B", Arc::new(|_| {}));
        assert_eq!(feed.subscription_count(), 3);
    }

    // ── FeedDrivenQuote ───────────────────────────────────────

    #[test]
    fn feed_driven_quote_empty_initially() {
        let feed = Arc::new(InMemoryFeed::new("sim"));
        let fdq = FeedDrivenQuote::new("AAPL", Arc::clone(&feed) as _, FeedField::Last);
        assert!(!fdq.quote().is_valid());
    }

    #[test]
    fn feed_driven_quote_updates_on_event() {
        let feed = Arc::new(InMemoryFeed::new("sim"));
        let fdq = FeedDrivenQuote::new("AAPL", Arc::clone(&feed) as _, FeedField::Last);

        feed.publish(FeedEvent::new("AAPL", 175.0));
        assert_eq!(fdq.quote().value().unwrap(), 175.0);

        feed.publish(FeedEvent::new("AAPL", 178.5));
        assert_eq!(fdq.quote().value().unwrap(), 178.5);
    }

    #[test]
    fn feed_driven_quote_uses_field() {
        let feed = Arc::new(InMemoryFeed::new("sim"));
        let fdq = FeedDrivenQuote::new("EUR/USD", Arc::clone(&feed) as _, FeedField::Mid);

        feed.publish(FeedEvent::with_bbo("EUR/USD", 1.0850, 1.0852));
        let v = fdq.quote().value().unwrap();
        assert!((v - 1.0851).abs() < 1e-10);
    }

    #[test]
    fn feed_driven_quote_notifies_observers() {
        let feed = Arc::new(InMemoryFeed::new("sim"));
        let fdq = FeedDrivenQuote::new("AAPL", Arc::clone(&feed) as _, FeedField::Last);

        let count = Arc::new(AtomicU32::new(0));
        let cnt = count.clone();
        struct CountObs(Arc<AtomicU32>);
        impl Observer for CountObs {
            fn update(&self) { self.0.fetch_add(1, Ordering::SeqCst); }
        }
        let obs: std::sync::Arc<dyn Observer> = Arc::new(CountObs(cnt));
        use crate::observable::Observable;
        fdq.quote().register_observer(&obs);

        feed.publish(FeedEvent::new("AAPL", 170.0));
        feed.publish(FeedEvent::new("AAPL", 171.0));

        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn feed_driven_quote_unsubscribes_on_drop() {
        let feed = Arc::new(InMemoryFeed::new("sim"));
        assert_eq!(feed.subscription_count(), 0);
        {
            let _fdq = FeedDrivenQuote::new("AAPL", Arc::clone(&feed) as _, FeedField::Last);
            assert_eq!(feed.subscription_count(), 1);
        } // _fdq dropped here
        assert_eq!(feed.subscription_count(), 0);
    }
}
