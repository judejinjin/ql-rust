# SecDB & Beacon Platform — Persistence Architecture for Financial Instruments

> **SecDB** is Goldman Sachs' legendary internal trading and risk management platform, developed from the mid-1990s. **Beacon Platform** is a commercial cloud-native platform built by former Goldman engineers (Mark Higgins, Kirat Singh, et al.) that embodies many of the same architectural principles. This document summarizes how these systems persist financial instruments, trades, and lifecycle events — and what lessons apply to a Rust re-implementation of QuantLib.

---

## 1. Core Philosophy: Everything Is an Object in a Graph

Both SecDB and Beacon reject the traditional relational-database approach to financial data. Instead:

- **Every entity** — a trade, an instrument, a yield curve, a market quote, a risk result — is a **first-class object** in a **directed acyclic graph (DAG)**.
- Objects have **typed properties** (called "terms" in SecDB parlance) that can be:
  - **Stored** (persisted input data), or
  - **Derived** (lazily computed from other objects/properties via the DAG).
- The system **automatically tracks dependencies** between objects. When an input changes (e.g., a market quote), all downstream derived values (e.g., NPV, Greeks, P&L) are invalidated and recomputed on demand.
- This is essentially a **giant, persistent spreadsheet** where cells are object properties and formulas are written in a domain-specific language.

---

## 2. The Object Model

### 2.1 Object Types (Classes)

Objects belong to a **type hierarchy** (similar to classes in OOP). Example taxonomy:

```
Object
├── Instrument
│   ├── Option
│   │   ├── VanillaOption
│   │   ├── BarrierOption
│   │   └── AsianOption
│   ├── Swap
│   │   ├── InterestRateSwap
│   │   ├── CrossCurrencySwap
│   │   └── CreditDefaultSwap
│   ├── Bond
│   │   ├── FixedRateBond
│   │   └── FloatingRateBond
│   └── FxForward
├── Trade
│   ├── links to: Instrument, Counterparty, Book
│   └── has: trade date, settlement date, notional, direction
├── Portfolio (collection of Trades)
├── MarketData
│   ├── Quote
│   ├── YieldCurve
│   ├── VolSurface
│   └── CreditCurve
├── LifecycleEvent
│   ├── Execution
│   ├── Amendment
│   ├── PartialTermination
│   ├── Exercise (option exercise)
│   ├── CashSettlement
│   ├── Novation
│   └── Maturity
├── RiskResult
│   ├── NPV, Greeks, Sensitivities
│   └── VaR, Stress scenarios
└── LegalEntity
    ├── Counterparty
    └── Book / Desk
```

### 2.2 Properties / Terms

Each object type defines a set of **properties** (SecDB calls them "terms"). Properties are either:

| Kind | Description | Example |
|---|---|---|
| **Stored** | Persisted directly on the object. Set by users or upstream systems. | `trade.notional = 10_000_000` |
| **Derived** | Computed from the object's stored terms + other objects via the DAG. Lazily evaluated and cached. | `trade.npv` (computed from instrument, curves, vol surface) |
| **Overridden** | A derived term whose value is manually pinned (like QuantLib's `freeze()`). | Override `trade.npv = 5_000` for testing |

**Key insight:** There is no conceptual distinction between "data" and "calculation." A yield curve's discount factors are just derived properties that depend on input quotes. An instrument's NPV is a derived property that depends on its terms and market data objects. The graph unifies storage and computation.

### 2.3 Object Identity & Versioning

- Every object has a **unique identifier** (OID / object key).
- Objects are **versioned** — every mutation creates a new version with a timestamp.
- The system supports **bitemporal queries**: "What was the value of this trade's NPV as-of market close on date X, as-known-at time T?"
- This is critical for P&L attribution, audit trails, and regulatory reporting.

---

## 3. Instrument Persistence

### 3.1 Instrument as Template vs. Trade as Instance

SecDB/Beacon separate the **instrument definition** (the abstract contract terms) from the **trade** (a specific transaction with a counterparty):

```
Instrument (template)                   Trade (instance)
─────────────────────                   ─────────────────
type: InterestRateSwap                  instrument: → IRS_USD_5Y_template
fixed_rate: 0.035                       counterparty: → JPMorgan
float_index: → USD_SOFR                 book: → Rates_NYC
payment_frequency: Semiannual           direction: Payer
day_count: Act360                       notional: 50_000_000
maturity_tenor: 5Y                      trade_date: 2025-06-15
                                        effective_date: 2025-06-17
                                        status: Active
```

- The **instrument** defines *what* the contract is.
- The **trade** defines *who*, *when*, *how much*, and links to the instrument.
- Multiple trades can reference the same instrument template (or each trade can have its own bespoke instrument definition).

### 3.2 Instrument Properties Are Hierarchical

Instruments are defined by a nested tree of terms, not flat columns:

```
VanillaOption:
  underlying: → AAPL_Equity
  option_type: Call
  strike: 150.0
  expiry: 2026-03-20
  exercise_style: European
  settlement_type: Physical
  payoff:
    type: PlainVanilla
  market_conventions:
    calendar: → NYSE
    day_counter: Act365Fixed
    business_day_convention: ModifiedFollowing
```

This hierarchical representation means:
- Complex instruments (e.g., callable bonds, TARNs, exotic structures) are naturally expressed as deeply nested objects.
- Sub-structures (legs, schedules, payoff definitions) are themselves objects that can be shared or referenced.

### 3.3 Leg-Based Structures

Multi-leg instruments (swaps, structured notes) persist each leg as a sub-object:

```
InterestRateSwap:
  legs:
    - FixedLeg:
        notional: 10_000_000
        rate: 0.035
        frequency: Semiannual
        day_count: Thirty360
        payment_dates: [generated from schedule]
    - FloatingLeg:
        notional: 10_000_000
        index: → USD_SOFR_3M
        spread: 0.001
        frequency: Quarterly
        day_count: Act360
        reset_dates: [generated from schedule]
        fixing_dates: [generated from schedule]
  schedule:
    effective_date: 2025-06-17
    termination_date: 2030-06-17
    calendar: → US_FedFunds
    roll_convention: ModifiedFollowing
    stub: ShortFront
```

---

## 4. Trade Persistence & Lifecycle

### 4.1 Trade Object

A trade is a first-class object with its own stored and derived terms:

```
Trade:
  # ── Stored (persisted) ──
  trade_id: "TRD-2025-0012345"
  instrument: → InterestRateSwap_12345
  counterparty: → CPTY_JPM
  book: → BOOK_RATES_NYC
  trader: → USER_JSMITH
  direction: Payer              # we pay fixed
  notional: 50_000_000
  trade_date: 2025-06-15
  effective_date: 2025-06-17
  status: Active
  lifecycle_events: [→ Event_1, → Event_2, ...]

  # ── Derived (computed via DAG) ──
  npv:               → f(instrument, yield_curve, vol_surface, ...)
  dv01:              → f(npv, yield_curve_bump)
  pv01:              → f(npv, yield_curve_bump)
  delta:             → f(npv, spot)
  gamma:             → f(delta, spot)
  theta:             → f(npv, time_shift)
  counterparty_risk: → f(npv, credit_curve, netting_set)
  margin:            → f(npv, initial_margin_model)
  accounting_pnl:    → f(npv, previous_npv, cashflows_settled)
```

### 4.2 Trade Status State Machine

Trades have a `status` field that follows a lifecycle state machine:

```
                    ┌──────────────┐
                    │   Pending    │ (pre-trade / indicative)
                    └──────┬───────┘
                           │ execute
                           ▼
                    ┌──────────────┐
              ┌─────│    Active    │─────┐
              │     └──────┬───────┘     │
              │            │             │
         amend│       mature/expire   novate/assign
              │            │             │
              ▼            ▼             ▼
        ┌──────────┐ ┌──────────┐ ┌───────────┐
        │ Amended  │ │ Matured  │ │ Novated   │
        │ (→Active)│ │(Terminal)│ │(Terminal) │
        └──────────┘ └──────────┘ └───────────┘
                                        │
         terminate/unwind               │ creates new trade
              │                         ▼
              ▼                  ┌───────────┐
        ┌──────────┐             │ New Trade │
        │Terminated│             │ (Active)  │
        │(Terminal)│             └───────────┘
        └──────────┘
```

### 4.3 Lifecycle Events as Immutable Objects

Every mutation to a trade is recorded as an immutable **LifecycleEvent** object:

```
LifecycleEvent:
  event_id: "EVT-2025-00456"
  trade: → TRD-2025-0012345
  event_type: Amendment
  event_date: 2025-09-10
  effective_date: 2025-09-12
  entered_by: → USER_JSMITH
  entered_at: 2025-09-10T14:23:00Z
  details:
    field_changed: "notional"
    old_value: 50_000_000
    new_value: 40_000_000
    reason: "Partial unwind per client request"
  approval:
    status: Approved
    approved_by: → USER_MJONES
    approved_at: 2025-09-10T14:45:00Z
```

**Event types include:**

| Event | Description |
|---|---|
| `Execution` | Initial trade booking |
| `Amendment` | Change to trade terms (notional, rate, dates) |
| `PartialTermination` | Reduce notional without fully closing |
| `FullTermination` | Close/unwind the trade |
| `Exercise` | Option exercise (full or partial) |
| `Assignment` / `Novation` | Transfer to a different counterparty |
| `CashSettlement` | Settlement of a cash flow |
| `PhysicalSettlement` | Delivery of underlying asset |
| `Reset` / `Fixing` | Rate fixing for a floating leg |
| `CreditEvent` | Trigger for CDS / credit-linked instruments |
| `Maturity` | Natural expiration |
| `Cancellation` | Voided / cancelled trade |
| `Restructure` | Structural change (e.g., fallback from LIBOR to SOFR) |

**Key principle:** The trade object itself is never "edited in place." Instead:
1. A new `LifecycleEvent` is created.
2. The event modifies the trade's stored terms (creating a new version).
3. All derived terms (NPV, risk, P&L) are automatically invalidated and recomputed.
4. The previous version remains accessible for audit.

---

## 5. The Dependency Graph (DAG)

### 5.1 How It Works

The DAG is the heart of SecDB/Beacon. Every derived property declares its dependencies:

```
trade.npv
  ├── depends on: trade.instrument
  │                 ├── instrument.cashflows
  │                 │     ├── instrument.schedule
  │                 │     ├── instrument.fixed_rate
  │                 │     └── instrument.float_index.forecast
  │                 │           └── yield_curve.discount(t)
  │                 │                 └── yield_curve.quotes[]
  │                 │                       └── market_quote.value
  │                 └── instrument.payoff
  ├── depends on: pricing_environment.yield_curve
  ├── depends on: pricing_environment.vol_surface
  └── depends on: settings.evaluation_date
```

When `market_quote.value` changes:
1. The quote marks itself dirty.
2. Dirty propagates up: `yield_curve` → `instrument.cashflows` → `trade.npv`.
3. Nothing is recomputed yet (lazy).
4. When someone reads `trade.npv`, the system walks the DAG, recomputes only what's dirty, and caches results.

### 5.2 Graph-Level Operations

| Operation | Description |
|---|---|
| **Bump & Revalue** | Clone the graph, bump one input, recompute NPV → gives a Greek |
| **Scenario Analysis** | Apply a set of bumps to the graph, recompute all trades |
| **What-If** | Fork the graph, make hypothetical changes, compare results |
| **Historical Replay** | Re-evaluate with past market data (bitemporal) |
| **Explain** | Walk the DAG to show *why* a value changed (P&L attribution) |

---

## 6. Persistence Architecture

### 6.1 SecDB: Custom Object Database

SecDB uses a **proprietary object database** (not a traditional RDBMS):

- Objects are stored as **serialized blobs** with indexed keys.
- The schema is defined in the object type system (Slang classes), not SQL tables.
- Queries are expressed as graph traversals, not SQL joins.
- Versioning is built into the storage layer — every write creates a new version.
- High-performance in-memory caching with disk persistence.

### 6.2 Beacon Platform: Modern Cloud-Native Storage

Beacon modernizes SecDB's ideas for the cloud:

- **Object store** backed by scalable cloud databases (e.g., DynamoDB, Cassandra, or similar).
- Objects are serialized as **structured documents** (conceptually JSON/Protobuf-like).
- **Event sourcing** — the log of lifecycle events *is* the primary source of truth; the current state is a derived projection.
- **Snapshotting** — periodic materialization of current state for query performance.
- **Graph computation** runs in distributed compute nodes, not in the database.
- **API-first** — objects are accessed via typed APIs, not raw database queries.

### 6.3 Storage Schema Patterns

**Pattern 1: Document-per-Object**
```json
{
  "oid": "TRD-2025-0012345",
  "type": "Trade",
  "version": 3,
  "timestamp": "2025-09-10T14:45:00Z",
  "stored_terms": {
    "instrument": "ref:IRS_USD_5Y_12345",
    "counterparty": "ref:CPTY_JPM",
    "book": "ref:BOOK_RATES_NYC",
    "direction": "Payer",
    "notional": 40000000,
    "trade_date": "2025-06-15",
    "status": "Active"
  },
  "derived_term_cache": {
    "npv": { "value": 125340.50, "as_of": "2025-09-10T16:00:00Z" }
  }
}
```

**Pattern 2: Event Log (Event Sourcing)**
```json
[
  { "event_id": "EVT-001", "type": "Execution",  "timestamp": "2025-06-15T10:00:00Z",
    "payload": { "notional": 50000000, "fixed_rate": 0.035, "direction": "Payer" }},
  { "event_id": "EVT-002", "type": "Amendment",  "timestamp": "2025-09-10T14:45:00Z",
    "payload": { "notional": 40000000, "reason": "Partial unwind" }},
  { "event_id": "EVT-003", "type": "CashSettlement", "timestamp": "2025-12-17T09:00:00Z",
    "payload": { "amount": -437500, "currency": "USD" }}
]
```

The current trade state is reconstructed by replaying events. This gives:
- Complete audit trail.
- Ability to reconstruct state at any point in time.
- Natural support for amendments, corrections, and backdated events.

---

## 7. SecDB's "Slang" Language / Beacon's Computation Language

Both platforms use a **domain-specific language** for defining derived terms:

### SecDB: Slang (a Lisp-like language)

```lisp
;; Define NPV as a derived term on Trade
(defterm Trade npv ()
  (let ((inst (instrument self))
        (curves (pricing-env self)))
    (sum (mapcar
          (lambda (cf)
            (* (cashflow-amount cf)
               (discount curves (cashflow-date cf))))
          (cashflows inst)))))
```

### Beacon: Python-Based DSL

Beacon uses Python with a reactive framework:

```python
@derived
def npv(trade):
    inst = trade.instrument
    curve = trade.pricing_env.yield_curve
    return sum(
        cf.amount * curve.discount(cf.date)
        for cf in inst.cashflows()
    )

@derived
def dv01(trade):
    base = trade.npv
    bumped_curve = trade.pricing_env.yield_curve.bump(1e-4)
    with override(trade.pricing_env.yield_curve, bumped_curve):
        bumped_npv = trade.npv  # recomputed with bumped curve
    return bumped_npv - base
```

The `@derived` decorator registers the function in the DAG and automatically tracks dependencies.

---

## 8. Market Data Integration

### 8.1 Market Data as Objects

Market data is not stored in a separate "market data system" — it lives in the same object graph:

```
MarketQuote:
  ticker: "USD_SOFR_3M"
  source: Bloomberg
  value: 0.0435          ← stored, updated by feed
  timestamp: 2025-09-10T16:00:00Z

YieldCurve:
  name: "USD_SOFR"
  reference_date: 2025-09-10
  quotes: [→ deposit_1M, → deposit_3M, → fra_6M, → swap_1Y, → swap_2Y, ...]
  interpolation: LogLinear
  bootstrap_method: Iterative
  discount_factors: [...]   ← derived, recomputed when any quote changes
```

### 8.2 Pricing Environments

A **PricingEnvironment** bundles the market data needed for a valuation:

```
PricingEnvironment:
  name: "EOD_2025-09-10"
  evaluation_date: 2025-09-10
  yield_curves:
    USD_SOFR: → curve_obj_1
    EUR_ESTR: → curve_obj_2
  vol_surfaces:
    SPX: → vol_surface_1
    EURUSD: → vol_surface_2
  credit_curves:
    JPM: → credit_curve_1
  fx_spots:
    EURUSD: → quote_obj_1
  inflation_curves:
    USD_CPI: → inflation_curve_1
```

Trades don't directly reference curves — they reference a `PricingEnvironment`, which can be swapped for scenario analysis.

---

## 9. Risk & P&L as Derived Properties

Risk measures are just more derived terms in the DAG:

```
Trade.greeks:
  delta:  → bump underlying spot ±1%, finite difference
  gamma:  → second-order bump
  vega:   → bump vol surface ±1%
  theta:  → shift evaluation date +1d
  rho:    → bump yield curve ±1bp
  dv01:   → bump yield curve ±1bp (fixed income specific)
  cr01:   → bump credit spread ±1bp

Trade.pnl:
  daily_pnl:      → today.npv - yesterday.npv + settled_cashflows
  explained_pnl:  → delta * ΔS + gamma/2 * ΔS² + theta * Δt + vega * Δσ + ...
  unexplained:    → daily_pnl - explained_pnl
```

Because these are in the graph, changing market data automatically invalidates and recomputes P&L across the entire portfolio.

---

## 10. Key Differences from QuantLib

| Aspect | QuantLib | SecDB / Beacon |
|---|---|---|
| **Scope** | Pricing library only | Full trade lifecycle + risk + P&L + persistence |
| **Persistence** | None (in-memory objects only) | Built-in object database with versioning |
| **Trade concept** | Not modeled | First-class `Trade` object with lifecycle |
| **Lifecycle events** | Not modeled | Immutable event log, event sourcing |
| **Market data** | External (`Handle<Quote>`) | In the same object graph |
| **Computation** | Pull-based lazy eval (`LazyObject`) | Pull-based DAG with graph-wide invalidation |
| **Language** | C++ library | DSL (Slang / Python) + platform |
| **Parallelism** | Manual | Graph-level parallelism across independent subgraphs |
| **Historical state** | Not supported | Bitemporal versioning |
| **Risk** | Computed ad-hoc by user code | Derived properties in the graph |
| **Deployment** | Linked library | Platform-as-a-Service |

---

## 11. Lessons for a Rust QuantLib Re-implementation

### 11.1 Consider Adding a Persistence Layer

QuantLib deliberately avoids persistence, but a Rust re-implementation could optionally include it:

```rust
/// Trait for objects that can be persisted.
pub trait Persistable: Serialize + Deserialize {
    fn object_id(&self) -> &ObjectId;
    fn object_type(&self) -> &str;
    fn version(&self) -> u64;
}

/// Object store abstraction.
pub trait ObjectStore {
    fn get<T: Persistable>(&self, id: &ObjectId) -> Result<T>;
    fn put<T: Persistable>(&self, obj: &T) -> Result<()>;
    fn get_as_of<T: Persistable>(&self, id: &ObjectId, as_of: DateTime) -> Result<T>;
    fn history<T: Persistable>(&self, id: &ObjectId) -> Result<Vec<(DateTime, T)>>;
}
```

### 11.2 Consider Adding Trade & Lifecycle Concepts

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade<I: Instrument> {
    pub trade_id: TradeId,
    pub instrument: I,
    pub counterparty: CounterpartyId,
    pub book: BookId,
    pub direction: Direction,
    pub notional: f64,
    pub trade_date: Date,
    pub status: TradeStatus,
    pub events: Vec<LifecycleEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeStatus { Pending, Active, Matured, Terminated, Novated }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LifecycleEvent {
    Execution { details: ExecutionDetails },
    Amendment { field: String, old_value: Value, new_value: Value },
    PartialTermination { new_notional: f64 },
    Exercise { exercise_date: Date, exercise_type: ExerciseDecision },
    CashSettlement { amount: f64, currency: Currency, settle_date: Date },
    Novation { new_counterparty: CounterpartyId },
    Maturity,
    Cancellation { reason: String },
}
```

### 11.3 Consider Event Sourcing for Trade State

```rust
impl<I: Instrument> Trade<I> {
    /// Reconstruct trade state by replaying events.
    pub fn from_events(events: &[LifecycleEvent]) -> Result<Self> {
        let mut trade = Self::default();
        for event in events {
            trade.apply(event)?;
        }
        Ok(trade)
    }

    fn apply(&mut self, event: &LifecycleEvent) -> Result<()> {
        match event {
            LifecycleEvent::Execution { details } => {
                self.status = TradeStatus::Active;
                self.notional = details.notional;
                // ...
            }
            LifecycleEvent::Amendment { field, new_value, .. } => {
                self.apply_amendment(field, new_value)?;
            }
            LifecycleEvent::PartialTermination { new_notional } => {
                self.notional = *new_notional;
            }
            LifecycleEvent::Maturity => {
                self.status = TradeStatus::Matured;
            }
            // ...
        }
        Ok(())
    }
}
```

### 11.4 Consider a Pricing Environment Abstraction

```rust
/// Bundles all market data needed for valuation.
pub struct PricingEnvironment {
    pub evaluation_date: Date,
    pub yield_curves: HashMap<String, Handle<dyn YieldTermStructure>>,
    pub vol_surfaces: HashMap<String, Handle<dyn BlackVolTermStructure>>,
    pub credit_curves: HashMap<String, Handle<dyn DefaultProbabilityTermStructure>>,
    pub fx_spots: HashMap<CurrencyPair, Handle<dyn Quote>>,
    pub inflation_curves: HashMap<String, Handle<dyn InflationTermStructure>>,
}

impl PricingEnvironment {
    /// Create a bumped clone for scenario/Greek calculation.
    pub fn with_yield_curve_bump(&self, curve_name: &str, bump_bps: f64) -> Self {
        let mut env = self.clone();
        // Replace the named curve with a bumped version
        todo!()
    }
}
```

### 11.5 Consider a DAG-Based Computation Engine

The most ambitious — but highest-value — lesson from SecDB:

```rust
use std::collections::HashMap;
use std::any::Any;

type NodeId = u64;

/// A node in the computation graph.
struct GraphNode {
    id: NodeId,
    value: RefCell<Option<Box<dyn Any>>>,
    dependencies: Vec<NodeId>,
    compute: Box<dyn Fn(&ComputationGraph) -> Box<dyn Any>>,
    dirty: Cell<bool>,
}

/// The computation graph — like SecDB's DAG.
pub struct ComputationGraph {
    nodes: HashMap<NodeId, GraphNode>,
    dependents: HashMap<NodeId, Vec<NodeId>>,  // reverse edges
}

impl ComputationGraph {
    /// Get a value, computing it (and its dependencies) if necessary.
    pub fn get<T: 'static>(&self, id: NodeId) -> &T {
        let node = &self.nodes[&id];
        if node.dirty.get() {
            // Recursively ensure dependencies are fresh
            for dep in &node.dependencies {
                self.get::<Box<dyn Any>>(*dep);
            }
            let value = (node.compute)(self);
            *node.value.borrow_mut() = Some(value);
            node.dirty.set(false);
        }
        node.value.borrow().as_ref().unwrap().downcast_ref::<T>().unwrap()
    }

    /// Mark a node (and all its dependents) as dirty.
    pub fn invalidate(&self, id: NodeId) {
        let node = &self.nodes[&id];
        if !node.dirty.get() {
            node.dirty.set(true);
            if let Some(dependents) = self.dependents.get(&id) {
                for dep in dependents {
                    self.invalidate(*dep);
                }
            }
        }
    }
}
```

This is the core innovation of SecDB: unifying data and computation in a single reactive graph that persists across sessions.

---

## 12. Summary Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    Traditional Quant Stack                       │
│                                                                 │
│  Market Data DB ──► Pricing Library ──► Risk System ──► Trade DB│
│  (separate)        (QuantLib, in-mem)   (separate)    (separate)│
│                    No persistence       No pricing    No calc   │
│                                                                 │
│  Each system has its own data model. Integration is fragile.    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    SecDB / Beacon Approach                       │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │            Unified Object Graph (DAG)                     │  │
│  │                                                           │  │
│  │  Market Data ◄──► Instruments ◄──► Trades ◄──► Risk      │  │
│  │       │                │               │          │       │  │
│  │       └────── Dependency Edges ────────┘          │       │  │
│  │                                                   │       │  │
│  │  Everything is an object. Everything is connected.│       │  │
│  │  Change one input → everything recomputes.        │       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Single data model. Single computation engine. Single store.    │
└─────────────────────────────────────────────────────────────────┘
```
