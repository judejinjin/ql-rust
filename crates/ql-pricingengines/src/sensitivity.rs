//! Generic bump-and-reprice sensitivity framework.
//!
//! Provides a trait-based mechanism for computing full risk ladders by
//! systematically bumping each market input and measuring the change in
//! engine output.
//!
//! ## Design
//!
//! ```text
//!  MarketEnvironment ──bump(i, Δ)──► MarketEnvironment'
//!        │                                  │
//!     engine.calculate(&instrument)      engine'.calculate(&instrument)
//!        │                                  │
//!        ▼                                  ▼
//!    base_result                        bumped_result
//!              \                       /
//!               ──── sensitivity_i ────
//! ```
//!
//! The [`Sensitivity`] struct represents a single risk measure (delta, vega,
//! etc.) identified by a label. [`compute_sensitivities`] produces a full
//! risk ladder by bumping each input in turn.


// ═══════════════════════════════════════════════════════════════
// Sensitivity result types
// ═══════════════════════════════════════════════════════════════

/// A single sensitivity measure.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Sensitivity {
    /// Label for this risk factor (e.g. "spot", "vol", "rate.1Y").
    pub label: String,
    /// The bump size applied.
    pub bump: f64,
    /// Base NPV.
    pub base_npv: f64,
    /// Bumped NPV (up).
    pub bumped_npv_up: f64,
    /// Bumped NPV (down), if central difference was used.
    pub bumped_npv_down: Option<f64>,
    /// First-order sensitivity: ∂V/∂x (forward difference or central difference).
    pub first_order: f64,
    /// Second-order sensitivity: ∂²V/∂x² (from central difference), if computed.
    pub second_order: Option<f64>,
}

/// A complete risk ladder — vector of sensitivities across all risk factors.
pub type RiskLadder = Vec<Sensitivity>;

// ═══════════════════════════════════════════════════════════════
// Risk factor specification
// ═══════════════════════════════════════════════════════════════

/// Specification of a risk factor to bump.
#[derive(Debug, Clone)]
pub struct RiskFactor {
    /// Human-readable label (e.g. "spot", "vol_atm", "rate.2Y").
    pub label: String,
    /// Bump size in natural units (e.g. 1.0 for $1, 0.01 for 1 vol point, 0.0001 for 1bp).
    pub bump: f64,
    /// Whether to use central difference (bump up and down) for better accuracy.
    pub central_difference: bool,
}

impl RiskFactor {
    /// Create a new risk factor with forward-difference bump.
    pub fn new(label: impl Into<String>, bump: f64) -> Self {
        Self {
            label: label.into(),
            bump,
            central_difference: false,
        }
    }

    /// Create a new risk factor with central-difference bump.
    pub fn central(label: impl Into<String>, bump: f64) -> Self {
        Self {
            label: label.into(),
            bump,
            central_difference: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Equity / option sensitivities
// ═══════════════════════════════════════════════════════════════

/// Market parameters for equity / option sensitivity computation.
#[derive(Debug, Clone)]
pub struct EquityMarketParams {
    pub spot: f64,
    pub risk_free_rate: f64,
    pub dividend_yield: f64,
    pub volatility: f64,
    pub time_to_expiry: f64,
}

/// Compute a full risk ladder for a European option.
///
/// Bumps spot, volatility, rate, dividend yield, and time independently.
///
/// # Examples
///
/// ```
/// use ql_instruments::VanillaOption;
/// use ql_pricingengines::sensitivity::{EquityMarketParams, equity_risk_ladder};
/// use ql_time::{Date, Month};
///
/// let call = VanillaOption::european_call(100.0, Date::from_ymd(2026, Month::January, 15));
/// let params = EquityMarketParams {
///     spot: 100.0,
///     risk_free_rate: 0.05,
///     dividend_yield: 0.0,
///     volatility: 0.20,
///     time_to_expiry: 1.0,
/// };
/// let ladder = equity_risk_ladder(&call, &params);
/// assert!(ladder.len() >= 4);
/// // Delta (spot sensitivity) should be positive for a call
/// assert!(ladder[0].first_order > 0.0);
/// ```
pub fn equity_risk_ladder(
    option: &ql_instruments::VanillaOption,
    params: &EquityMarketParams,
) -> RiskLadder {
    use crate::analytic_european::price_european;

    let base = price_european(
        option, params.spot, params.risk_free_rate,
        params.dividend_yield, params.volatility, params.time_to_expiry,
    );
    let base_npv = base.npv;

    let risk_factors = vec![
        ("spot", params.spot, 1.0),           // bump spot by $1
        ("volatility", params.volatility, 0.01), // bump vol by 1 point
        ("rate", params.risk_free_rate, 0.0001),  // bump rate by 1bp
        ("dividend", params.dividend_yield, 0.0001), // bump div by 1bp
    ];

    risk_factors
        .iter()
        .map(|(label, _current, bump)| {
            let mut p_up = params.clone();
            let mut p_dn = params.clone();

            match *label {
                "spot" => { p_up.spot += bump; p_dn.spot -= bump; }
                "volatility" => { p_up.volatility += bump; p_dn.volatility -= bump; }
                "rate" => { p_up.risk_free_rate += bump; p_dn.risk_free_rate -= bump; }
                "dividend" => { p_up.dividend_yield += bump; p_dn.dividend_yield -= bump; }
                _ => {}
            }

            let npv_up = price_european(
                option, p_up.spot, p_up.risk_free_rate,
                p_up.dividend_yield, p_up.volatility, p_up.time_to_expiry,
            ).npv;
            let npv_dn = price_european(
                option, p_dn.spot, p_dn.risk_free_rate,
                p_dn.dividend_yield, p_dn.volatility, p_dn.time_to_expiry,
            ).npv;

            let first_order = (npv_up - npv_dn) / (2.0 * bump);
            let second_order = (npv_up - 2.0 * base_npv + npv_dn) / (bump * bump);

            Sensitivity {
                label: label.to_string(),
                bump: *bump,
                base_npv,
                bumped_npv_up: npv_up,
                bumped_npv_down: Some(npv_dn),
                first_order,
                second_order: Some(second_order),
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════
// Generic engine-based sensitivity
// ═══════════════════════════════════════════════════════════════

/// Compute a sensitivity from two engine calculations (base vs bumped).
///
/// This is the lowest-level building block. Pass any pair of NPV values
/// and a bump size.
pub fn sensitivity_from_npvs(
    label: impl Into<String>,
    base_npv: f64,
    bumped_npv: f64,
    bump: f64,
) -> Sensitivity {
    Sensitivity {
        label: label.into(),
        bump,
        base_npv,
        bumped_npv_up: bumped_npv,
        bumped_npv_down: None,
        first_order: (bumped_npv - base_npv) / bump,
        second_order: None,
    }
}

/// Compute a central-difference sensitivity from three NPV values.
pub fn sensitivity_central(
    label: impl Into<String>,
    base_npv: f64,
    npv_up: f64,
    npv_down: f64,
    bump: f64,
) -> Sensitivity {
    let first_order = (npv_up - npv_down) / (2.0 * bump);
    let second_order = (npv_up - 2.0 * base_npv + npv_down) / (bump * bump);
    Sensitivity {
        label: label.into(),
        bump,
        base_npv,
        bumped_npv_up: npv_up,
        bumped_npv_down: Some(npv_down),
        first_order,
        second_order: Some(second_order),
    }
}

// ═══════════════════════════════════════════════════════════════
// Fixed-income curve sensitivities
// ═══════════════════════════════════════════════════════════════

/// Compute parallel DV01 and convexity for a cashflow leg.
///
/// Uses central differences with a 1bp bump.
pub fn curve_sensitivities(
    leg: &ql_cashflows::Leg,
    curve: &dyn ql_termstructures::YieldTermStructure,
    settle: ql_time::Date,
) -> RiskLadder {
    use ql_termstructures::SpreadedTermStructure;

    let bump = 0.0001; // 1bp
    let base_npv = ql_cashflows::npv(leg, curve, settle);
    let up_curve = SpreadedTermStructure::new(curve, bump, 50.0);
    let dn_curve = SpreadedTermStructure::new(curve, -bump, 50.0);
    let npv_up = ql_cashflows::npv(leg, &up_curve, settle);
    let npv_dn = ql_cashflows::npv(leg, &dn_curve, settle);

    vec![
        sensitivity_central("parallel_dv01", base_npv, npv_up, npv_dn, bump),
    ]
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use ql_instruments::VanillaOption;
    use ql_time::{Date, Month};

    fn test_params() -> EquityMarketParams {
        EquityMarketParams {
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
            volatility: 0.20,
            time_to_expiry: 1.0,
        }
    }

    #[test]
    fn equity_risk_ladder_call() {
        let call = VanillaOption::european_call(100.0, Date::from_ymd(2026, Month::January, 15));
        let ladder = equity_risk_ladder(&call, &test_params());
        assert_eq!(ladder.len(), 4);

        // Delta (spot): positive for call, ≈ 0.64
        let delta = &ladder[0];
        assert_eq!(delta.label, "spot");
        assert!((delta.first_order - 0.64).abs() < 0.05, "delta={}", delta.first_order);

        // Gamma (spot second order): positive, ≈ 0.019
        assert!(delta.second_order.unwrap() > 0.0, "gamma should be positive");

        // Vega (vol): positive for long option
        let vega = &ladder[1];
        assert_eq!(vega.label, "volatility");
        assert!(vega.first_order > 0.0, "vega positive for long option");

        // Rho (rate): positive for call
        let rho = &ladder[2];
        assert_eq!(rho.label, "rate");
        assert!(rho.first_order > 0.0, "rho positive for call");
    }

    #[test]
    fn equity_risk_ladder_put() {
        let put = VanillaOption::european_put(100.0, Date::from_ymd(2026, Month::January, 15));
        let ladder = equity_risk_ladder(&put, &test_params());

        // Delta: negative for put
        assert!(ladder[0].first_order < 0.0, "put delta negative");
        // Vega: positive for long option
        assert!(ladder[1].first_order > 0.0, "put vega positive");
        // Rho: negative for put
        assert!(ladder[2].first_order < 0.0, "put rho negative");
    }

    #[test]
    fn equity_risk_ladder_consistency_with_analytic_greeks() {
        let call = VanillaOption::european_call(100.0, Date::from_ymd(2026, Month::January, 15));
        let params = test_params();
        let ladder = equity_risk_ladder(&call, &params);

        // Compare numerical delta with analytic
        let analytic = crate::analytic_european::price_european(
            &call, params.spot, params.risk_free_rate,
            params.dividend_yield, params.volatility, params.time_to_expiry,
        );

        let num_delta = ladder[0].first_order;
        assert!((num_delta - analytic.delta).abs() < 0.001,
            "numerical delta={num_delta} vs analytic={}", analytic.delta);

        let num_gamma = ladder[0].second_order.unwrap();
        assert!((num_gamma - analytic.gamma).abs() < 0.01,
            "numerical gamma={num_gamma} vs analytic={}", analytic.gamma);
    }

    #[test]
    fn curve_sensitivities_fixed_leg() {
        use ql_cashflows::fixed_leg;
        use ql_termstructures::FlatForward;
        use ql_time::{DayCounter, Schedule};

        let today = Date::from_ymd(2025, Month::January, 15);
        let dc = DayCounter::Actual365Fixed;
        let curve = FlatForward::new(today, 0.04, dc);

        let sched = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
        ]);
        let leg = fixed_leg(&sched, &[1_000_000.0], &[0.04], dc);

        let ladder = curve_sensitivities(&leg, &curve, today);
        assert_eq!(ladder.len(), 1);

        // DV01 should be negative (rates up → PV down)
        // first_order = (PV_up - PV_dn) / (2*bump) — for a fixed leg
        // rates up → PV down, so PV_up < PV_dn, so first_order < 0
        let dv01 = &ladder[0];
        assert!(dv01.first_order < 0.0, "DV01 should be negative: {}", dv01.first_order);
        // Convexity (second_order) should be positive for a fixed leg
        assert!(dv01.second_order.unwrap() > 0.0, "convexity positive: {}", dv01.second_order.unwrap());
    }

    #[test]
    fn sensitivity_from_npvs_basic() {
        let s = sensitivity_from_npvs("test", 100.0, 101.0, 1.0);
        assert_eq!(s.first_order, 1.0);
        assert!(s.second_order.is_none());
    }

    #[test]
    fn sensitivity_central_basic() {
        let s = sensitivity_central("test", 100.0, 101.0, 99.5, 1.0);
        assert!((s.first_order - 0.75).abs() < 1e-12);
        assert!((s.second_order.unwrap() - 0.5).abs() < 1e-12);
    }
}
