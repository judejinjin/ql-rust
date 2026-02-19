//! Cross-validation golden tests against QuantLib C++ reference values.
//!
//! These tests compare ql-rust output against known QuantLib C++ results
//! for calendar holidays, yield curve bootstrapping, and Black-Scholes
//! option pricing.

use approx::assert_abs_diff_eq;
use serde_json::Value;

// ---------------------------------------------------------------------------
// Test 1: Calendar holidays and schedule dates (Phase 1)
// ---------------------------------------------------------------------------

mod golden_calendar {
    use super::*;
    use ql_time::{Calendar, Date, DayCounter, Frequency, Month, Schedule};
    use ql_time::calendar::USMarket;
    use ql_time::day_counter::Thirty360Convention;
    use ql_time::schedule::DateGenerationRule;

    fn load_golden() -> Value {
        let data = include_str!("data/golden_calendar.json");
        serde_json::from_str(data).expect("parse golden_calendar.json")
    }

    fn parse_date(s: &str) -> Date {
        let parts: Vec<&str> = s.split('-').collect();
        let y: i32 = parts[0].parse().unwrap();
        let m: u32 = parts[1].parse().unwrap();
        let d: u32 = parts[2].parse().unwrap();
        let month = match m {
            1 => Month::January,
            2 => Month::February,
            3 => Month::March,
            4 => Month::April,
            5 => Month::May,
            6 => Month::June,
            7 => Month::July,
            8 => Month::August,
            9 => Month::September,
            10 => Month::October,
            11 => Month::November,
            12 => Month::December,
            _ => panic!("invalid month"),
        };
        Date::from_ymd(y, month, d)
    }

    #[test]
    fn target_holidays_2025() {
        let golden = load_golden();
        let case = &golden["test_cases"]["target_holidays_2025"];
        let cal = Calendar::Target;

        let expected_holidays: Vec<Date> = case["holidays"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| parse_date(v.as_str().unwrap()))
            .collect();

        // Verify each expected holiday is NOT a business day
        for &hol in &expected_holidays {
            assert!(
                !cal.is_business_day(hol),
                "{hol} should be a TARGET holiday but was reported as business day"
            );
        }

        // Verify business day count for January 2025
        let jan_start = Date::from_ymd(2025, Month::January, 1);
        let jan_end = Date::from_ymd(2025, Month::January, 31);
        let expected_bd_jan = case["business_days_jan"].as_u64().unwrap() as i32;
        let actual_bd_jan = cal.business_days_between(jan_start, jan_end);
        assert_eq!(
            actual_bd_jan, expected_bd_jan,
            "TARGET Jan 2025 business days: expected {expected_bd_jan}, got {actual_bd_jan}"
        );
    }

    #[test]
    fn us_settlement_holidays_2025() {
        let golden = load_golden();
        let case = &golden["test_cases"]["us_settlement_holidays_2025"];
        let cal = Calendar::UnitedStates(USMarket::Settlement);

        let expected_holidays: Vec<Date> = case["holidays"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| parse_date(v.as_str().unwrap()))
            .collect();

        for &hol in &expected_holidays {
            assert!(
                !cal.is_business_day(hol),
                "{hol} should be a US Settlement holiday but was reported as business day"
            );
        }
    }

    #[test]
    fn uk_holidays_2025() {
        let golden = load_golden();
        let case = &golden["test_cases"]["uk_holidays_2025"];
        let cal = Calendar::UnitedKingdom;

        let expected_holidays: Vec<Date> = case["holidays"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| parse_date(v.as_str().unwrap()))
            .collect();

        for &hol in &expected_holidays {
            assert!(
                !cal.is_business_day(hol),
                "{hol} should be a UK holiday but was reported as business day"
            );
        }
    }

    #[test]
    fn day_count_fractions() {
        let golden = load_golden();
        let case = &golden["test_cases"]["day_count_fractions"];

        let d1 = parse_date(case["start_date"].as_str().unwrap());
        let d2 = parse_date(case["end_date"].as_str().unwrap());

        let act360 = DayCounter::Actual360;
        let act365 = DayCounter::Actual365Fixed;
        let thirty360 = DayCounter::Thirty360(Thirty360Convention::BondBasis);

        let expected_act360 = case["fractions"]["Actual360"].as_f64().unwrap();
        let expected_act365 = case["fractions"]["Actual365Fixed"].as_f64().unwrap();
        let expected_30360 = case["fractions"]["Thirty360"].as_f64().unwrap();

        assert_abs_diff_eq!(act360.year_fraction(d1, d2), expected_act360, epsilon = 1e-6);
        assert_abs_diff_eq!(act365.year_fraction(d1, d2), expected_act365, epsilon = 1e-6);
        assert_abs_diff_eq!(thirty360.year_fraction(d1, d2), expected_30360, epsilon = 1e-6);
    }

    #[test]
    fn schedule_semiannual_forward() {
        let golden = load_golden();
        let case = &golden["test_cases"]["schedule_semiannual_forward"];

        let eff = parse_date(case["effective_date"].as_str().unwrap());
        let term = parse_date(case["termination_date"].as_str().unwrap());

        let schedule = Schedule::builder()
            .effective_date(eff)
            .termination_date(term)
            .frequency(Frequency::Semiannual)
            .calendar(Calendar::NullCalendar)
            .convention(ql_time::BusinessDayConvention::Unadjusted)
            .rule(DateGenerationRule::Forward)
            .build()
            .unwrap();

        let expected_dates: Vec<Date> = case["expected_dates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| parse_date(v.as_str().unwrap()))
            .collect();

        assert_eq!(
            schedule.len(),
            expected_dates.len(),
            "schedule length mismatch"
        );
        for (i, &expected) in expected_dates.iter().enumerate() {
            assert_eq!(
                schedule.date(i),
                expected,
                "schedule date {i} mismatch: expected {expected}, got {}",
                schedule.date(i)
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2: Yield curve discount factors (Phase 3)
// ---------------------------------------------------------------------------

mod golden_yield_curve {
    use super::*;
    use ql_termstructures::{FlatForward, YieldTermStructure, DiscountCurve, ZeroCurve};
    use ql_time::{Date, DayCounter, Month};

    fn load_golden() -> Value {
        let data = include_str!("data/golden_yield_curve.json");
        serde_json::from_str(data).expect("parse golden_yield_curve.json")
    }

    fn parse_date(s: &str) -> Date {
        let parts: Vec<&str> = s.split('-').collect();
        let y: i32 = parts[0].parse().unwrap();
        let m: u32 = parts[1].parse().unwrap();
        let d: u32 = parts[2].parse().unwrap();
        let month = match m {
            1 => Month::January,
            2 => Month::February,
            3 => Month::March,
            4 => Month::April,
            5 => Month::May,
            6 => Month::June,
            7 => Month::July,
            8 => Month::August,
            9 => Month::September,
            10 => Month::October,
            11 => Month::November,
            12 => Month::December,
            _ => panic!("invalid month"),
        };
        Date::from_ymd(y, month, d)
    }

    #[test]
    fn flat_curve_discount_factors() {
        let golden = load_golden();
        let case = &golden["test_cases"]["flat_curve_discount_factors"];

        let ref_date = parse_date(case["reference_date"].as_str().unwrap());
        let rate = case["zero_rate_continuous"].as_f64().unwrap();
        let dc = DayCounter::Actual365Fixed;

        let curve = FlatForward::new(ref_date, rate, dc);

        let tenors: Vec<f64> = case["tenors_years"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let expected_dfs: Vec<f64> = case["expected_discount_factors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        for (i, &t) in tenors.iter().enumerate() {
            let df = curve.discount_t(t);
            assert_abs_diff_eq!(
                df,
                expected_dfs[i],
                epsilon = 1e-6
            );
        }

        // Forward rates on a flat curve should all equal the zero rate
        let expected_fwds: Vec<f64> = case["expected_forward_rates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        for (i, &t) in tenors.iter().enumerate() {
            let fwd = curve.forward_rate_t(t);
            assert_abs_diff_eq!(fwd, expected_fwds[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn zero_curve_interpolation() {
        let golden = load_golden();
        let case = &golden["test_cases"]["zero_curve_interpolation"];

        let _ref_date = parse_date(case["reference_date"].as_str().unwrap());
        let dc = DayCounter::Actual365Fixed;
        let tol = case["tolerance"].as_f64().unwrap();

        let node_dates: Vec<Date> = case["nodes"]["dates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| parse_date(v.as_str().unwrap()))
            .collect();
        let node_rates: Vec<f64> = case["nodes"]["zero_rates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let curve = ZeroCurve::new(node_dates, node_rates, dc).unwrap();

        let query_tenors: Vec<f64> = case["query_tenors_years"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let expected_rates: Vec<f64> = case["expected_zero_rates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        for (i, &t) in query_tenors.iter().enumerate() {
            // Compute zero rate from discount factor: r = -ln(df) / t
            let df = curve.discount_t(t);
            let implied_rate = if t > 0.0 { -df.ln() / t } else { 0.0 };
            assert_abs_diff_eq!(
                implied_rate,
                expected_rates[i],
                epsilon = tol
            );
        }
    }

    #[test]
    fn discount_curve_from_nodes() {
        // Build a DiscountCurve from known nodes and check interpolation
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let dc = DayCounter::Actual365Fixed;

        let dates = vec![
            ref_date,
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            Date::from_ymd(2027, Month::January, 15),
        ];
        // At 5% continuous: df(t) = exp(-0.05*t)
        let dfs: Vec<f64> = dates
            .iter()
            .map(|&d| {
                let t = dc.year_fraction(ref_date, d);
                (-0.05 * t).exp()
            })
            .collect();

        let curve = DiscountCurve::new(dates.clone(), dfs.clone(), dc).unwrap();

        // At each node, discount should match exactly
        for (i, &d) in dates.iter().enumerate() {
            let df = curve.discount(d);
            assert_abs_diff_eq!(df, dfs[i], epsilon = 1e-10);
        }
    }
}

// ---------------------------------------------------------------------------
// Test 3: Black-Scholes option pricing (Phase 5)
// ---------------------------------------------------------------------------

mod golden_black_scholes {
    use super::*;
    use ql_instruments::VanillaOption;
    use ql_pricingengines::price_european;
    use ql_time::{Date, Month};

    fn load_golden() -> Value {
        let data = include_str!("data/golden_black_scholes.json");
        serde_json::from_str(data).expect("parse golden_black_scholes.json")
    }

    #[test]
    fn european_call_atm() {
        let golden = load_golden();
        let case = &golden["test_cases"]["european_call_atm"];
        let tol = case["tolerance"].as_f64().unwrap();

        let expiry = Date::from_ymd(2026, Month::January, 15);
        let opt = VanillaOption::european_call(
            case["strike"].as_f64().unwrap(),
            expiry,
        );

        let result = price_european(
            &opt,
            case["spot"].as_f64().unwrap(),
            case["risk_free_rate"].as_f64().unwrap(),
            case["dividend_yield"].as_f64().unwrap(),
            case["volatility"].as_f64().unwrap(),
            case["time_to_expiry_years"].as_f64().unwrap(),
        );

        assert_abs_diff_eq!(result.npv, case["expected_price"].as_f64().unwrap(), epsilon = tol);
        assert_abs_diff_eq!(result.delta, case["expected_delta"].as_f64().unwrap(), epsilon = tol);
        assert_abs_diff_eq!(result.gamma, case["expected_gamma"].as_f64().unwrap(), epsilon = tol * 0.1);
    }

    #[test]
    fn european_put_otm() {
        let golden = load_golden();
        let case = &golden["test_cases"]["european_put_otm"];
        let tol = case["tolerance"].as_f64().unwrap();

        let expiry = Date::from_ymd(2026, Month::January, 15);
        let opt = VanillaOption::european_put(
            case["strike"].as_f64().unwrap(),
            expiry,
        );

        let result = price_european(
            &opt,
            case["spot"].as_f64().unwrap(),
            case["risk_free_rate"].as_f64().unwrap(),
            case["dividend_yield"].as_f64().unwrap(),
            case["volatility"].as_f64().unwrap(),
            case["time_to_expiry_years"].as_f64().unwrap(),
        );

        assert_abs_diff_eq!(result.npv, case["expected_price"].as_f64().unwrap(), epsilon = tol);
        assert_abs_diff_eq!(result.delta, case["expected_delta"].as_f64().unwrap(), epsilon = tol);
    }

    #[test]
    fn european_call_itm() {
        let golden = load_golden();
        let case = &golden["test_cases"]["european_call_itm"];
        let tol = case["tolerance"].as_f64().unwrap();

        let expiry = Date::from_ymd(2026, Month::January, 15);
        let opt = VanillaOption::european_call(
            case["strike"].as_f64().unwrap(),
            expiry,
        );

        let result = price_european(
            &opt,
            case["spot"].as_f64().unwrap(),
            case["risk_free_rate"].as_f64().unwrap(),
            case["dividend_yield"].as_f64().unwrap(),
            case["volatility"].as_f64().unwrap(),
            case["time_to_expiry_years"].as_f64().unwrap(),
        );

        assert_abs_diff_eq!(result.npv, case["expected_price"].as_f64().unwrap(), epsilon = tol);
    }

    #[test]
    fn european_call_with_dividend() {
        let golden = load_golden();
        let case = &golden["test_cases"]["european_call_with_dividend"];
        let tol = case["tolerance"].as_f64().unwrap();

        let expiry = Date::from_ymd(2026, Month::January, 15);
        let opt = VanillaOption::european_call(
            case["strike"].as_f64().unwrap(),
            expiry,
        );

        let result = price_european(
            &opt,
            case["spot"].as_f64().unwrap(),
            case["risk_free_rate"].as_f64().unwrap(),
            case["dividend_yield"].as_f64().unwrap(),
            case["volatility"].as_f64().unwrap(),
            case["time_to_expiry_years"].as_f64().unwrap(),
        );

        assert_abs_diff_eq!(result.npv, case["expected_price"].as_f64().unwrap(), epsilon = tol);
    }

    #[test]
    fn put_call_parity() {
        let golden = load_golden();
        let case = &golden["test_cases"]["put_call_parity"];
        let tol = case["parity_lhs_minus_rhs_tolerance"].as_f64().unwrap();

        let spot = case["spot"].as_f64().unwrap();
        let strike = case["strike"].as_f64().unwrap();
        let r = case["risk_free_rate"].as_f64().unwrap();
        let q = case["dividend_yield"].as_f64().unwrap();
        let vol = case["volatility"].as_f64().unwrap();
        let t = case["time_to_expiry_years"].as_f64().unwrap();

        let expiry = Date::from_ymd(2026, Month::January, 15);
        let call = VanillaOption::european_call(strike, expiry);
        let put = VanillaOption::european_put(strike, expiry);

        let call_result = price_european(&call, spot, r, q, vol, t);
        let put_result = price_european(&put, spot, r, q, vol, t);

        // Put-call parity: C - P = S*exp(-q*T) - K*exp(-r*T)
        let lhs = call_result.npv - put_result.npv;
        let rhs = spot * (-q * t).exp() - strike * (-r * t).exp();

        assert_abs_diff_eq!(lhs, rhs, epsilon = tol);
    }
}
