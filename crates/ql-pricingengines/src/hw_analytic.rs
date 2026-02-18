//! Analytic Hull-White pricing engines.
//!
//! Provides closed-form pricing for:
//! - Zero-coupon bond options (calls/puts)
//! - Caplets/floorlets
//! - European swaptions (Jamshidian decomposition)
//!
//! All formulas assume constant a and σ in the Hull-White model
//! dr = (θ(t) − a r) dt + σ dW.

#![allow(clippy::too_many_arguments)]

use ql_math::distributions::NormalDistribution;

/// Result from an analytic Hull-White pricing.
#[derive(Debug, Clone)]
pub struct HWAnalyticResult {
    pub npv: f64,
}

/// Price a European option on a zero-coupon bond under Hull-White.
///
/// Call/put on a zero-coupon bond maturing at `bond_maturity`,
/// with option expiry at `option_expiry` and strike `strike`.
///
/// P(0, T) = discount factor to bond maturity T
/// P(0, S) = discount factor to option expiry S
///
/// σ_P = (σ/a)(1 − e^{−a(T−S)}) √((1 − e^{−2aS})/(2a))
///
/// Then price = ω [P(0,T) N(ω d₁) − K P(0,S) N(ω d₂)]
///
/// # Parameters
/// - `a`: mean-reversion speed
/// - `sigma`: HW volatility
/// - `p_option`: discount factor P(0, option_expiry)
/// - `p_bond`: discount factor P(0, bond_maturity)
/// - `option_expiry`: time to option expiry
/// - `bond_maturity`: time to bond maturity
/// - `strike`: bond option strike
/// - `is_call`: true for call, false for put
pub fn hw_bond_option(
    a: f64,
    sigma: f64,
    p_option: f64,
    p_bond: f64,
    option_expiry: f64,
    bond_maturity: f64,
    strike: f64,
    is_call: bool,
) -> HWAnalyticResult {
    let n = NormalDistribution::standard();
    let omega = if is_call { 1.0 } else { -1.0 };

    let tau = bond_maturity - option_expiry;
    let b_tau = if a.abs() < 1e-15 {
        tau
    } else {
        (1.0 - (-a * tau).exp()) / a
    };

    let sigma_p = if a.abs() < 1e-15 {
        sigma * tau * option_expiry.sqrt()
    } else {
        sigma * b_tau * ((1.0 - (-2.0 * a * option_expiry).exp()) / (2.0 * a)).sqrt()
    };

    if sigma_p < 1e-15 {
        // Degenerate case
        let intrinsic = (omega * (p_bond - strike * p_option)).max(0.0);
        return HWAnalyticResult { npv: intrinsic };
    }

    let d1 = (p_bond / (p_option * strike)).ln() / sigma_p + 0.5 * sigma_p;
    let d2 = d1 - sigma_p;

    let npv = omega * (p_bond * n.cdf(omega * d1) - strike * p_option * n.cdf(omega * d2));

    HWAnalyticResult { npv: npv.max(0.0) }
}

/// Price a caplet under Hull-White.
///
/// A caplet pays max(L(S,T) − K, 0) × τ × N at time T,
/// where L is the LIBOR rate, S is the fixing date, T is the payment date.
///
/// This is equivalent to (1 + K τ) puts on a zero-coupon bond
/// with strike 1/(1 + K τ).
///
/// # Parameters
/// - `a`: mean-reversion
/// - `sigma`: HW vol
/// - `p_fixing`: P(0, fixing_date)
/// - `p_payment`: P(0, payment_date)
/// - `fixing_date`: caplet fixing time
/// - `payment_date`: caplet payment time
/// - `strike_rate`: cap strike rate
/// - `notional`: notional amount
pub fn hw_caplet(
    a: f64,
    sigma: f64,
    p_fixing: f64,
    p_payment: f64,
    fixing_date: f64,
    payment_date: f64,
    strike_rate: f64,
    notional: f64,
) -> HWAnalyticResult {
    let tau = payment_date - fixing_date;
    let bond_strike = 1.0 / (1.0 + strike_rate * tau);

    // Caplet = (1 + K τ) × Put on ZCB
    let put = hw_bond_option(a, sigma, p_fixing, p_payment, fixing_date, payment_date, bond_strike, false);

    HWAnalyticResult {
        npv: notional * (1.0 + strike_rate * tau) * put.npv,
    }
}

/// Price a floorlet under Hull-White.
///
/// A floorlet pays max(K − L(S,T), 0) × τ × N at time T.
/// This is equivalent to (1 + K τ) calls on a zero-coupon bond.
pub fn hw_floorlet(
    a: f64,
    sigma: f64,
    p_fixing: f64,
    p_payment: f64,
    fixing_date: f64,
    payment_date: f64,
    strike_rate: f64,
    notional: f64,
) -> HWAnalyticResult {
    let tau = payment_date - fixing_date;
    let bond_strike = 1.0 / (1.0 + strike_rate * tau);

    let call = hw_bond_option(a, sigma, p_fixing, p_payment, fixing_date, payment_date, bond_strike, true);

    HWAnalyticResult {
        npv: notional * (1.0 + strike_rate * tau) * call.npv,
    }
}

/// Price a European swaption under Hull-White via Jamshidian decomposition.
///
/// The Jamshidian trick decomposes a coupon bond option into a portfolio
/// of zero-coupon bond options, all with the same critical rate r*.
///
/// For a payer swaption (right to enter pay-fixed swap):
///   Value = Σᵢ cᵢ Put(P(0,Tᵢ), Kᵢ) where Kᵢ = P_model(r*, Tᵢ)
///
/// # Parameters
/// - `a`: mean-reversion
/// - `sigma`: HW vol
/// - `option_expiry`: swaption expiry
/// - `swap_tenors`: payment times [T₁, T₂, ..., Tₙ] after option expiry
/// - `fixed_rate`: swap fixed rate
/// - `discount_factors`: P(0, Tᵢ) for each payment date
/// - `p_option`: P(0, option_expiry)
/// - `notional`: swap notional
/// - `is_payer`: true for payer swaption, false for receiver
#[allow(clippy::too_many_arguments)]
pub fn hw_jamshidian_swaption(
    a: f64,
    sigma: f64,
    option_expiry: f64,
    swap_tenors: &[f64],
    fixed_rate: f64,
    discount_factors: &[f64],
    p_option: f64,
    notional: f64,
    is_payer: bool,
) -> HWAnalyticResult {
    let n_payments = swap_tenors.len();
    assert_eq!(discount_factors.len(), n_payments);

    // Coupon amounts: cᵢ = fixed_rate × (Tᵢ − Tᵢ₋₁)
    // Last payment includes notional return: cₙ += 1
    let mut coupons = Vec::with_capacity(n_payments);
    for i in 0..n_payments {
        let tau_i = if i == 0 {
            swap_tenors[0] - option_expiry
        } else {
            swap_tenors[i] - swap_tenors[i - 1]
        };
        let mut c = fixed_rate * tau_i;
        if i == n_payments - 1 {
            c += 1.0; // principal return
        }
        coupons.push(c);
    }

    // Find critical rate r* such that Σ cᵢ P(r*, Tᵢ) = 1
    // P(r*, Tᵢ) = A(Tᵢ − S) exp(−B(Tᵢ − S) r*) with S = option_expiry
    // Use Newton's method
    let taus: Vec<f64> = swap_tenors.iter().map(|&t| t - option_expiry).collect();
    let b_vals: Vec<f64> = taus
        .iter()
        .map(|&tau| {
            if a.abs() < 1e-15 {
                tau
            } else {
                (1.0 - (-a * tau).exp()) / a
            }
        })
        .collect();

    // A(τ) for HW with constant a, sigma — uses the formula
    // A(τ) = P(0,T)/P(0,S) exp(B(τ) f(0,S) − σ²/(4a) B(τ)² (1−e^{−2aS}))
    // We simplify: A_model(τ, r) = P(0, S+τ)/P(0,S) × adjustment
    // For Jamshidian, we just need bond prices at the critical rate
    // P_model(r, τ) = exp(A_coeff(τ) − B(τ) r)
    // where A_coeff captures the curve fit.

    // Actually, for the Jamshidian trick, we need:
    // Bond strike K_i = P_HW(r*, T_i - S) where P_HW is the model bond price
    // at the critical rate r*.
    //
    // Use the fitted A(τ) = ln(P(0,T_i)/P(0,S)) + B(τ_i) f(0,S) - σ²B²(1-e^{-2aS})/(4a)

    // Forward rate at option_expiry (approximated from discount factors)
    // f(0,S) ≈ -d/dS ln P(0,S) ≈ -ln(P(0,S))/S for simplicity
    // Better: use the exact fit. For now, we use the provided discount factors.

    // Simplified approach: given P(0,Tᵢ) and P(0,S), we know:
    //   P_model(r, τᵢ) = P(0,Tᵢ)/P(0,S) × exp(-B(τᵢ)(r - r_fwd) + convexity)
    // where r_fwd is the model's forward rate.
    //
    // For practical Jamshidian, the key formula is:
    //   K_i = P(0,T_i)/P(0,S) × exp(-B(τ_i) (r* - r_fwd))
    // where r_fwd is determined from the model.
    //
    // In the constant-parameter HW with exact curve fit:
    //   ln P_model(r, τ) = ln(P(0,T)/P(0,S)) + B(τ)(f_S - r) - σ²B(τ)²(1-e^{-2aS})/(4a)
    //
    // We compute A_i coefficients from market data:
    let s = option_expiry;
    let var_factor = if a.abs() < 1e-15 {
        sigma * sigma * s
    } else {
        sigma * sigma * (1.0 - (-2.0 * a * s).exp()) / (2.0 * a)
    };

    // Forward rate f(0,S) from market discount factors
    let f_s = -p_option.ln() / s; // instantaneous forward ≈ spot rate at S

    // A_i = ln(P(0,T_i)) - ln(P(0,S)) + B_i × f_s - 0.5 × B_i² × var_factor
    let a_coeffs: Vec<f64> = (0..n_payments)
        .map(|i| {
            (discount_factors[i] / p_option).ln() + b_vals[i] * f_s
                - 0.5 * b_vals[i] * b_vals[i] * var_factor
        })
        .collect();

    // Find r* via Newton: g(r*) = Σ cᵢ exp(Aᵢ − Bᵢ r*) − 1 = 0
    let mut r_star = f_s;
    for _ in 0..50 {
        let mut g = -1.0;
        let mut g_prime = 0.0;
        for i in 0..n_payments {
            let p_i = (a_coeffs[i] - b_vals[i] * r_star).exp();
            g += coupons[i] * p_i;
            g_prime -= coupons[i] * b_vals[i] * p_i;
        }
        if g_prime.abs() < 1e-30 {
            break;
        }
        let dr = g / g_prime;
        r_star -= dr;
        if dr.abs() < 1e-12 {
            break;
        }
    }

    // Bond strikes: K_i = exp(A_i - B_i r*)
    let bond_strikes: Vec<f64> = (0..n_payments)
        .map(|i| (a_coeffs[i] - b_vals[i] * r_star).exp())
        .collect();

    // Sum of ZCB options
    // For payer swaption: we want coupon bond put = Σ cᵢ Put(ZCB_i, K_i)
    // For receiver swaption: Σ cᵢ Call(ZCB_i, K_i)
    let is_bond_call = !is_payer; // payer = put on coupon bond
    let mut total = 0.0;
    for i in 0..n_payments {
        let res = hw_bond_option(
            a,
            sigma,
            p_option,
            discount_factors[i],
            option_expiry,
            swap_tenors[i],
            bond_strikes[i],
            is_bond_call,
        );
        total += coupons[i] * res.npv;
    }

    HWAnalyticResult {
        npv: notional * total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn hw_bond_option_call_positive() {
        let p_s = (-0.05_f64).exp(); // P(0,1)
        let p_t = (-0.05 * 5.0_f64).exp(); // P(0,5)
        let res = hw_bond_option(0.1, 0.01, p_s, p_t, 1.0, 5.0, 0.80, true);
        assert!(res.npv > 0.0, "Bond call should be positive: {}", res.npv);
    }

    #[test]
    fn hw_bond_option_put_call_parity() {
        let p_s = (-0.05_f64).exp();
        let p_t = (-0.05 * 5.0_f64).exp();
        let k = 0.80;
        let call = hw_bond_option(0.1, 0.01, p_s, p_t, 1.0, 5.0, k, true);
        let put = hw_bond_option(0.1, 0.01, p_s, p_t, 1.0, 5.0, k, false);
        let parity = call.npv - put.npv - p_t + k * p_s;
        assert_abs_diff_eq!(parity, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn hw_bond_option_higher_vol_higher_price() {
        let p_s = (-0.05_f64).exp();
        let p_t = (-0.05 * 5.0_f64).exp();
        let c_low = hw_bond_option(0.1, 0.005, p_s, p_t, 1.0, 5.0, 0.80, true);
        let c_high = hw_bond_option(0.1, 0.02, p_s, p_t, 1.0, 5.0, 0.80, true);
        assert!(c_high.npv > c_low.npv, "Higher vol should give higher price");
    }

    #[test]
    fn hw_caplet_positive() {
        let p_fix = (-0.05_f64).exp(); // P(0,1)
        let p_pay = (-0.05 * 1.25_f64).exp(); // P(0,1.25)
        let res = hw_caplet(0.1, 0.01, p_fix, p_pay, 1.0, 1.25, 0.05, 1_000_000.0);
        assert!(res.npv > 0.0, "Caplet should be positive: {}", res.npv);
    }

    #[test]
    fn hw_caplet_floorlet_parity() {
        let p_fix = (-0.05_f64).exp();
        let p_pay = (-0.05 * 1.25_f64).exp();
        let tau = 0.25;
        let k = 0.05;
        let notional = 1_000_000.0;

        let cap = hw_caplet(0.1, 0.01, p_fix, p_pay, 1.0, 1.25, k, notional);
        let floor = hw_floorlet(0.1, 0.01, p_fix, p_pay, 1.0, 1.25, k, notional);

        // Caplet - Floorlet = Notional × (P_fix - (1+Kτ) P_pay)
        let forward_val = notional * (p_fix - (1.0 + k * tau) * p_pay);
        let parity = cap.npv - floor.npv - forward_val;
        assert_abs_diff_eq!(parity, 0.0, epsilon = 0.01);
    }

    #[test]
    fn hw_jamshidian_swaption_positive() {
        // 1Y into 4Y swaption, annual payments
        let r: f64 = 0.05;
        let option_expiry = 1.0;
        let swap_tenors = vec![2.0, 3.0, 4.0, 5.0];
        let fixed_rate = 0.05;
        let discount_factors: Vec<f64> = swap_tenors.iter().map(|&t| (-r * t).exp()).collect();
        let p_option = (-r * option_expiry).exp();

        let res = hw_jamshidian_swaption(
            0.1,
            0.01,
            option_expiry,
            &swap_tenors,
            fixed_rate,
            &discount_factors,
            p_option,
            1_000_000.0,
            true,
        );
        assert!(
            res.npv > 0.0,
            "Payer swaption should be positive: {}",
            res.npv
        );
    }

    #[test]
    fn hw_jamshidian_swaption_payer_receiver() {
        let r: f64 = 0.05;
        let option_expiry = 1.0;
        let swap_tenors = vec![2.0, 3.0, 4.0, 5.0];
        let fixed_rate = 0.05;
        let dfs: Vec<f64> = swap_tenors.iter().map(|&t| (-r * t).exp()).collect();
        let p_opt = (-r * option_expiry).exp();

        let payer = hw_jamshidian_swaption(
            0.1, 0.01, option_expiry, &swap_tenors, fixed_rate, &dfs, p_opt, 1.0, true,
        );
        let receiver = hw_jamshidian_swaption(
            0.1, 0.01, option_expiry, &swap_tenors, fixed_rate, &dfs, p_opt, 1.0, false,
        );

        // Payer − Receiver = value of underlying swap at expiry
        // = P(0,S) − Σ cᵢ P(0,Tᵢ) where cᵢ includes final principal
        let mut swap_val = p_opt;
        for i in 0..swap_tenors.len() {
            let tau_i = if i == 0 {
                swap_tenors[0] - option_expiry
            } else {
                swap_tenors[i] - swap_tenors[i - 1]
            };
            let mut c = fixed_rate * tau_i;
            if i == swap_tenors.len() - 1 {
                c += 1.0;
            }
            swap_val -= c * dfs[i];
        }

        let diff = payer.npv - receiver.npv - swap_val;
        assert_abs_diff_eq!(diff, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn hw_jamshidian_higher_vol_higher_price() {
        let r: f64 = 0.05;
        let option_expiry = 1.0;
        let swap_tenors = vec![2.0, 3.0, 4.0, 5.0];
        let fixed_rate = 0.05;
        let dfs: Vec<f64> = swap_tenors.iter().map(|&t| (-r * t).exp()).collect();
        let p_opt = (-r * option_expiry).exp();

        let low_vol = hw_jamshidian_swaption(
            0.1, 0.005, option_expiry, &swap_tenors, fixed_rate, &dfs, p_opt, 1.0, true,
        );
        let high_vol = hw_jamshidian_swaption(
            0.1, 0.02, option_expiry, &swap_tenors, fixed_rate, &dfs, p_opt, 1.0, true,
        );

        assert!(
            high_vol.npv > low_vol.npv,
            "Higher HW vol should give higher swaption: {} vs {}",
            high_vol.npv,
            low_vol.npv
        );
    }
}
