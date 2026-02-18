//! Optionlet/caplet volatility stripping from cap quotes.
//!
//! Strips flat cap volatilities into individual optionlet (caplet) volatilities
//! using a bootstrap procedure. This is essential for pricing non-standard
//! cap/floor products.
//!
//! The key insight: a cap is a portfolio of caplets (optionlets). Market quotes
//! give a single flat vol for each cap maturity. To price individual caplets,
//! we need to extract per-period optionlet volatilities.
//!
//! Reference: Brigo & Mercurio, "Interest Rate Models — Theory and Practice," Ch. 6.

/// Result of optionlet stripping.
#[derive(Debug, Clone)]
pub struct StrippedOptionletVolatilities {
    /// Caplet maturities (in years from today).
    pub maturities: Vec<f64>,
    /// Stripped optionlet volatilities for each maturity.
    pub optionlet_vols: Vec<f64>,
}

/// Black caplet price using lognormal model.
///
/// Price = discount × τ × [F Φ(d₁) − K Φ(d₂)]
/// where d₁ = [ln(F/K) + σ²T/2] / (σ√T), d₂ = d₁ − σ√T.
fn black_caplet_price(
    forward: f64,
    strike: f64,
    vol: f64,
    expiry: f64,
    accrual: f64,
    discount: f64,
) -> f64 {
    if vol <= 0.0 || expiry <= 0.0 {
        return discount * accrual * (forward - strike).max(0.0);
    }

    let std_dev = vol * expiry.sqrt();
    let d1 = ((forward / strike).ln() + 0.5 * std_dev * std_dev) / std_dev;
    let d2 = d1 - std_dev;

    let n = ql_math::distributions::NormalDistribution::standard();
    discount * accrual * (forward * n.cdf(d1) - strike * n.cdf(d2))
}

/// Cap price as sum of caplet prices, all using the same flat volatility.
fn cap_price_flat_vol(
    forwards: &[f64],
    strike: f64,
    flat_vol: f64,
    expiries: &[f64],
    accruals: &[f64],
    discounts: &[f64],
) -> f64 {
    let n = forwards.len();
    let mut total = 0.0;
    for i in 0..n {
        total += black_caplet_price(forwards[i], strike, flat_vol, expiries[i], accruals[i], discounts[i]);
    }
    total
}

/// Strip optionlet volatilities from flat cap volatilities using bootstrap.
///
/// # Parameters
/// - `cap_maturities` — sorted cap maturities (in years), e.g. [1, 2, 3, 5, 7, 10]
/// - `cap_flat_vols` — market flat cap volatilities for each maturity
/// - `strike` — ATM or fixed strike for the caps
/// - `forwards` — forward rates for each caplet period
/// - `expiries` — fixing dates for each caplet (time from today)
/// - `accruals` — year fractions for each caplet period
/// - `discounts` — discount factors to each caplet payment date
///
/// The caplets are ordered by expiry. Each cap with maturity M is the sum of
/// all caplets with expiry ≤ M.
///
/// # Returns
/// Stripped optionlet volatilities for each caplet.
pub fn strip_optionlet_volatilities(
    cap_maturities: &[f64],
    cap_flat_vols: &[f64],
    strike: f64,
    forwards: &[f64],
    expiries: &[f64],
    accruals: &[f64],
    discounts: &[f64],
) -> StrippedOptionletVolatilities {
    let n_caplets = forwards.len();
    let n_caps = cap_maturities.len();
    assert_eq!(n_caplets, expiries.len());
    assert_eq!(n_caplets, accruals.len());
    assert_eq!(n_caplets, discounts.len());
    assert_eq!(n_caps, cap_flat_vols.len());

    let mut optionlet_vols = vec![0.0_f64; n_caplets];

    // For each cap maturity, determine which caplets it includes
    let mut cap_idx = 0;
    for i in 0..n_caplets {
        // Find the cap that first includes this caplet
        while cap_idx < n_caps && cap_maturities[cap_idx] < expiries[i] - 1e-10 {
            cap_idx += 1;
        }
        if cap_idx >= n_caps {
            // Extrapolate: use the last known optionlet vol
            optionlet_vols[i] = if i > 0 { optionlet_vols[i - 1] } else { cap_flat_vols[n_caps - 1] };
            continue;
        }

        // Cap price target = price using flat vol for all caplets up to this cap
        let target_cap_price = cap_price_flat_vol(
            &forwards[..=i],
            strike,
            cap_flat_vols[cap_idx],
            &expiries[..=i],
            &accruals[..=i],
            &discounts[..=i],
        );

        // Price of already-stripped caplets 0..i
        let mut known_price = 0.0;
        for j in 0..i {
            known_price += black_caplet_price(
                forwards[j],
                strike,
                optionlet_vols[j],
                expiries[j],
                accruals[j],
                discounts[j],
            );
        }

        // Residual price for caplet i
        let residual = target_cap_price - known_price;

        if residual <= 0.0 {
            // Edge case: use flat vol
            optionlet_vols[i] = cap_flat_vols[cap_idx];
            continue;
        }

        // Solve for σ_i such that black_caplet_price(σ_i) = residual
        // Use bisection
        let mut lo = 1e-6_f64;
        let mut hi = 5.0_f64;
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            let price = black_caplet_price(forwards[i], strike, mid, expiries[i], accruals[i], discounts[i]);
            if price < residual {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        optionlet_vols[i] = 0.5 * (lo + hi);
    }

    StrippedOptionletVolatilities {
        maturities: expiries.to_vec(),
        optionlet_vols,
    }
}

/// Interpolate stripped optionlet vols to get vol at an arbitrary expiry.
///
/// Uses linear interpolation on the stripped maturities/vols.
pub fn interpolate_optionlet_vol(
    stripped: &StrippedOptionletVolatilities,
    expiry: f64,
) -> f64 {
    let mats = &stripped.maturities;
    let vols = &stripped.optionlet_vols;

    if mats.is_empty() {
        return 0.0;
    }
    if expiry <= mats[0] {
        return vols[0];
    }
    if expiry >= *mats.last().unwrap() {
        return *vols.last().unwrap();
    }

    // Find bracketing interval
    let mut i = 0;
    while i < mats.len() - 1 && mats[i + 1] < expiry {
        i += 1;
    }

    let t = (expiry - mats[i]) / (mats[i + 1] - mats[i]);
    vols[i] * (1.0 - t) + vols[i + 1] * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // Build a simple 3-caplet scenario with constant forwards and flat term structure
    fn setup_3_caplets() -> (Vec<f64>, Vec<f64>, f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let cap_maturities = vec![1.0, 2.0, 3.0];
        let cap_flat_vols = vec![0.20, 0.22, 0.25];
        let strike = 0.03;
        let forwards = vec![0.035, 0.036, 0.037];
        let expiries = vec![0.75, 1.75, 2.75]; // fixing dates
        let accruals = vec![0.25, 0.25, 0.25]; // quarterly
        let r: f64 = 0.03;
        let discounts = vec![
            (-r * 1.0).exp(),
            (-r * 2.0).exp(),
            (-r * 3.0).exp(),
        ];

        (cap_maturities, cap_flat_vols, strike, forwards, expiries, accruals, discounts)
    }

    #[test]
    fn black_caplet_price_positive() {
        let price = black_caplet_price(0.035, 0.03, 0.20, 1.0, 0.25, 0.97);
        assert!(price > 0.0, "Caplet price should be positive: {price}");
    }

    #[test]
    fn black_caplet_itm() {
        // Deep ITM caplet: price ≈ discount × accrual × (F − K)
        let price = black_caplet_price(0.10, 0.03, 0.20, 1.0, 0.25, 0.97);
        let intrinsic = 0.97 * 0.25 * (0.10 - 0.03);
        assert!(price >= intrinsic * 0.99, "Deep ITM caplet should be near intrinsic");
    }

    #[test]
    fn strip_single_caplet() {
        // With one caplet, optionlet vol = flat cap vol
        let stripped = strip_optionlet_volatilities(
            &[1.0],
            &[0.20],
            0.03,
            &[0.035],
            &[0.75],
            &[0.25],
            &[0.97],
        );
        assert_eq!(stripped.optionlet_vols.len(), 1);
        assert_abs_diff_eq!(stripped.optionlet_vols[0], 0.20, epsilon = 1e-6);
    }

    #[test]
    fn stripped_vols_positive() {
        let (mats, vols, k, fwds, exp, acc, disc) = setup_3_caplets();
        let stripped = strip_optionlet_volatilities(&mats, &vols, k, &fwds, &exp, &acc, &disc);
        for (i, v) in stripped.optionlet_vols.iter().enumerate() {
            assert!(*v > 0.0, "Optionlet vol {i} should be positive: {v}");
        }
    }

    #[test]
    fn stripped_vols_recover_cap_prices() {
        let (mats, vols, k, fwds, exp, acc, disc) = setup_3_caplets();
        let stripped = strip_optionlet_volatilities(&mats, &vols, k, &fwds, &exp, &acc, &disc);

        // For each cap maturity, the sum of stripped-vol caplets should match the flat-vol cap price
        for (cap_i, &cap_mat) in mats.iter().enumerate() {
            let flat_price = cap_price_flat_vol(&fwds[..=cap_i], k, vols[cap_i], &exp[..=cap_i], &acc[..=cap_i], &disc[..=cap_i]);
            let mut stripped_price = 0.0;
            for j in 0..=cap_i {
                stripped_price += black_caplet_price(fwds[j], k, stripped.optionlet_vols[j], exp[j], acc[j], disc[j]);
            }
            assert_abs_diff_eq!(
                flat_price, stripped_price,
                epsilon = flat_price * 1e-4
            );
            let _ = cap_mat;
        }
    }

    #[test]
    fn interpolate_within_range() {
        let stripped = StrippedOptionletVolatilities {
            maturities: vec![1.0, 2.0, 3.0],
            optionlet_vols: vec![0.20, 0.25, 0.30],
        };
        let v = interpolate_optionlet_vol(&stripped, 1.5);
        assert_abs_diff_eq!(v, 0.225, epsilon = 1e-10);
    }

    #[test]
    fn interpolate_extrapolate_flat() {
        let stripped = StrippedOptionletVolatilities {
            maturities: vec![1.0, 2.0, 3.0],
            optionlet_vols: vec![0.20, 0.25, 0.30],
        };
        // Below range → first vol
        assert_abs_diff_eq!(interpolate_optionlet_vol(&stripped, 0.5), 0.20, epsilon = 1e-10);
        // Above range → last vol
        assert_abs_diff_eq!(interpolate_optionlet_vol(&stripped, 5.0), 0.30, epsilon = 1e-10);
    }

    #[test]
    fn cap_price_increases_with_vol() {
        let fwds = [0.035];
        let exp = [0.75];
        let acc = [0.25];
        let disc = [0.97];
        let k = 0.03;
        let p1 = cap_price_flat_vol(&fwds, k, 0.15, &exp, &acc, &disc);
        let p2 = cap_price_flat_vol(&fwds, k, 0.30, &exp, &acc, &disc);
        assert!(p2 > p1, "Cap price should increase with vol: {p1} vs {p2}");
    }
}
