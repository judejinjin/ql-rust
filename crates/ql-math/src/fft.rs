//! Fast Fourier Transform for option pricing.
//!
//! Radix-2 Cooley-Tukey FFT, with Carr-Madan helper for European option pricing
//! via characteristic function inversion.

use std::f64::consts::PI;

/// Complex number (simple implementation to avoid extra dependency).
#[derive(Clone, Copy, Debug)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }

    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn exp(self) -> Self {
        let r = self.re.exp();
        Self {
            re: r * self.im.cos(),
            im: r * self.im.sin(),
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

/// In-place radix-2 Cooley-Tukey FFT.
///
/// `data` must have power-of-2 length. `inverse` = true for inverse FFT.
pub fn fft(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    assert!(n.is_power_of_two(), "FFT requires power-of-2 length");

    // Bit-reversal permutation
    let bits = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_reverse(i, bits);
        if i < j {
            data.swap(i, j);
        }
    }

    // Butterfly stages
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = sign * 2.0 * PI / len as f64;
        let wn = Complex::from_polar(1.0, angle);

        for start in (0..n).step_by(len) {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..half {
                let u = data[start + j];
                let t = w * data[start + j + half];
                data[start + j] = u + t;
                data[start + j + half] = u - t;
                w = w * wn;
            }
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / n as f64;
        for d in data.iter_mut() {
            d.re *= inv_n;
            d.im *= inv_n;
        }
    }
}

/// Compute DFT directly (for reference/testing). O(n²).
pub fn dft(data: &[Complex]) -> Vec<Complex> {
    let n = data.len();
    let mut result = vec![Complex::zero(); n];
    for (k, rk) in result.iter_mut().enumerate() {
        for (j, &val) in data.iter().enumerate() {
            let angle = -2.0 * PI * k as f64 * j as f64 / n as f64;
            let w = Complex::from_polar(1.0, angle);
            *rk = *rk + w * val;
        }
    }
    result
}

fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Carr-Madan FFT pricing for European call options.
///
/// Given a characteristic function `char_fn(u)` of the log-asset price,
/// computes call prices across a grid of log-strikes.
///
/// Returns (strikes, prices).
#[allow(clippy::too_many_arguments)]
pub fn carr_madan_fft<F>(
    char_fn: F,
    spot: f64,
    rate: f64,
    expiry: f64,
    n: usize,
    eta: f64,
    alpha: f64,
) -> (Vec<f64>, Vec<f64>)
where
    F: Fn(Complex) -> Complex,
{
    assert!(n.is_power_of_two());

    let lambda = 2.0 * PI / (n as f64 * eta);
    let b = lambda * n as f64 / 2.0;

    let discount = (-rate * expiry).exp();

    let mut x = vec![Complex::zero(); n];

    for (j, xj) in x.iter_mut().enumerate() {
        let v = j as f64 * eta;
        let u = Complex::new(v - (alpha + 1.0), 0.0);

        let cf = char_fn(u);

        let denom = Complex::new(
            alpha * alpha + alpha - v * v,
            (2.0 * alpha + 1.0) * v,
        );

        let integrand = (Complex::new(-rate * expiry, 0.0).exp() * cf)
            * Complex::new(1.0 / (denom.re * denom.re + denom.im * denom.im), 0.0)
            * Complex::new(denom.re, -denom.im);

        // Simpson's 3/8 weights
        let simpson = if j == 0 {
            eta / 3.0
        } else if j % 2 == 0 {
            2.0 * eta / 3.0
        } else {
            4.0 * eta / 3.0
        };

        // Phase shift for centering
        let phase = Complex::from_polar(1.0, b * v);

        *xj = integrand * phase * simpson;
    }

    fft(&mut x, false);

    let mut strikes = Vec::with_capacity(n);
    let mut prices = Vec::with_capacity(n);

    for (k, xk) in x.iter().enumerate() {
        let log_k = -b + lambda * k as f64;
        let strike = log_k.exp();
        let price = (discount * xk.re / PI).max(0.0);

        strikes.push(strike);
        prices.push(price);
    }

    let _ = (spot, discount); // spot incorporated via char_fn

    (strikes, prices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn fft_matches_dft() {
        let data: Vec<Complex> = (0..8)
            .map(|i| Complex::new(i as f64, (i as f64 * 0.3).sin()))
            .collect();

        let dft_result = dft(&data);

        let mut fft_data = data.clone();
        fft(&mut fft_data, false);

        for (i, (f, d)) in fft_data.iter().zip(dft_result.iter()).enumerate() {
            assert_abs_diff_eq!(f.re, d.re, epsilon = 1e-10);
            assert_abs_diff_eq!(f.im, d.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn fft_inverse_roundtrip() {
        let original: Vec<Complex> = (0..16)
            .map(|i| Complex::new((i as f64 * 0.5).sin(), (i as f64 * 0.3).cos()))
            .collect();

        let mut data = original.clone();
        fft(&mut data, false);
        fft(&mut data, true);

        for (orig, recovered) in original.iter().zip(data.iter()) {
            assert_abs_diff_eq!(orig.re, recovered.re, epsilon = 1e-10);
            assert_abs_diff_eq!(orig.im, recovered.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn fft_delta_function() {
        // FFT of [1, 0, 0, ..., 0] should give all 1s
        let mut data = vec![Complex::zero(); 8];
        data[0] = Complex::new(1.0, 0.0);
        fft(&mut data, false);
        for d in &data {
            assert_abs_diff_eq!(d.re, 1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(d.im, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn fft_constant_signal() {
        // FFT of constant = n * delta at k=0
        let n = 8;
        let mut data = vec![Complex::new(3.0, 0.0); n];
        fft(&mut data, false);
        assert_abs_diff_eq!(data[0].re, 24.0, epsilon = 1e-10); // n * 3
        for d in &data[1..] {
            assert_abs_diff_eq!(d.norm(), 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn fft_parsevals_theorem() {
        // Sum of |x|^2 in time domain = (1/N) * sum of |X|^2 in freq domain
        let data: Vec<Complex> = (0..16)
            .map(|i| Complex::new((i as f64 * 0.7).sin(), 0.0))
            .collect();
        let n = data.len();

        let time_energy: f64 = data.iter().map(|c| c.norm_sq()).sum();

        let mut freq = data.clone();
        fft(&mut freq, false);
        let freq_energy: f64 = freq.iter().map(|c| c.norm_sq()).sum::<f64>() / n as f64;

        assert_abs_diff_eq!(time_energy, freq_energy, epsilon = 1e-8);
    }

    #[test]
    fn fft_large() {
        // Test with 1024 points
        let n = 1024;
        let mut data: Vec<Complex> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                Complex::new((2.0 * PI * 3.0 * t).sin(), 0.0)
            })
            .collect();
        fft(&mut data, false);
        // Peak should be at k=3
        let peak_idx = data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).unwrap())
            .unwrap()
            .0;
        assert!(peak_idx == 3 || peak_idx == n - 3, "Peak at {peak_idx}, expected 3 or {}", n - 3);
    }

    #[test]
    fn complex_arithmetic() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let c = a * b;
        // (1+2i)(3+4i) = 3+4i+6i+8i² = 3+10i-8 = -5+10i
        assert_abs_diff_eq!(c.re, -5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c.im, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn complex_exp() {
        let z = Complex::new(0.0, PI);
        let e = z.exp();
        // e^{iπ} = -1
        assert_abs_diff_eq!(e.re, -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(e.im, 0.0, epsilon = 1e-10);
    }
}
