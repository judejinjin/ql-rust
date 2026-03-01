//! Generic complex number type for AD-aware characteristic function evaluation.
//!
//! This is a minimal complex number implementation generic over `T: Number`,
//! enabling derivative propagation through the Heston/Bates characteristic
//! function. It replaces the duplicated inline `struct C` from the pricing
//! engine files.

use crate::number::Number;

/// A complex number with real and imaginary parts of type `T`.
///
/// When `T = f64`, this is a standard complex number.
/// When `T = Dual` or `DualVec<N>`, derivatives propagate through all
/// complex arithmetic and transcendental operations.
#[derive(Clone, Copy, Debug)]
pub struct Complex<T: Number> {
    pub re: T,
    pub im: T,
}

impl<T: Number> Complex<T> {
    /// Create a complex number from real and imaginary parts.
    #[inline]
    pub fn new(re: T, im: T) -> Self {
        Self { re, im }
    }

    /// Create a purely real complex number.
    #[inline]
    pub fn from_real(re: T) -> Self {
        Self { re, im: T::zero() }
    }

    /// Create a purely imaginary complex number.
    #[inline]
    pub fn from_imag(im: T) -> Self {
        Self { re: T::zero(), im }
    }

    /// Squared modulus: |z|² = re² + im².
    #[inline]
    pub fn norm_sq(self) -> T {
        self.re * self.re + self.im * self.im
    }

    /// Modulus: |z| = √(re² + im²).
    #[inline]
    pub fn abs(self) -> T {
        self.norm_sq().sqrt()
    }

    /// Argument (phase angle): arg(z) = atan2(im, re).
    #[inline]
    pub fn arg(self) -> T {
        self.im.atan2(self.re)
    }

    /// Complex exponential: exp(a + bi) = e^a (cos b + i sin b).
    #[inline]
    pub fn exp(self) -> Self {
        let r = self.re.exp();
        Self {
            re: r * self.im.cos(),
            im: r * self.im.sin(),
        }
    }

    /// Complex natural logarithm (principal branch):
    ///     ln(z) = ln|z| + i·arg(z)
    #[inline]
    pub fn ln(self) -> Self {
        Self {
            re: self.abs().ln(),
            im: self.arg(),
        }
    }

    /// Complex square root (principal branch):
    ///     √z = √|z| · (cos(θ/2) + i·sin(θ/2))
    #[inline]
    pub fn sqrt(self) -> Self {
        let r = self.abs().sqrt();
        let half_theta = self.arg() * T::half();
        Self {
            re: r * half_theta.cos(),
            im: r * half_theta.sin(),
        }
    }

    /// Scalar multiply: z * c.
    #[inline]
    pub fn scale(self, c: T) -> Self {
        Self {
            re: self.re * c,
            im: self.im * c,
        }
    }

    /// Conjugate: re - i·im.
    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: T::zero() - self.im,
        }
    }
}

// --- Arithmetic operators ---

impl<T: Number> std::ops::Add for Complex<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl<T: Number> std::ops::Sub for Complex<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl<T: Number> std::ops::Mul for Complex<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl<T: Number> std::ops::Div for Complex<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let d = rhs.norm_sq();
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / d,
            im: (self.im * rhs.re - self.re * rhs.im) / d,
        }
    }
}

impl<T: Number> std::ops::Neg for Complex<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            re: T::zero() - self.re,
            im: T::zero() - self.im,
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn complex_add() {
        let a = Complex::<f64>::new(1.0, 2.0);
        let b = Complex::<f64>::new(3.0, 4.0);
        let c = a + b;
        assert_abs_diff_eq!(c.re, 4.0, epsilon = 1e-15);
        assert_abs_diff_eq!(c.im, 6.0, epsilon = 1e-15);
    }

    #[test]
    fn complex_mul() {
        // (1+2i)(3+4i) = 3+4i+6i+8i² = -5+10i
        let a = Complex::<f64>::new(1.0, 2.0);
        let b = Complex::<f64>::new(3.0, 4.0);
        let c = a * b;
        assert_abs_diff_eq!(c.re, -5.0, epsilon = 1e-14);
        assert_abs_diff_eq!(c.im, 10.0, epsilon = 1e-14);
    }

    #[test]
    fn complex_div() {
        // (1+2i) / (1+2i) = 1
        let a = Complex::<f64>::new(1.0, 2.0);
        let c = a / a;
        assert_abs_diff_eq!(c.re, 1.0, epsilon = 1e-14);
        assert_abs_diff_eq!(c.im, 0.0, epsilon = 1e-14);
    }

    #[test]
    fn complex_exp() {
        // exp(0 + πi) = -1 + 0i  (Euler's identity)
        let z = Complex::<f64>::new(0.0, std::f64::consts::PI);
        let e = z.exp();
        assert_abs_diff_eq!(e.re, -1.0, epsilon = 1e-14);
        assert_abs_diff_eq!(e.im, 0.0, epsilon = 1e-14);
    }

    #[test]
    fn complex_ln_exp_roundtrip() {
        let z = Complex::<f64>::new(1.0, 2.0);
        let rt = z.ln().exp();
        assert_abs_diff_eq!(rt.re, z.re, epsilon = 1e-12);
        assert_abs_diff_eq!(rt.im, z.im, epsilon = 1e-12);
    }

    #[test]
    fn complex_sqrt_squared() {
        let z = Complex::<f64>::new(3.0, 4.0);
        let s = z.sqrt();
        let s2 = s * s;
        assert_abs_diff_eq!(s2.re, z.re, epsilon = 1e-12);
        assert_abs_diff_eq!(s2.im, z.im, epsilon = 1e-12);
    }

    #[test]
    fn complex_norm_sq() {
        let z = Complex::<f64>::new(3.0, 4.0);
        assert_abs_diff_eq!(z.norm_sq(), 25.0, epsilon = 1e-14);
        assert_abs_diff_eq!(z.abs(), 5.0, epsilon = 1e-14);
    }

    #[test]
    fn complex_arg() {
        let z = Complex::<f64>::new(1.0, 1.0);
        assert_abs_diff_eq!(z.arg(), std::f64::consts::FRAC_PI_4, epsilon = 1e-14);
    }

    #[test]
    fn complex_dual_derivative_through_mul() {
        use crate::dual::Dual;
        // z = (x + 0i) * (0 + i) = 0 + xi  where x is variable
        let x = Dual::variable(3.0);
        let z1 = Complex::new(x, Dual::constant(0.0));
        let z2 = Complex::new(Dual::constant(0.0), Dual::constant(1.0));
        let z = z1 * z2;
        // z.re = 0, z.im = x → dz.im/dx = 1
        assert_abs_diff_eq!(z.re.dot, 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(z.im.dot, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn complex_dual_exp() {
        use crate::dual::Dual;
        // exp(x + 0i) = exp(x) + 0i, d/dx exp(x) = exp(x)
        let x = Dual::variable(1.0);
        let z = Complex::new(x, Dual::constant(0.0));
        let e = z.exp();
        assert_abs_diff_eq!(e.re.val, 1.0_f64.exp(), epsilon = 1e-12);
        assert_abs_diff_eq!(e.re.dot, 1.0_f64.exp(), epsilon = 1e-12);
        assert_abs_diff_eq!(e.im.val, 0.0, epsilon = 1e-12);
    }
}
