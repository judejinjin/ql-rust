//! Model parameters and constraints.
//!
//! Provides [`Parameter`] (a named, constrained vector of model parameters)
//! and the [`Constraint`] trait with concrete implementations.

// ---------------------------------------------------------------------------
// Constraint
// ---------------------------------------------------------------------------

/// A constraint on parameter values.
pub trait Constraint: Send + Sync {
    /// Check whether the given parameter values satisfy the constraint.
    fn test(&self, params: &[f64]) -> bool;

    /// Upper bounds for the parameters (may be +∞).
    fn upper_bound(&self, params: &[f64]) -> Vec<f64>;

    /// Lower bounds for the parameters (may be −∞).
    fn lower_bound(&self, params: &[f64]) -> Vec<f64>;
}

/// No constraint — any values are accepted.
#[derive(Clone, Debug, Default)]
pub struct NoConstraint;

impl Constraint for NoConstraint {
    fn test(&self, _params: &[f64]) -> bool {
        true
    }
    fn upper_bound(&self, params: &[f64]) -> Vec<f64> {
        vec![f64::INFINITY; params.len()]
    }
    fn lower_bound(&self, params: &[f64]) -> Vec<f64> {
        vec![f64::NEG_INFINITY; params.len()]
    }
}

/// All parameters must be strictly positive.
#[derive(Clone, Debug, Default)]
pub struct PositiveConstraint;

impl Constraint for PositiveConstraint {
    fn test(&self, params: &[f64]) -> bool {
        params.iter().all(|&p| p > 0.0)
    }
    fn upper_bound(&self, params: &[f64]) -> Vec<f64> {
        vec![f64::INFINITY; params.len()]
    }
    fn lower_bound(&self, params: &[f64]) -> Vec<f64> {
        vec![0.0; params.len()]
    }
}

/// Each parameter must lie within [lo, hi].
#[derive(Clone, Debug)]
pub struct BoundaryConstraint {
    /// Lower bounds.
    pub lo: Vec<f64>,
    /// Upper bounds.
    pub hi: Vec<f64>,
}

impl BoundaryConstraint {
    /// Create a boundary constraint.
    pub fn new(lo: Vec<f64>, hi: Vec<f64>) -> Self {
        assert_eq!(lo.len(), hi.len(), "lo and hi must have same length");
        Self { lo, hi }
    }
}

impl Constraint for BoundaryConstraint {
    fn test(&self, params: &[f64]) -> bool {
        params.iter().enumerate().all(|(i, &p)| p >= self.lo[i] && p <= self.hi[i])
    }
    fn upper_bound(&self, _params: &[f64]) -> Vec<f64> {
        self.hi.clone()
    }
    fn lower_bound(&self, _params: &[f64]) -> Vec<f64> {
        self.lo.clone()
    }
}

/// Composite constraint: both inner constraints must be satisfied.
pub struct CompositeConstraint {
    c1: Box<dyn Constraint>,
    c2: Box<dyn Constraint>,
}

impl CompositeConstraint {
    /// Combine two constraints.
    pub fn new(c1: Box<dyn Constraint>, c2: Box<dyn Constraint>) -> Self {
        Self { c1, c2 }
    }
}

impl Constraint for CompositeConstraint {
    fn test(&self, params: &[f64]) -> bool {
        self.c1.test(params) && self.c2.test(params)
    }
    fn upper_bound(&self, params: &[f64]) -> Vec<f64> {
        let u1 = self.c1.upper_bound(params);
        let u2 = self.c2.upper_bound(params);
        u1.iter().zip(u2.iter()).map(|(&a, &b)| a.min(b)).collect()
    }
    fn lower_bound(&self, params: &[f64]) -> Vec<f64> {
        let l1 = self.c1.lower_bound(params);
        let l2 = self.c2.lower_bound(params);
        l1.iter().zip(l2.iter()).map(|(&a, &b)| a.max(b)).collect()
    }
}

// ---------------------------------------------------------------------------
// Parameter
// ---------------------------------------------------------------------------

/// A named model parameter with an associated constraint.
pub struct Parameter {
    /// Parameter values (one or more).
    pub values: Vec<f64>,
    /// Constraint on the values.
    pub constraint: Box<dyn Constraint>,
}

impl Parameter {
    /// Create a new scalar parameter with a constraint.
    pub fn new(value: f64, constraint: Box<dyn Constraint>) -> Self {
        Self {
            values: vec![value],
            constraint,
        }
    }

    /// Create a new vector parameter.
    pub fn from_vec(values: Vec<f64>, constraint: Box<dyn Constraint>) -> Self {
        Self { values, constraint }
    }

    /// The scalar value (panics if multi-valued).
    pub fn value(&self) -> f64 {
        assert_eq!(self.values.len(), 1, "Parameter is multi-valued");
        self.values[0]
    }

    /// Set the scalar value.
    pub fn set_value(&mut self, v: f64) {
        assert_eq!(self.values.len(), 1, "Parameter is multi-valued");
        self.values[0] = v;
    }

    /// Check whether current values satisfy the constraint.
    pub fn is_valid(&self) -> bool {
        self.constraint.test(&self.values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_constraint_accepts_anything() {
        let c = NoConstraint;
        assert!(c.test(&[-100.0, 0.0, 100.0]));
    }

    #[test]
    fn positive_constraint_rejects_negative() {
        let c = PositiveConstraint;
        assert!(c.test(&[1.0, 2.0, 0.001]));
        assert!(!c.test(&[1.0, -0.001, 2.0]));
        assert!(!c.test(&[0.0]));
    }

    #[test]
    fn boundary_constraint() {
        let c = BoundaryConstraint::new(vec![0.0, -1.0], vec![1.0, 1.0]);
        assert!(c.test(&[0.5, 0.0]));
        assert!(!c.test(&[1.5, 0.0]));
        assert!(!c.test(&[0.5, -1.5]));
    }

    #[test]
    fn composite_constraint() {
        let c1 = Box::new(PositiveConstraint);
        let c2 = Box::new(BoundaryConstraint::new(vec![0.0], vec![10.0]));
        let cc = CompositeConstraint::new(c1, c2);
        assert!(cc.test(&[5.0]));
        assert!(!cc.test(&[-1.0]));
        assert!(!cc.test(&[11.0]));
    }

    #[test]
    fn parameter_scalar() {
        let mut p = Parameter::new(0.5, Box::new(PositiveConstraint));
        assert_eq!(p.value(), 0.5);
        assert!(p.is_valid());
        p.set_value(-1.0);
        assert!(!p.is_valid());
    }
}
