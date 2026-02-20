//! Composite instrument — a weighted portfolio of sub-instruments.
//!
//! A `CompositeInstrument` holds a list of (weight, NPV) entries and
//! computes the portfolio's aggregate NPV as the weighted sum.
//!
//! This is useful for replication strategies, hedging portfolios, and
//! synthetic instruments (e.g. a collar = long cap + short floor).

use serde::{Deserialize, Serialize};

/// A component in a composite instrument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeComponent {
    /// Human-readable label.
    pub label: String,
    /// Weight (positive = long, negative = short).
    pub weight: f64,
    /// NPV of this component (pre-computed by its own pricing engine).
    pub npv: f64,
}

/// A weighted portfolio of sub-instruments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeInstrument {
    /// Name of the composite.
    pub name: String,
    /// The components.
    pub components: Vec<CompositeComponent>,
}

impl CompositeInstrument {
    /// Create an empty composite instrument.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            components: Vec::new(),
        }
    }

    /// Add a long (weight = +1) component.
    pub fn add(&mut self, label: &str, npv: f64) {
        self.components.push(CompositeComponent {
            label: label.to_string(),
            weight: 1.0,
            npv,
        });
    }

    /// Subtract (weight = −1) a component.
    pub fn subtract(&mut self, label: &str, npv: f64) {
        self.components.push(CompositeComponent {
            label: label.to_string(),
            weight: -1.0,
            npv,
        });
    }

    /// Add a component with arbitrary weight.
    pub fn add_weighted(&mut self, label: &str, weight: f64, npv: f64) {
        self.components.push(CompositeComponent {
            label: label.to_string(),
            weight,
            npv,
        });
    }

    /// Aggregate NPV (weighted sum of component NPVs).
    pub fn npv(&self) -> f64 {
        self.components.iter().map(|c| c.weight * c.npv).sum()
    }

    /// Number of components.
    pub fn size(&self) -> usize {
        self.components.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_composite() {
        let c = CompositeInstrument::new("empty");
        assert_eq!(c.size(), 0);
        assert!((c.npv()).abs() < 1e-15);
    }

    #[test]
    fn long_short_components() {
        let mut collar = CompositeInstrument::new("collar");
        collar.add("long cap", 25_000.0);
        collar.subtract("short floor", 12_000.0);

        assert_eq!(collar.size(), 2);
        assert!((collar.npv() - 13_000.0).abs() < 1e-10);
    }

    #[test]
    fn weighted_components() {
        let mut basket = CompositeInstrument::new("basket");
        basket.add_weighted("AAPL", 0.4, 100.0);
        basket.add_weighted("MSFT", 0.35, 200.0);
        basket.add_weighted("GOOG", 0.25, 150.0);

        let expected = 0.4 * 100.0 + 0.35 * 200.0 + 0.25 * 150.0;
        assert!((basket.npv() - expected).abs() < 1e-10);
    }

    #[test]
    fn serde_roundtrip() {
        let mut c = CompositeInstrument::new("hedge");
        c.add("bond", 1_000_000.0);
        c.subtract("swap", 50_000.0);

        let json = serde_json::to_string(&c).unwrap();
        let c2: CompositeInstrument = serde_json::from_str(&json).unwrap();
        assert!((c.npv() - c2.npv()).abs() < 1e-10);
    }
}
