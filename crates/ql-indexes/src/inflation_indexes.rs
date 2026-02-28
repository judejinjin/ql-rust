//! Named inflation indexes.
//!
//! Provides concrete, named inflation index types for the most commonly
//! referenced inflation indexes worldwide. Each type specifies the
//! publication frequency, interpolation convention, and observation lag.
//!
//! | Index | Region | Frequency | Typical Lag |
//! |-------|--------|-----------|-------------|
//! | `USCPI` | US | Monthly | 3 months |
//! | `UKRPI` | UK | Monthly | 2 months |
//! | `EUHICP` | Eurozone | Monthly | 3 months |
//! | `AUCPI` | Australia | Quarterly | 2 quarters |
//! | `ZACPI` | South Africa | Monthly | 3 months |
//! | `FRHICP` | France | Monthly | 3 months |
//! | `JPCPI` | Japan | Monthly | 3 months |
//!
//! Reference:
//! - QuantLib: USCPI, UKRPI, EUHICP, AUCPI, ZACPII in inflationindex.hpp

use serde::{Deserialize, Serialize};
use ql_time::Date;

/// Interpolation type for inflation indexes.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum InflationInterpolation {
    /// No interpolation (use the value for the reference month).
    Flat,
    /// Linear interpolation between months.
    Linear,
}

/// Frequency of inflation index publication.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum InflationFrequency {
    /// Published monthly.
    Monthly,
    /// Published quarterly.
    Quarterly,
}

/// A named inflation index with its conventions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NamedInflationIndex {
    /// Short name (e.g. "USCPI", "UKRPI").
    pub name: String,
    /// Full name.
    pub full_name: String,
    /// ISO region/country code.
    pub region: String,
    /// ISO currency code.
    pub currency: String,
    /// Publication frequency.
    pub frequency: InflationFrequency,
    /// Observation lag in months.
    pub observation_lag_months: u32,
    /// Whether this is a revised index (subject to revisions).
    pub revised: bool,
    /// Default interpolation convention.
    pub interpolation: InflationInterpolation,
    /// Base date (start of available data).
    pub availability_start: Option<Date>,
}

/// US CPI (Consumer Price Index for All Urban Consumers, NSA).
pub fn uscpi() -> NamedInflationIndex {
    NamedInflationIndex {
        name: "USCPI".to_string(),
        full_name: "US Consumer Price Index — All Urban Consumers (CPI-U), Not Seasonally Adjusted".to_string(),
        region: "US".to_string(),
        currency: "USD".to_string(),
        frequency: InflationFrequency::Monthly,
        observation_lag_months: 3,
        revised: false,
        interpolation: InflationInterpolation::Linear,
        availability_start: None,
    }
}

/// UK RPI (Retail Price Index).
pub fn ukrpi() -> NamedInflationIndex {
    NamedInflationIndex {
        name: "UKRPI".to_string(),
        full_name: "UK Retail Price Index".to_string(),
        region: "GB".to_string(),
        currency: "GBP".to_string(),
        frequency: InflationFrequency::Monthly,
        observation_lag_months: 2,
        revised: false,
        interpolation: InflationInterpolation::Linear,
        availability_start: None,
    }
}

/// Eurozone HICP (Harmonised Index of Consumer Prices, ex-tobacco).
pub fn euhicp() -> NamedInflationIndex {
    NamedInflationIndex {
        name: "EUHICP".to_string(),
        full_name: "Euro Area HICP (Harmonised Index of Consumer Prices), Ex-Tobacco".to_string(),
        region: "EU".to_string(),
        currency: "EUR".to_string(),
        frequency: InflationFrequency::Monthly,
        observation_lag_months: 3,
        revised: true,
        interpolation: InflationInterpolation::Linear,
        availability_start: None,
    }
}

/// Australian CPI (Consumer Price Index, All Groups).
pub fn aucpi() -> NamedInflationIndex {
    NamedInflationIndex {
        name: "AUCPI".to_string(),
        full_name: "Australian Consumer Price Index — All Groups".to_string(),
        region: "AU".to_string(),
        currency: "AUD".to_string(),
        frequency: InflationFrequency::Quarterly,
        observation_lag_months: 6, // ~2 quarters
        revised: true,
        interpolation: InflationInterpolation::Flat,
        availability_start: None,
    }
}

/// South African CPI.
pub fn zacpi() -> NamedInflationIndex {
    NamedInflationIndex {
        name: "ZACPI".to_string(),
        full_name: "South Africa Consumer Price Index".to_string(),
        region: "ZA".to_string(),
        currency: "ZAR".to_string(),
        frequency: InflationFrequency::Monthly,
        observation_lag_months: 3,
        revised: false,
        interpolation: InflationInterpolation::Linear,
        availability_start: None,
    }
}

/// French HICP.
pub fn frhicp() -> NamedInflationIndex {
    NamedInflationIndex {
        name: "FRHICP".to_string(),
        full_name: "French HICP (Harmonised Index of Consumer Prices), Ex-Tobacco".to_string(),
        region: "FR".to_string(),
        currency: "EUR".to_string(),
        frequency: InflationFrequency::Monthly,
        observation_lag_months: 3,
        revised: true,
        interpolation: InflationInterpolation::Linear,
        availability_start: None,
    }
}

/// Japanese CPI (Consumer Price Index, All Items).
pub fn jpcpi() -> NamedInflationIndex {
    NamedInflationIndex {
        name: "JPCPI".to_string(),
        full_name: "Japan Consumer Price Index — All Items".to_string(),
        region: "JP".to_string(),
        currency: "JPY".to_string(),
        frequency: InflationFrequency::Monthly,
        observation_lag_months: 3,
        revised: true,
        interpolation: InflationInterpolation::Linear,
        availability_start: None,
    }
}

/// Look up a named inflation index by its short name (case-insensitive).
pub fn inflation_index_by_name(name: &str) -> Option<NamedInflationIndex> {
    match name.to_uppercase().as_str() {
        "USCPI" | "US CPI" | "CPI-U" => Some(uscpi()),
        "UKRPI" | "UK RPI" | "RPI" => Some(ukrpi()),
        "EUHICP" | "HICP" | "EU HICP" => Some(euhicp()),
        "AUCPI" | "AU CPI" => Some(aucpi()),
        "ZACPI" | "ZA CPI" => Some(zacpi()),
        "FRHICP" | "FR HICP" => Some(frhicp()),
        "JPCPI" | "JP CPI" => Some(jpcpi()),
        _ => None,
    }
}

/// Return all available named inflation indexes.
pub fn all_inflation_indexes() -> Vec<NamedInflationIndex> {
    vec![uscpi(), ukrpi(), euhicp(), aucpi(), zacpi(), frhicp(), jpcpi()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uscpi() {
        let idx = uscpi();
        assert_eq!(idx.name, "USCPI");
        assert_eq!(idx.region, "US");
        assert_eq!(idx.currency, "USD");
        assert_eq!(idx.observation_lag_months, 3);
        assert_eq!(idx.frequency, InflationFrequency::Monthly);
    }

    #[test]
    fn test_ukrpi() {
        let idx = ukrpi();
        assert_eq!(idx.name, "UKRPI");
        assert_eq!(idx.observation_lag_months, 2);
    }

    #[test]
    fn test_euhicp() {
        let idx = euhicp();
        assert_eq!(idx.name, "EUHICP");
        assert!(idx.revised);
    }

    #[test]
    fn test_aucpi_quarterly() {
        let idx = aucpi();
        assert_eq!(idx.frequency, InflationFrequency::Quarterly);
        assert_eq!(idx.observation_lag_months, 6);
    }

    #[test]
    fn test_lookup_by_name() {
        assert!(inflation_index_by_name("USCPI").is_some());
        assert!(inflation_index_by_name("us cpi").is_some());
        assert!(inflation_index_by_name("UKRPI").is_some());
        assert!(inflation_index_by_name("EUHICP").is_some());
        assert!(inflation_index_by_name("FOOBAR").is_none());
    }

    #[test]
    fn test_all_indexes() {
        let all = all_inflation_indexes();
        assert_eq!(all.len(), 7);
        let names: Vec<&str> = all.iter().map(|i| i.name.as_str()).collect();
        assert!(names.contains(&"USCPI"));
        assert!(names.contains(&"UKRPI"));
        assert!(names.contains(&"EUHICP"));
        assert!(names.contains(&"AUCPI"));
        assert!(names.contains(&"ZACPI"));
    }
}
