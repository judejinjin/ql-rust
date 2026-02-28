//! Currency definitions following ISO 4217.
//!
//! Each currency carries its ISO alpha code, numeric code, name, and rounding
//! convention. Currencies are value types that can be compared by code.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Rounding
// ---------------------------------------------------------------------------

/// Rounding convention for monetary amounts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Rounding {
    /// No rounding.
    None,
    /// Round to nearest, half up.
    Closest(u32),
    /// Always round up (ceiling).
    Up(u32),
    /// Always round down (floor).
    Down(u32),
}

impl Rounding {
    /// Apply this rounding convention to a value.
    pub fn round(&self, value: f64) -> f64 {
        match *self {
            Rounding::None => value,
            Rounding::Closest(precision) => {
                let mult = 10f64.powi(precision as i32);
                (value * mult).round() / mult
            }
            Rounding::Up(precision) => {
                let mult = 10f64.powi(precision as i32);
                (value * mult).ceil() / mult
            }
            Rounding::Down(precision) => {
                let mult = 10f64.powi(precision as i32);
                (value * mult).floor() / mult
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Currency
// ---------------------------------------------------------------------------

/// A currency identified by its ISO 4217 code.
///
/// Currencies are immutable value types. Two currencies are equal if and only
/// if they share the same ISO alpha code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Currency {
    /// ISO 4217 three-letter code (e.g., "USD").
    pub code: &'static str,
    /// Full name (e.g., "US dollar").
    pub name: &'static str,
    /// ISO 4217 numeric code (e.g., 840 for USD).
    pub numeric_code: u16,
    /// Symbol (e.g., "$", "€").
    pub symbol: &'static str,
    /// Fractional unit name (e.g., "cent").
    pub fraction_symbol: &'static str,
    /// Number of fractional units per whole unit (e.g., 100 cents per dollar).
    pub fractions_per_unit: u32,
    /// Rounding convention.
    pub rounding: Rounding,
}

impl PartialEq for Currency {
    fn eq(&self, other: &Self) -> bool {
        self.code == other.code
    }
}

impl Eq for Currency {}

impl std::hash::Hash for Currency {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.code.hash(state);
    }
}

impl std::fmt::Display for Currency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.code)
    }
}

// ---------------------------------------------------------------------------
// Concrete Currencies
// ---------------------------------------------------------------------------

impl Currency {
    /// US Dollar.
    pub fn usd() -> Self {
        Self {
            code: "USD",
            name: "US dollar",
            numeric_code: 840,
            symbol: "$",
            fraction_symbol: "¢",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Euro.
    pub fn eur() -> Self {
        Self {
            code: "EUR",
            name: "euro",
            numeric_code: 978,
            symbol: "€",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// British Pound Sterling.
    pub fn gbp() -> Self {
        Self {
            code: "GBP",
            name: "British pound sterling",
            numeric_code: 826,
            symbol: "£",
            fraction_symbol: "p",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Japanese Yen.
    pub fn jpy() -> Self {
        Self {
            code: "JPY",
            name: "Japanese yen",
            numeric_code: 392,
            symbol: "¥",
            fraction_symbol: "",
            fractions_per_unit: 1,
            rounding: Rounding::None,
        }
    }

    /// Swiss Franc.
    pub fn chf() -> Self {
        Self {
            code: "CHF",
            name: "Swiss franc",
            numeric_code: 756,
            symbol: "CHF",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Australian Dollar.
    pub fn aud() -> Self {
        Self {
            code: "AUD",
            name: "Australian dollar",
            numeric_code: 36,
            symbol: "A$",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Canadian Dollar.
    pub fn cad() -> Self {
        Self {
            code: "CAD",
            name: "Canadian dollar",
            numeric_code: 124,
            symbol: "C$",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// New Zealand dollar (NZD).
    pub fn nzd() -> Self {
        Self {
            code: "NZD",
            name: "New Zealand dollar",
            numeric_code: 554,
            symbol: "NZ$",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Swedish krona (SEK).
    pub fn sek() -> Self {
        Self {
            code: "SEK",
            name: "Swedish krona",
            numeric_code: 752,
            symbol: "kr",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Danish krone (DKK).
    pub fn dkk() -> Self {
        Self {
            code: "DKK",
            name: "Danish krone",
            numeric_code: 208,
            symbol: "kr",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Norwegian krone (NOK).
    pub fn nok() -> Self {
        Self {
            code: "NOK",
            name: "Norwegian krone",
            numeric_code: 578,
            symbol: "kr",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Brazilian real (BRL).
    pub fn brl() -> Self {
        Self {
            code: "BRL",
            name: "Brazilian real",
            numeric_code: 986,
            symbol: "R$",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Polish zloty (PLN).
    pub fn pln() -> Self {
        Self {
            code: "PLN",
            name: "Polish zloty",
            numeric_code: 985,
            symbol: "zł",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Czech koruna (CZK).
    pub fn czk() -> Self {
        Self {
            code: "CZK",
            name: "Czech koruna",
            numeric_code: 203,
            symbol: "Kč",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Hungarian forint (HUF).
    pub fn huf() -> Self {
        Self {
            code: "HUF",
            name: "Hungarian forint",
            numeric_code: 348,
            symbol: "Ft",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Romanian leu (RON).
    pub fn ron() -> Self {
        Self {
            code: "RON",
            name: "Romanian leu",
            numeric_code: 946,
            symbol: "lei",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Turkish lira (TRY).
    pub fn try_() -> Self {
        Self {
            code: "TRY",
            name: "Turkish lira",
            numeric_code: 949,
            symbol: "₺",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// South African rand (ZAR).
    pub fn zar() -> Self {
        Self {
            code: "ZAR",
            name: "South African rand",
            numeric_code: 710,
            symbol: "R",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Chinese renminbi (CNY).
    pub fn cny() -> Self {
        Self {
            code: "CNY",
            name: "Chinese renminbi",
            numeric_code: 156,
            symbol: "¥",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Indian rupee (INR).
    pub fn inr() -> Self {
        Self {
            code: "INR",
            name: "Indian rupee",
            numeric_code: 356,
            symbol: "₹",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// South Korean won (KRW).
    pub fn krw() -> Self {
        Self {
            code: "KRW",
            name: "South Korean won",
            numeric_code: 410,
            symbol: "₩",
            fraction_symbol: "",
            fractions_per_unit: 1,
            rounding: Rounding::Closest(0),
        }
    }

    /// Singapore dollar (SGD).
    pub fn sgd() -> Self {
        Self {
            code: "SGD",
            name: "Singapore dollar",
            numeric_code: 702,
            symbol: "S$",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Hong Kong dollar (HKD).
    pub fn hkd() -> Self {
        Self {
            code: "HKD",
            name: "Hong Kong dollar",
            numeric_code: 344,
            symbol: "HK$",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    // -----------------------------------------------------------------------
    // Americas
    // -----------------------------------------------------------------------

    /// Mexican peso (MXN).
    pub fn mxn() -> Self {
        Self {
            code: "MXN",
            name: "Mexican peso",
            numeric_code: 484,
            symbol: "Mex$",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Argentine peso (ARS).
    pub fn ars() -> Self {
        Self {
            code: "ARS",
            name: "Argentine peso",
            numeric_code: 32,
            symbol: "$",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Chilean peso (CLP).
    pub fn clp() -> Self {
        Self {
            code: "CLP",
            name: "Chilean peso",
            numeric_code: 152,
            symbol: "$",
            fraction_symbol: "",
            fractions_per_unit: 1,
            rounding: Rounding::Closest(0),
        }
    }

    /// Colombian peso (COP).
    pub fn cop() -> Self {
        Self {
            code: "COP",
            name: "Colombian peso",
            numeric_code: 170,
            symbol: "$",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Peruvian sol (PEN).
    pub fn pen() -> Self {
        Self {
            code: "PEN",
            name: "Peruvian sol",
            numeric_code: 604,
            symbol: "S/",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Uruguayan peso (UYU).
    pub fn uyu() -> Self {
        Self {
            code: "UYU",
            name: "Uruguayan peso",
            numeric_code: 858,
            symbol: "$U",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Trinidad & Tobago dollar (TTD).
    pub fn ttd() -> Self {
        Self {
            code: "TTD",
            name: "Trinidad & Tobago dollar",
            numeric_code: 780,
            symbol: "TT$",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    // -----------------------------------------------------------------------
    // Europe (additional)
    // -----------------------------------------------------------------------

    /// Icelandic króna (ISK).
    pub fn isk() -> Self {
        Self {
            code: "ISK",
            name: "Icelandic króna",
            numeric_code: 352,
            symbol: "kr",
            fraction_symbol: "",
            fractions_per_unit: 1,
            rounding: Rounding::Closest(0),
        }
    }

    /// Bulgarian lev (BGN).
    pub fn bgn() -> Self {
        Self {
            code: "BGN",
            name: "Bulgarian lev",
            numeric_code: 975,
            symbol: "лв",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Croatian kuna (HRK) — legacy, replaced by EUR 2023.
    pub fn hrk() -> Self {
        Self {
            code: "HRK",
            name: "Croatian kuna",
            numeric_code: 191,
            symbol: "kn",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Serbian dinar (RSD).
    pub fn rsd() -> Self {
        Self {
            code: "RSD",
            name: "Serbian dinar",
            numeric_code: 941,
            symbol: "din.",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Ukrainian hryvnia (UAH).
    pub fn uah() -> Self {
        Self {
            code: "UAH",
            name: "Ukrainian hryvnia",
            numeric_code: 980,
            symbol: "₴",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Russian ruble (RUB).
    pub fn rub() -> Self {
        Self {
            code: "RUB",
            name: "Russian ruble",
            numeric_code: 643,
            symbol: "₽",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Georgian lari (GEL).
    pub fn gel() -> Self {
        Self {
            code: "GEL",
            name: "Georgian lari",
            numeric_code: 981,
            symbol: "₾",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    // -----------------------------------------------------------------------
    // Asia / Middle East (additional)
    // -----------------------------------------------------------------------

    /// New Taiwan dollar (TWD).
    pub fn twd() -> Self {
        Self {
            code: "TWD",
            name: "New Taiwan dollar",
            numeric_code: 901,
            symbol: "NT$",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Thai baht (THB).
    pub fn thb() -> Self {
        Self {
            code: "THB",
            name: "Thai baht",
            numeric_code: 764,
            symbol: "฿",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Malaysian ringgit (MYR).
    pub fn myr() -> Self {
        Self {
            code: "MYR",
            name: "Malaysian ringgit",
            numeric_code: 458,
            symbol: "RM",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Indonesian rupiah (IDR).
    pub fn idr() -> Self {
        Self {
            code: "IDR",
            name: "Indonesian rupiah",
            numeric_code: 360,
            symbol: "Rp",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Philippine peso (PHP).
    pub fn php() -> Self {
        Self {
            code: "PHP",
            name: "Philippine peso",
            numeric_code: 608,
            symbol: "₱",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Pakistani rupee (PKR).
    pub fn pkr() -> Self {
        Self {
            code: "PKR",
            name: "Pakistani rupee",
            numeric_code: 586,
            symbol: "₨",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Bangladeshi taka (BDT).
    pub fn bdt() -> Self {
        Self {
            code: "BDT",
            name: "Bangladeshi taka",
            numeric_code: 50,
            symbol: "৳",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Sri Lankan rupee (LKR).
    pub fn lkr() -> Self {
        Self {
            code: "LKR",
            name: "Sri Lankan rupee",
            numeric_code: 144,
            symbol: "Rs",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Vietnamese dong (VND).
    pub fn vnd() -> Self {
        Self {
            code: "VND",
            name: "Vietnamese dong",
            numeric_code: 704,
            symbol: "₫",
            fraction_symbol: "",
            fractions_per_unit: 1,
            rounding: Rounding::Closest(0),
        }
    }

    /// Israeli new shekel (ILS).
    pub fn ils() -> Self {
        Self {
            code: "ILS",
            name: "Israeli new shekel",
            numeric_code: 376,
            symbol: "₪",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Saudi riyal (SAR).
    pub fn sar() -> Self {
        Self {
            code: "SAR",
            name: "Saudi riyal",
            numeric_code: 682,
            symbol: "﷼",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// UAE dirham (AED).
    pub fn aed() -> Self {
        Self {
            code: "AED",
            name: "UAE dirham",
            numeric_code: 784,
            symbol: "د.إ",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Qatari riyal (QAR).
    pub fn qar() -> Self {
        Self {
            code: "QAR",
            name: "Qatari riyal",
            numeric_code: 634,
            symbol: "﷼",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Bahraini dinar (BHD).
    pub fn bhd() -> Self {
        Self {
            code: "BHD",
            name: "Bahraini dinar",
            numeric_code: 48,
            symbol: "BD",
            fraction_symbol: "",
            fractions_per_unit: 1000,
            rounding: Rounding::Closest(3),
        }
    }

    /// Kuwaiti dinar (KWD).
    pub fn kwd() -> Self {
        Self {
            code: "KWD",
            name: "Kuwaiti dinar",
            numeric_code: 414,
            symbol: "د.ك",
            fraction_symbol: "",
            fractions_per_unit: 1000,
            rounding: Rounding::Closest(3),
        }
    }

    /// Omani rial (OMR).
    pub fn omr() -> Self {
        Self {
            code: "OMR",
            name: "Omani rial",
            numeric_code: 512,
            symbol: "﷼",
            fraction_symbol: "",
            fractions_per_unit: 1000,
            rounding: Rounding::Closest(3),
        }
    }

    /// Jordanian dinar (JOD).
    pub fn jod() -> Self {
        Self {
            code: "JOD",
            name: "Jordanian dinar",
            numeric_code: 400,
            symbol: "د.ا",
            fraction_symbol: "",
            fractions_per_unit: 1000,
            rounding: Rounding::Closest(3),
        }
    }

    // -----------------------------------------------------------------------
    // Africa (additional)
    // -----------------------------------------------------------------------

    /// Nigerian naira (NGN).
    pub fn ngn() -> Self {
        Self {
            code: "NGN",
            name: "Nigerian naira",
            numeric_code: 566,
            symbol: "₦",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Moroccan dirham (MAD).
    pub fn mad() -> Self {
        Self {
            code: "MAD",
            name: "Moroccan dirham",
            numeric_code: 504,
            symbol: "MAD",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Egyptian pound (EGP).
    pub fn egp() -> Self {
        Self {
            code: "EGP",
            name: "Egyptian pound",
            numeric_code: 818,
            symbol: "E£",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Kenyan shilling (KES).
    pub fn kes() -> Self {
        Self {
            code: "KES",
            name: "Kenyan shilling",
            numeric_code: 404,
            symbol: "KSh",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Ghanaian cedi (GHS).
    pub fn ghs() -> Self {
        Self {
            code: "GHS",
            name: "Ghanaian cedi",
            numeric_code: 936,
            symbol: "GH₵",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Tunisian dinar (TND).
    pub fn tnd() -> Self {
        Self {
            code: "TND",
            name: "Tunisian dinar",
            numeric_code: 788,
            symbol: "DT",
            fraction_symbol: "",
            fractions_per_unit: 1000,
            rounding: Rounding::Closest(3),
        }
    }

    /// Botswana pula (BWP).
    pub fn bwp() -> Self {
        Self {
            code: "BWP",
            name: "Botswana pula",
            numeric_code: 72,
            symbol: "P",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    // -----------------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------------

    /// Look up a currency by its ISO 4217 alpha code.
    ///
    /// Returns `None` if the code is not recognized.
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            // G10
            "USD" => Some(Self::usd()),
            "EUR" => Some(Self::eur()),
            "GBP" => Some(Self::gbp()),
            "JPY" => Some(Self::jpy()),
            "CHF" => Some(Self::chf()),
            "AUD" => Some(Self::aud()),
            "CAD" => Some(Self::cad()),
            "NZD" => Some(Self::nzd()),
            "SEK" => Some(Self::sek()),
            "DKK" => Some(Self::dkk()),
            "NOK" => Some(Self::nok()),
            // Emerging
            "BRL" => Some(Self::brl()),
            "PLN" => Some(Self::pln()),
            "CZK" => Some(Self::czk()),
            "HUF" => Some(Self::huf()),
            "RON" => Some(Self::ron()),
            "TRY" => Some(Self::try_()),
            "ZAR" => Some(Self::zar()),
            "CNY" => Some(Self::cny()),
            "INR" => Some(Self::inr()),
            "KRW" => Some(Self::krw()),
            "SGD" => Some(Self::sgd()),
            "HKD" => Some(Self::hkd()),
            // Americas
            "MXN" => Some(Self::mxn()),
            "ARS" => Some(Self::ars()),
            "CLP" => Some(Self::clp()),
            "COP" => Some(Self::cop()),
            "PEN" => Some(Self::pen()),
            "UYU" => Some(Self::uyu()),
            "TTD" => Some(Self::ttd()),
            // Europe additional
            "ISK" => Some(Self::isk()),
            "BGN" => Some(Self::bgn()),
            "HRK" => Some(Self::hrk()),
            "RSD" => Some(Self::rsd()),
            "UAH" => Some(Self::uah()),
            "RUB" => Some(Self::rub()),
            "GEL" => Some(Self::gel()),
            // Asia / Middle East
            "TWD" => Some(Self::twd()),
            "THB" => Some(Self::thb()),
            "MYR" => Some(Self::myr()),
            "IDR" => Some(Self::idr()),
            "PHP" => Some(Self::php()),
            "PKR" => Some(Self::pkr()),
            "BDT" => Some(Self::bdt()),
            "LKR" => Some(Self::lkr()),
            "VND" => Some(Self::vnd()),
            "ILS" => Some(Self::ils()),
            "SAR" => Some(Self::sar()),
            "AED" => Some(Self::aed()),
            "QAR" => Some(Self::qar()),
            "BHD" => Some(Self::bhd()),
            "KWD" => Some(Self::kwd()),
            "OMR" => Some(Self::omr()),
            "JOD" => Some(Self::jod()),
            // Africa
            "NGN" => Some(Self::ngn()),
            "MAD" => Some(Self::mad()),
            "EGP" => Some(Self::egp()),
            "KES" => Some(Self::kes()),
            "GHS" => Some(Self::ghs()),
            "TND" => Some(Self::tnd()),
            "BWP" => Some(Self::bwp()),
            _ => None,
        }
    }

    /// Apply the currency's rounding convention to a monetary amount.
    pub fn round(&self, amount: f64) -> f64 {
        self.rounding.round(amount)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usd_properties() {
        let usd = Currency::usd();
        assert_eq!(usd.code, "USD");
        assert_eq!(usd.numeric_code, 840);
        assert_eq!(usd.fractions_per_unit, 100);
        assert_eq!(usd.to_string(), "USD");
    }

    #[test]
    fn eur_properties() {
        let eur = Currency::eur();
        assert_eq!(eur.code, "EUR");
        assert_eq!(eur.numeric_code, 978);
    }

    #[test]
    fn jpy_no_fractions() {
        let jpy = Currency::jpy();
        assert_eq!(jpy.fractions_per_unit, 1);
    }

    #[test]
    fn currency_equality() {
        assert_eq!(Currency::usd(), Currency::usd());
        assert_ne!(Currency::usd(), Currency::eur());
    }

    #[test]
    fn rounding_closest() {
        let r = Rounding::Closest(2);
        assert!((r.round(1.2345) - 1.23).abs() < 1e-15);
        assert!((r.round(1.235) - 1.24).abs() < 1e-15);
    }

    #[test]
    fn rounding_up() {
        let r = Rounding::Up(2);
        assert!((r.round(1.231) - 1.24).abs() < 1e-15);
    }

    #[test]
    fn rounding_down() {
        let r = Rounding::Down(2);
        assert!((r.round(1.239) - 1.23).abs() < 1e-15);
    }

    #[test]
    fn rounding_none() {
        let r = Rounding::None;
        let val = 1.23456789;
        assert!((r.round(val) - val).abs() < 1e-15);
    }

    #[test]
    fn currency_round_amount() {
        let usd = Currency::usd();
        assert!((usd.round(100.999) - 101.0).abs() < 1e-15);
        assert!((usd.round(100.001) - 100.0).abs() < 1e-15);
    }

    #[test]
    fn from_code_g10() {
        for code in ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "SEK", "DKK", "NOK"] {
            let c = Currency::from_code(code).unwrap();
            assert_eq!(c.code, code);
        }
    }

    #[test]
    fn from_code_unknown_returns_none() {
        assert!(Currency::from_code("XYZ").is_none());
    }

    #[test]
    fn three_digit_dinar_currencies() {
        // BHD, KWD, OMR, JOD, TND all use 3-digit minor units
        for ccy_fn in [Currency::bhd, Currency::kwd, Currency::omr, Currency::jod, Currency::tnd] {
            let c = ccy_fn();
            assert_eq!(c.fractions_per_unit, 1000, "{} should have 1000 fractions", c.code);
        }
    }

    #[test]
    fn zero_fraction_currencies() {
        // JPY, KRW, CLP, ISK, VND have no fractional units
        for ccy_fn in [Currency::jpy, Currency::krw, Currency::clp, Currency::isk, Currency::vnd] {
            let c = ccy_fn();
            assert_eq!(c.fractions_per_unit, 1, "{} should have fractions=1", c.code);
        }
    }

    #[test]
    fn all_currencies_have_unique_codes() {
        let all: Vec<Currency> = [
            Currency::usd(), Currency::eur(), Currency::gbp(), Currency::jpy(),
            Currency::chf(), Currency::aud(), Currency::cad(), Currency::nzd(),
            Currency::sek(), Currency::dkk(), Currency::nok(), Currency::brl(),
            Currency::pln(), Currency::czk(), Currency::huf(), Currency::ron(),
            Currency::try_(), Currency::zar(), Currency::cny(), Currency::inr(),
            Currency::krw(), Currency::sgd(), Currency::hkd(),
            // New
            Currency::mxn(), Currency::ars(), Currency::clp(), Currency::cop(),
            Currency::pen(), Currency::uyu(), Currency::ttd(),
            Currency::isk(), Currency::bgn(), Currency::hrk(), Currency::rsd(),
            Currency::uah(), Currency::rub(), Currency::gel(),
            Currency::twd(), Currency::thb(), Currency::myr(), Currency::idr(),
            Currency::php(), Currency::pkr(), Currency::bdt(), Currency::lkr(),
            Currency::vnd(), Currency::ils(), Currency::sar(), Currency::aed(),
            Currency::qar(), Currency::bhd(), Currency::kwd(), Currency::omr(),
            Currency::jod(),
            Currency::ngn(), Currency::mad(), Currency::egp(), Currency::kes(),
            Currency::ghs(), Currency::tnd(), Currency::bwp(),
        ].to_vec();
        let mut codes: Vec<&str> = all.iter().map(|c| c.code).collect();
        let n = codes.len();
        codes.sort();
        codes.dedup();
        assert_eq!(codes.len(), n, "duplicate codes detected");
    }

    #[test]
    fn from_code_roundtrip() {
        for code in ["MXN", "ARS", "ISK", "RUB", "TWD", "THB", "ILS", "SAR",
                      "AED", "BHD", "KWD", "NGN", "EGP", "BWP"] {
            let c = Currency::from_code(code).unwrap();
            assert_eq!(c.code, code);
        }
    }
}
