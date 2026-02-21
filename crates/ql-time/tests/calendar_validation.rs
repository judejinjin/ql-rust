//! Calendar holiday validation tests.
//!
//! These tests systematically validate holiday functions for each calendar
//! against a curated set of **known-correct** holiday dates spanning multiple
//! years (2020-2026). Sources: official central-bank / exchange holiday
//! schedules, ISO 8601 weekend rules, and published Easter tables.
//!
//! Any regression in the holiday logic will surface as a `is_business_day`
//! assertion failure in these tests.

use ql_time::{Calendar, Date, Month};
use ql_time::calendar::USMarket;

// ============================================================================
//  Helper
// ============================================================================

fn assert_holiday(cal: &Calendar, year: i32, month: Month, day: u32, label: &str) {
    let d = Date::from_ymd(year, month, day);
    assert!(
        !cal.is_business_day(d),
        "{label}: {year}-{month:?}-{day:02} should be a holiday but was reported as business day",
    );
}

fn assert_business_day(cal: &Calendar, year: i32, month: Month, day: u32, label: &str) {
    let d = Date::from_ymd(year, month, day);
    assert!(
        cal.is_business_day(d),
        "{label}: {year}-{month:?}-{day:02} should be a business day but was reported as holiday",
    );
}

// ============================================================================
//  Easter dates (authoritative table)
// ============================================================================
//
//  Year | Easter Sunday | Good Friday | Easter Monday
//  2020 | Apr 12        | Apr 10      | Apr 13
//  2021 | Apr  4        | Apr  2      | Apr  5
//  2022 | Apr 17        | Apr 15      | Apr 18
//  2023 | Apr  9        | Apr  7      | Apr 10
//  2024 | Mar 31        | Mar 29      | Apr  1
//  2025 | Apr 20        | Apr 18      | Apr 21
//  2026 | Apr  5        | Apr  3      | Apr  6

// ============================================================================
//  TARGET calendar (ECB)
// ============================================================================

#[test]
fn target_holidays_multiyear() {
    let cal = Calendar::Target;

    // New Year's Day
    for yr in 2020..=2026 {
        assert_holiday(&cal, yr, Month::January, 1, "TARGET New Year");
    }

    // Labour Day (May 1) - TARGET holiday from 2000 onward
    for yr in 2020..=2026 {
        assert_holiday(&cal, yr, Month::May, 1, "TARGET Labour Day");
    }

    // Christmas / St. Stephen's
    for yr in 2020..=2026 {
        assert_holiday(&cal, yr, Month::December, 25, "TARGET Christmas");
        assert_holiday(&cal, yr, Month::December, 26, "TARGET St. Stephen");
    }

    // Good Friday (Easter-based)
    assert_holiday(&cal, 2020, Month::April, 10, "TARGET Good Friday 2020");
    assert_holiday(&cal, 2021, Month::April, 2, "TARGET Good Friday 2021");
    assert_holiday(&cal, 2022, Month::April, 15, "TARGET Good Friday 2022");
    assert_holiday(&cal, 2023, Month::April, 7, "TARGET Good Friday 2023");
    assert_holiday(&cal, 2024, Month::March, 29, "TARGET Good Friday 2024");
    assert_holiday(&cal, 2025, Month::April, 18, "TARGET Good Friday 2025");
    assert_holiday(&cal, 2026, Month::April, 3, "TARGET Good Friday 2026");

    // Easter Monday
    assert_holiday(&cal, 2020, Month::April, 13, "TARGET Easter Mon 2020");
    assert_holiday(&cal, 2021, Month::April, 5, "TARGET Easter Mon 2021");
    assert_holiday(&cal, 2022, Month::April, 18, "TARGET Easter Mon 2022");
    assert_holiday(&cal, 2023, Month::April, 10, "TARGET Easter Mon 2023");
    assert_holiday(&cal, 2024, Month::April, 1, "TARGET Easter Mon 2024");
    assert_holiday(&cal, 2025, Month::April, 21, "TARGET Easter Mon 2025");
    assert_holiday(&cal, 2026, Month::April, 6, "TARGET Easter Mon 2026");

    // A handful of normal business days
    assert_business_day(&cal, 2025, Month::June, 16, "TARGET Mon 2025-06-16");
    assert_business_day(&cal, 2025, Month::March, 10, "TARGET Mon 2025-03-10");
}

// ============================================================================
//  United States — Settlement
// ============================================================================

#[test]
fn us_settlement_holidays_multiyear() {
    let cal = Calendar::UnitedStates(USMarket::Settlement);

    // New Year's Day or observed
    assert_holiday(&cal, 2025, Month::January, 1, "US New Year 2025");
    assert_holiday(&cal, 2023, Month::January, 2, "US New Year observed 2023 (Jan 1 is Sun→Mon)");

    // MLK Day: 3rd Monday of January
    assert_holiday(&cal, 2025, Month::January, 20, "US MLK 2025");
    assert_holiday(&cal, 2024, Month::January, 15, "US MLK 2024");
    assert_holiday(&cal, 2023, Month::January, 16, "US MLK 2023");
    assert_holiday(&cal, 2022, Month::January, 17, "US MLK 2022");

    // Presidents' Day: 3rd Monday of February
    assert_holiday(&cal, 2025, Month::February, 17, "US Presidents Day 2025");
    assert_holiday(&cal, 2024, Month::February, 19, "US Presidents Day 2024");

    // Memorial Day: last Monday of May
    assert_holiday(&cal, 2025, Month::May, 26, "US Memorial Day 2025");
    assert_holiday(&cal, 2024, Month::May, 27, "US Memorial Day 2024");
    assert_holiday(&cal, 2023, Month::May, 29, "US Memorial Day 2023");

    // Juneteenth (June 19, observed, ≥ 2022)
    assert_holiday(&cal, 2025, Month::June, 19, "US Juneteenth 2025");
    assert_holiday(&cal, 2024, Month::June, 19, "US Juneteenth 2024");
    assert_holiday(&cal, 2023, Month::June, 19, "US Juneteenth 2023");
    assert_holiday(&cal, 2022, Month::June, 20, "US Juneteenth observed 2022 (Sun→Mon)");

    // Independence Day (July 4, observed)
    assert_holiday(&cal, 2025, Month::July, 4, "US July 4th 2025");
    assert_holiday(&cal, 2026, Month::July, 3, "US July 4th observed 2026 (Sat→Fri)");
    assert_holiday(&cal, 2021, Month::July, 5, "US July 4th observed 2021 (Sun→Mon)");

    // Labor Day: 1st Monday of September
    assert_holiday(&cal, 2025, Month::September, 1, "US Labor Day 2025");
    assert_holiday(&cal, 2024, Month::September, 2, "US Labor Day 2024");

    // Columbus Day: 2nd Monday of October (Settlement calendar)
    assert_holiday(&cal, 2025, Month::October, 13, "US Columbus Day 2025");
    assert_holiday(&cal, 2024, Month::October, 14, "US Columbus Day 2024");

    // Veterans Day (Nov 11, observed)
    assert_holiday(&cal, 2025, Month::November, 11, "US Veterans Day 2025");

    // Thanksgiving: 4th Thursday of November
    assert_holiday(&cal, 2025, Month::November, 27, "US Thanksgiving 2025");
    assert_holiday(&cal, 2024, Month::November, 28, "US Thanksgiving 2024");
    assert_holiday(&cal, 2023, Month::November, 23, "US Thanksgiving 2023");

    // Christmas (Dec 25, observed)
    assert_holiday(&cal, 2025, Month::December, 25, "US Christmas 2025");
    assert_holiday(&cal, 2021, Month::December, 24, "US Christmas observed 2021 (Sat→Fri)");
    assert_holiday(&cal, 2022, Month::December, 26, "US Christmas observed 2022 (Sun→Mon)");

    // Spot-check: regular business day
    assert_business_day(&cal, 2025, Month::March, 17, "US Mon 2025-03-17");
}

// ============================================================================
//  United States — NYSE
// ============================================================================

#[test]
fn us_nyse_holidays() {
    let cal = Calendar::UnitedStates(USMarket::NYSE);

    // NYSE does NOT observe Columbus Day or Veterans Day
    assert_business_day(&cal, 2025, Month::October, 13, "NYSE no Columbus Day");
    assert_business_day(&cal, 2025, Month::November, 11, "NYSE no Veterans Day");

    // But does observe Good Friday
    assert_holiday(&cal, 2025, Month::April, 18, "NYSE Good Friday 2025");
    assert_holiday(&cal, 2024, Month::March, 29, "NYSE Good Friday 2024");

    // Standard holidays
    assert_holiday(&cal, 2025, Month::January, 20, "NYSE MLK 2025");
    assert_holiday(&cal, 2025, Month::November, 27, "NYSE Thanksgiving 2025");
}

// ============================================================================
//  United Kingdom
// ============================================================================

#[test]
fn uk_holidays_multiyear() {
    let cal = Calendar::UnitedKingdom;

    // New Year observed: 2023 Jan 1 is Sunday → observed Jan 2
    assert_holiday(&cal, 2023, Month::January, 2, "UK New Year observed 2023");
    assert_holiday(&cal, 2025, Month::January, 1, "UK New Year 2025");

    // Good Friday / Easter Monday
    assert_holiday(&cal, 2025, Month::April, 18, "UK Good Friday 2025");
    assert_holiday(&cal, 2025, Month::April, 21, "UK Easter Monday 2025");
    assert_holiday(&cal, 2024, Month::March, 29, "UK Good Friday 2024");
    assert_holiday(&cal, 2024, Month::April, 1, "UK Easter Monday 2024");

    // Early May Bank Holiday (1st Monday of May)
    assert_holiday(&cal, 2025, Month::May, 5, "UK Early May 2025");
    assert_holiday(&cal, 2024, Month::May, 6, "UK Early May 2024");

    // Spring Bank Holiday (last Monday of May)
    assert_holiday(&cal, 2025, Month::May, 26, "UK Spring BH 2025");
    assert_holiday(&cal, 2024, Month::May, 27, "UK Spring BH 2024");

    // Summer Bank Holiday (last Monday of August)
    assert_holiday(&cal, 2025, Month::August, 25, "UK Summer BH 2025");
    assert_holiday(&cal, 2024, Month::August, 26, "UK Summer BH 2024");

    // Christmas / Boxing Day
    assert_holiday(&cal, 2025, Month::December, 25, "UK Christmas 2025");
    assert_holiday(&cal, 2025, Month::December, 26, "UK Boxing Day 2025");
}

// ============================================================================
//  Canada (TSX)
// ============================================================================

#[test]
fn canada_holidays_multiyear() {
    let cal = Calendar::Canada;

    // New Year (observed)
    assert_holiday(&cal, 2025, Month::January, 1, "CA New Year 2025");

    // Family Day: 3rd Monday of February (≥ 2008)
    assert_holiday(&cal, 2025, Month::February, 17, "CA Family Day 2025");
    assert_holiday(&cal, 2024, Month::February, 19, "CA Family Day 2024");

    // Good Friday
    assert_holiday(&cal, 2025, Month::April, 18, "CA Good Friday 2025");

    // Victoria Day: Monday before May 25
    assert_holiday(&cal, 2025, Month::May, 19, "CA Victoria Day 2025");
    assert_holiday(&cal, 2024, Month::May, 20, "CA Victoria Day 2024");

    // Canada Day (July 1, observed)
    assert_holiday(&cal, 2025, Month::July, 1, "CA Canada Day 2025");

    // Labour Day: 1st Monday of September
    assert_holiday(&cal, 2025, Month::September, 1, "CA Labour Day 2025");

    // Truth & Reconciliation (Sep 30, ≥ 2021)
    assert_holiday(&cal, 2025, Month::September, 30, "CA Truth & Rec 2025");

    // Thanksgiving: 2nd Monday of October
    assert_holiday(&cal, 2025, Month::October, 13, "CA Thanksgiving 2025");

    // Christmas
    assert_holiday(&cal, 2025, Month::December, 25, "CA Christmas 2025");
    assert_holiday(&cal, 2025, Month::December, 26, "CA Boxing Day 2025");
}

// ============================================================================
//  Australia (ASX)
// ============================================================================

#[test]
fn australia_holidays_multiyear() {
    let cal = Calendar::Australia;

    // New Year (observed)
    assert_holiday(&cal, 2025, Month::January, 1, "AU New Year 2025");

    // Australia Day (Jan 26, observed)
    assert_holiday(&cal, 2025, Month::January, 27, "AU Australia Day observed 2025 (Sun→Mon)");
    assert_holiday(&cal, 2024, Month::January, 26, "AU Australia Day 2024");

    // Good Friday + Easter Saturday + Easter Monday
    assert_holiday(&cal, 2025, Month::April, 18, "AU Good Friday 2025");
    assert_holiday(&cal, 2025, Month::April, 19, "AU Easter Saturday 2025");
    assert_holiday(&cal, 2025, Month::April, 21, "AU Easter Monday 2025");

    // ANZAC Day (Apr 25)
    assert_holiday(&cal, 2025, Month::April, 25, "AU ANZAC Day 2025");

    // Christmas / Boxing
    assert_holiday(&cal, 2025, Month::December, 25, "AU Christmas 2025");
    assert_holiday(&cal, 2025, Month::December, 26, "AU Boxing Day 2025");
}

// ============================================================================
//  Japan (JPX/TSE)
// ============================================================================

#[test]
fn japan_holidays_multiyear() {
    let cal = Calendar::Japan;

    // New Year triple (Jan 1-3)
    for yr in 2023..=2025 {
        assert_holiday(&cal, yr, Month::January, 1, &format!("JP New Year {yr}"));
        assert_holiday(&cal, yr, Month::January, 2, &format!("JP Bank Hol {yr}"));
        assert_holiday(&cal, yr, Month::January, 3, &format!("JP Bank Hol {yr}"));
    }

    // Coming of Age: 2nd Monday of January
    assert_holiday(&cal, 2025, Month::January, 13, "JP Coming of Age 2025");

    // National Foundation Day (Feb 11)
    assert_holiday(&cal, 2025, Month::February, 11, "JP Foundation Day 2025");

    // Emperor's Birthday (Feb 23, ≥ 2020)
    assert_holiday(&cal, 2025, Month::February, 24, "JP Emperor Bday observed 2025 (Sun→Mon)");
    assert_holiday(&cal, 2024, Month::February, 23, "JP Emperor Bday 2024");

    // Vernal Equinox (~Mar 20-21)
    assert_holiday(&cal, 2025, Month::March, 20, "JP Vernal Equinox 2025");

    // Showa Day (Apr 29)
    assert_holiday(&cal, 2025, Month::April, 29, "JP Showa Day 2025");

    // Golden Week: Constitution (May 3), Greenery (May 4), Children's (May 5)
    assert_holiday(&cal, 2025, Month::May, 3, "JP Constitution 2025");
    // May 4 2025 is Sunday, May 5 is Monday (Children's Day), May 6 is substitute
    assert_holiday(&cal, 2025, Month::May, 5, "JP Children's Day 2025");
    assert_holiday(&cal, 2025, Month::May, 6, "JP substitute 2025");

    // Marine Day: 3rd Monday of July
    assert_holiday(&cal, 2025, Month::July, 21, "JP Marine Day 2025");

    // Mountain Day (Aug 11)
    assert_holiday(&cal, 2025, Month::August, 11, "JP Mountain Day 2025");

    // Respect for Aged: 3rd Monday of September
    assert_holiday(&cal, 2025, Month::September, 15, "JP Respect Aged 2025");

    // Sport Day: 2nd Monday of October
    assert_holiday(&cal, 2025, Month::October, 13, "JP Sport Day 2025");

    // Culture Day (Nov 3)
    assert_holiday(&cal, 2025, Month::November, 3, "JP Culture Day 2025");

    // Labour Thanksgiving (Nov 23)
    assert_holiday(&cal, 2025, Month::November, 24, "JP Labour Thanksgiving observed 2025 (Sun→Mon)");
}

// ============================================================================
//  Brazil (B3)
// ============================================================================

#[test]
fn brazil_holidays_multiyear() {
    let cal = Calendar::Brazil;

    // Carnival Monday & Tuesday (47 and 46 days before Easter Sunday)
    // 2025: Easter = Apr 20 → Carnival Mon = Mar 3, Tue = Mar 4
    assert_holiday(&cal, 2025, Month::March, 3, "BR Carnival Mon 2025");
    assert_holiday(&cal, 2025, Month::March, 4, "BR Carnival Tue 2025");

    // Good Friday (2 days before Easter Sunday)
    assert_holiday(&cal, 2025, Month::April, 18, "BR Good Friday 2025");

    // Corpus Christi (60 days after Easter Sunday)
    // 2025: Apr 20 + 60 = June 19
    assert_holiday(&cal, 2025, Month::June, 19, "BR Corpus Christi 2025");

    // Fixed holidays
    assert_holiday(&cal, 2025, Month::January, 1, "BR New Year 2025");
    assert_holiday(&cal, 2025, Month::April, 21, "BR Tiradentes 2025");
    assert_holiday(&cal, 2025, Month::May, 1, "BR Labour Day 2025");
    assert_holiday(&cal, 2025, Month::September, 7, "BR Independence 2025");
    assert_holiday(&cal, 2025, Month::October, 12, "BR Our Lady 2025");
    assert_holiday(&cal, 2025, Month::November, 2, "BR All Souls 2025");
    assert_holiday(&cal, 2025, Month::November, 15, "BR Republic Day 2025");
    assert_holiday(&cal, 2025, Month::December, 25, "BR Christmas 2025");
}

// ============================================================================
//  Germany (FWB)
// ============================================================================

#[test]
fn germany_holidays_multiyear() {
    let cal = Calendar::Germany;

    assert_holiday(&cal, 2025, Month::January, 1, "DE New Year 2025");
    assert_holiday(&cal, 2025, Month::April, 18, "DE Good Friday 2025");
    assert_holiday(&cal, 2025, Month::April, 21, "DE Easter Monday 2025");
    assert_holiday(&cal, 2025, Month::May, 1, "DE Labour Day 2025");
    assert_holiday(&cal, 2025, Month::October, 3, "DE Unity Day 2025");
    assert_holiday(&cal, 2025, Month::December, 24, "DE Christmas Eve 2025");
    assert_holiday(&cal, 2025, Month::December, 25, "DE Christmas 2025");
    assert_holiday(&cal, 2025, Month::December, 26, "DE St. Stephen 2025");
    assert_holiday(&cal, 2025, Month::December, 31, "DE New Year Eve 2025");

    // 2024 Easter dates
    assert_holiday(&cal, 2024, Month::March, 29, "DE Good Friday 2024");
    assert_holiday(&cal, 2024, Month::April, 1, "DE Easter Monday 2024");
}

// ============================================================================
//  France
// ============================================================================

#[test]
fn france_holidays_multiyear() {
    let cal = Calendar::France;

    assert_holiday(&cal, 2025, Month::January, 1, "FR New Year 2025");
    assert_holiday(&cal, 2025, Month::April, 18, "FR Good Friday 2025");
    assert_holiday(&cal, 2025, Month::April, 21, "FR Easter Monday 2025");
    assert_holiday(&cal, 2025, Month::May, 1, "FR Labour Day 2025");

    // Ascension Thursday: 39 days after Easter Sunday
    // 2025: Apr 20 + 39 = May 29
    assert_holiday(&cal, 2025, Month::May, 29, "FR Ascension 2025");

    // Whit Monday: 50 days after Easter Sunday
    // 2025: Apr 20 + 50 = June 9
    assert_holiday(&cal, 2025, Month::June, 9, "FR Whit Monday 2025");

    assert_holiday(&cal, 2025, Month::July, 14, "FR Bastille Day 2025");
    assert_holiday(&cal, 2025, Month::August, 15, "FR Assumption 2025");
    assert_holiday(&cal, 2025, Month::November, 1, "FR All Saints 2025");
    assert_holiday(&cal, 2025, Month::November, 11, "FR Armistice 2025");
    assert_holiday(&cal, 2025, Month::December, 25, "FR Christmas 2025");
}

// ============================================================================
//  Italy
// ============================================================================

#[test]
fn italy_holidays_multiyear() {
    let cal = Calendar::Italy;

    assert_holiday(&cal, 2025, Month::January, 1, "IT New Year 2025");
    assert_holiday(&cal, 2025, Month::January, 6, "IT Epiphany 2025");
    assert_holiday(&cal, 2025, Month::April, 18, "IT Good Friday 2025");
    assert_holiday(&cal, 2025, Month::April, 21, "IT Easter Monday 2025");
    assert_holiday(&cal, 2025, Month::April, 25, "IT Liberation Day 2025");
    assert_holiday(&cal, 2025, Month::May, 1, "IT Labour Day 2025");
    assert_holiday(&cal, 2025, Month::June, 2, "IT Republic Day 2025");
    assert_holiday(&cal, 2025, Month::August, 15, "IT Assumption 2025");
    assert_holiday(&cal, 2025, Month::November, 1, "IT All Saints 2025");
    assert_holiday(&cal, 2025, Month::December, 8, "IT Immaculate Conception 2025");
    assert_holiday(&cal, 2025, Month::December, 25, "IT Christmas 2025");
    assert_holiday(&cal, 2025, Month::December, 26, "IT St. Stephen 2025");
}

// ============================================================================
//  Switzerland
// ============================================================================

#[test]
fn switzerland_holidays_multiyear() {
    let cal = Calendar::Switzerland;

    assert_holiday(&cal, 2025, Month::January, 2, "CH Berchtoldstag 2025");
    assert_holiday(&cal, 2025, Month::April, 18, "CH Good Friday 2025");
    assert_holiday(&cal, 2025, Month::April, 21, "CH Easter Monday 2025");

    // Ascension Thursday
    assert_holiday(&cal, 2025, Month::May, 29, "CH Ascension 2025");

    // Whit Monday
    assert_holiday(&cal, 2025, Month::June, 9, "CH Whit Monday 2025");

    assert_holiday(&cal, 2025, Month::May, 1, "CH Labour Day 2025");
    assert_holiday(&cal, 2025, Month::August, 1, "CH National Day 2025");
    assert_holiday(&cal, 2025, Month::December, 25, "CH Christmas 2025");
    assert_holiday(&cal, 2025, Month::December, 26, "CH St. Stephen 2025");
}

// ============================================================================
//  Sweden
// ============================================================================

#[test]
fn sweden_holidays_multiyear() {
    let cal = Calendar::Sweden;

    assert_holiday(&cal, 2025, Month::January, 1, "SE New Year 2025");
    assert_holiday(&cal, 2025, Month::January, 6, "SE Epiphany 2025");
    assert_holiday(&cal, 2025, Month::April, 18, "SE Good Friday 2025");
    assert_holiday(&cal, 2025, Month::April, 21, "SE Easter Monday 2025");

    // Ascension
    assert_holiday(&cal, 2025, Month::May, 29, "SE Ascension 2025");

    assert_holiday(&cal, 2025, Month::May, 1, "SE May Day 2025");
    assert_holiday(&cal, 2025, Month::June, 6, "SE National Day 2025");

    // Midsummer Eve: Friday between Jun 19-25
    assert_holiday(&cal, 2025, Month::June, 20, "SE Midsummer Eve 2025");

    assert_holiday(&cal, 2025, Month::December, 24, "SE Christmas Eve 2025");
    assert_holiday(&cal, 2025, Month::December, 25, "SE Christmas 2025");
    assert_holiday(&cal, 2025, Month::December, 26, "SE Boxing Day 2025");
    assert_holiday(&cal, 2025, Month::December, 31, "SE New Year Eve 2025");
}

// ============================================================================
//  Denmark
// ============================================================================

#[test]
fn denmark_holidays_multiyear() {
    let cal = Calendar::Denmark;

    assert_holiday(&cal, 2025, Month::January, 1, "DK New Year 2025");

    // Maundy/Holy Thursday (3 days before Easter Sunday)
    // 2025: Apr 17
    assert_holiday(&cal, 2025, Month::April, 17, "DK Holy Thursday 2025");
    assert_holiday(&cal, 2025, Month::April, 18, "DK Good Friday 2025");
    assert_holiday(&cal, 2025, Month::April, 21, "DK Easter Monday 2025");

    // Great Prayer Day (4th Friday after Easter Sunday)
    // 2025: Apr 20 + 26 = May 16
    assert_holiday(&cal, 2025, Month::May, 16, "DK Great Prayer Day 2025");

    // Ascension
    assert_holiday(&cal, 2025, Month::May, 29, "DK Ascension 2025");

    // Whit Monday
    assert_holiday(&cal, 2025, Month::June, 9, "DK Whit Monday 2025");

    assert_holiday(&cal, 2025, Month::June, 5, "DK Constitution Day 2025");
    assert_holiday(&cal, 2025, Month::December, 24, "DK Christmas Eve 2025");
    assert_holiday(&cal, 2025, Month::December, 25, "DK Christmas 2025");
    assert_holiday(&cal, 2025, Month::December, 26, "DK Boxing Day 2025");
}

// ============================================================================
//  Norway
// ============================================================================

#[test]
fn norway_holidays_multiyear() {
    let cal = Calendar::Norway;

    assert_holiday(&cal, 2025, Month::January, 1, "NO New Year 2025");
    assert_holiday(&cal, 2025, Month::April, 17, "NO Holy Thursday 2025");
    assert_holiday(&cal, 2025, Month::April, 18, "NO Good Friday 2025");
    assert_holiday(&cal, 2025, Month::April, 21, "NO Easter Monday 2025");
    assert_holiday(&cal, 2025, Month::May, 1, "NO Labour Day 2025");
    assert_holiday(&cal, 2025, Month::May, 17, "NO Constitution Day 2025");
    assert_holiday(&cal, 2025, Month::May, 29, "NO Ascension 2025");
    assert_holiday(&cal, 2025, Month::June, 9, "NO Whit Monday 2025");
    assert_holiday(&cal, 2025, Month::December, 25, "NO Christmas 2025");
    assert_holiday(&cal, 2025, Month::December, 26, "NO Boxing Day 2025");
}

// ============================================================================
//  Poland
// ============================================================================

#[test]
fn poland_holidays_multiyear() {
    let cal = Calendar::Poland;

    assert_holiday(&cal, 2025, Month::January, 1, "PL New Year 2025");
    assert_holiday(&cal, 2025, Month::January, 6, "PL Epiphany 2025");
    assert_holiday(&cal, 2025, Month::April, 21, "PL Easter Monday 2025");
    assert_holiday(&cal, 2025, Month::May, 1, "PL Labour Day 2025");
    assert_holiday(&cal, 2025, Month::May, 3, "PL Constitution Day 2025");

    // Corpus Christi (60 days after Easter Sunday)
    assert_holiday(&cal, 2025, Month::June, 19, "PL Corpus Christi 2025");

    assert_holiday(&cal, 2025, Month::August, 15, "PL Assumption 2025");
    assert_holiday(&cal, 2025, Month::November, 1, "PL All Saints 2025");
    assert_holiday(&cal, 2025, Month::November, 11, "PL Independence Day 2025");
    assert_holiday(&cal, 2025, Month::December, 25, "PL Christmas 2025");
    assert_holiday(&cal, 2025, Month::December, 26, "PL St. Stephen 2025");
}

// ============================================================================
//  Mexico (BMV)
// ============================================================================

#[test]
fn mexico_holidays_multiyear() {
    let cal = Calendar::Mexico;

    assert_holiday(&cal, 2025, Month::January, 1, "MX New Year 2025");

    // Constitution Day: 1st Monday of February
    assert_holiday(&cal, 2025, Month::February, 3, "MX Constitution Day 2025");

    // Benito Juárez: 3rd Monday of March
    assert_holiday(&cal, 2025, Month::March, 17, "MX Benito Juárez 2025");

    // Holy Thursday & Good Friday
    assert_holiday(&cal, 2025, Month::April, 17, "MX Holy Thursday 2025");
    assert_holiday(&cal, 2025, Month::April, 18, "MX Good Friday 2025");

    assert_holiday(&cal, 2025, Month::May, 1, "MX Labour Day 2025");
    assert_holiday(&cal, 2025, Month::September, 16, "MX Independence 2025");

    // Revolution Day: 3rd Monday of November
    assert_holiday(&cal, 2025, Month::November, 17, "MX Revolution Day 2025");

    assert_holiday(&cal, 2025, Month::December, 25, "MX Christmas 2025");
}

// ============================================================================
//  South Africa (JSE)
// ============================================================================

#[test]
fn south_africa_holidays_multiyear() {
    let cal = Calendar::SouthAfrica;

    assert_holiday(&cal, 2025, Month::January, 1, "ZA New Year 2025");
    assert_holiday(&cal, 2025, Month::March, 21, "ZA Human Rights Day 2025");
    assert_holiday(&cal, 2025, Month::April, 18, "ZA Good Friday 2025");
    assert_holiday(&cal, 2025, Month::April, 21, "ZA Family Day 2025");
    assert_holiday(&cal, 2025, Month::April, 28, "ZA Freedom Day observed 2025");
    assert_holiday(&cal, 2025, Month::May, 1, "ZA Workers Day 2025");
    assert_holiday(&cal, 2025, Month::June, 16, "ZA Youth Day 2025");
    assert_holiday(&cal, 2025, Month::August, 9, "ZA Nat Women's Day 2025");
    assert_holiday(&cal, 2025, Month::September, 24, "ZA Heritage Day 2025");
    assert_holiday(&cal, 2025, Month::December, 16, "ZA Reconciliation 2025");
    assert_holiday(&cal, 2025, Month::December, 25, "ZA Christmas 2025");
    assert_holiday(&cal, 2025, Month::December, 26, "ZA Goodwill Day 2025");
}

// ============================================================================
//  New Zealand (NZX)
// ============================================================================

#[test]
fn new_zealand_holidays_multiyear() {
    let cal = Calendar::NewZealand;

    // New Year + Day After
    assert_holiday(&cal, 2025, Month::January, 1, "NZ New Year 2025");
    assert_holiday(&cal, 2025, Month::January, 2, "NZ Day After 2025");

    // Waitangi Day (Feb 6, observed)
    assert_holiday(&cal, 2025, Month::February, 6, "NZ Waitangi Day 2025");

    // Easter
    assert_holiday(&cal, 2025, Month::April, 18, "NZ Good Friday 2025");
    assert_holiday(&cal, 2025, Month::April, 21, "NZ Easter Monday 2025");

    // ANZAC Day
    assert_holiday(&cal, 2025, Month::April, 25, "NZ ANZAC Day 2025");

    // King's Birthday (1st Monday of June)
    assert_holiday(&cal, 2025, Month::June, 2, "NZ King's Birthday 2025");

    // Labour Day (4th Monday of October)
    assert_holiday(&cal, 2025, Month::October, 27, "NZ Labour Day 2025");

    // Christmas + Boxing
    assert_holiday(&cal, 2025, Month::December, 25, "NZ Christmas 2025");
    assert_holiday(&cal, 2025, Month::December, 26, "NZ Boxing Day 2025");
}

// ============================================================================
//  Business days between — cross-calendar consistency
// ============================================================================

#[test]
fn business_days_between_consistency() {
    // For any calendar, the number of business days in a year should be
    // roughly 250-253 for standard calendars
    let cals = [
        Calendar::Target,
        Calendar::UnitedStates(USMarket::Settlement),
        Calendar::UnitedKingdom,
        Calendar::Canada,
        Calendar::Japan,
    ];

    let start = Date::from_ymd(2025, Month::January, 1);
    let end = Date::from_ymd(2025, Month::December, 31);

    for cal in &cals {
        let bd = cal.business_days_between(start, end);
        assert!(
            (240..=260).contains(&bd),
            "Calendar {cal:?}: business days in 2025 = {bd}, expected 240-260"
        );
    }
}

// ============================================================================
//  Easter algorithm cross-check
// ============================================================================

#[test]
fn easter_algorithm_known_dates() {
    // Validate against authoritative Easter table (1900-2099)
    let known_easters: &[(i32, Month, u32)] = &[
        (2000, Month::April, 24),
        (2005, Month::March, 28),
        (2010, Month::April, 5),
        (2015, Month::April, 6),
        (2016, Month::March, 28),
        (2017, Month::April, 17),
        (2018, Month::April, 2),
        (2019, Month::April, 22),
        (2020, Month::April, 13),
        (2021, Month::April, 5),
        (2022, Month::April, 18),
        (2023, Month::April, 10),
        (2024, Month::April, 1),
        (2025, Month::April, 21),
        (2026, Month::April, 6),
        (2027, Month::March, 29),
        (2028, Month::April, 17),
        (2029, Month::April, 2),
        (2030, Month::April, 22),
    ];

    // Easter Monday should be a TARGET holiday for all these years
    let cal = Calendar::Target;
    for &(yr, em_month, em_day) in known_easters {
        assert_holiday(
            &cal,
            yr,
            em_month,
            em_day,
            &format!("Easter Monday {yr}"),
        );
    }
}
