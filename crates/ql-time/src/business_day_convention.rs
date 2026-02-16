//! Business day adjustment conventions.

use serde::{Deserialize, Serialize};

/// Convention for adjusting a date that falls on a holiday or weekend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BusinessDayConvention {
    /// Do not adjust.
    Unadjusted,
    /// Choose the first business day after the given holiday.
    Following,
    /// Choose the first business day after the given holiday unless it
    /// belongs to a different month, in which case choose the first
    /// business day before the holiday.
    ModifiedFollowing,
    /// Choose the first business day before the given holiday.
    Preceding,
    /// Choose the first business day before the given holiday unless it
    /// belongs to a different month, in which case choose the first
    /// business day after the holiday.
    ModifiedPreceding,
    /// Choose the nearest business day to the given holiday.
    Nearest,
    /// Choose the first business day after the given holiday; if the
    /// original date falls in the second half of the month, choose using
    /// modified-following convention instead.
    HalfMonthModifiedFollowing,
}
