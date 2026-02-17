//! Global evaluation settings (singleton).
//!
//! The [`Settings`] singleton holds the current evaluation date and global
//! flags that affect pricing calculations throughout the library.

use std::sync::{OnceLock, RwLock};

use crate::observable::{Observable, SimpleObservable};

/// Global settings singleton.
///
/// Access via [`Settings::instance()`]. The evaluation date can be changed
/// at runtime and all observers (term structures, instruments) will be
/// notified.
pub struct Settings {
    evaluation_date: RwLock<Option<i32>>,
    include_reference_date_events: RwLock<bool>,
    include_todays_cashflows: RwLock<Option<bool>>,
    /// Observable for evaluation date changes.
    pub(crate) observable: SimpleObservable,
}

static SETTINGS: OnceLock<Settings> = OnceLock::new();

impl Settings {
    /// Get the global settings instance.
    pub fn instance() -> &'static Settings {
        SETTINGS.get_or_init(|| Settings {
            evaluation_date: RwLock::new(None),
            include_reference_date_events: RwLock::new(false),
            include_todays_cashflows: RwLock::new(None),
            observable: SimpleObservable::new(),
        })
    }

    /// Get the current evaluation date as a serial number.
    ///
    /// If no evaluation date has been set, returns `None` (callers should
    /// fall back to today's date).
    pub fn evaluation_date_serial(&self) -> Option<i32> {
        *self.evaluation_date.read().unwrap_or_else(|p| p.into_inner())
    }

    /// Set the evaluation date (as a serial number) and notify observers.
    pub fn set_evaluation_date_serial(&self, serial: i32) {
        *self.evaluation_date.write().unwrap_or_else(|p| p.into_inner()) = Some(serial);
        self.observable.notify_observers();
    }

    /// Clear the evaluation date (revert to "today").
    pub fn clear_evaluation_date(&self) {
        *self.evaluation_date.write().unwrap_or_else(|p| p.into_inner()) = None;
        self.observable.notify_observers();
    }

    /// Whether events on the reference date should be included.
    pub fn include_reference_date_events(&self) -> bool {
        *self.include_reference_date_events.read().unwrap_or_else(|p| p.into_inner())
    }

    /// Set whether events on the reference date should be included.
    pub fn set_include_reference_date_events(&self, include: bool) {
        *self.include_reference_date_events.write().unwrap_or_else(|p| p.into_inner()) = include;
    }

    /// Whether today's cash flows should be included.
    pub fn include_todays_cashflows(&self) -> Option<bool> {
        *self.include_todays_cashflows.read().unwrap_or_else(|p| p.into_inner())
    }

    /// Set whether today's cash flows should be included.
    pub fn set_include_todays_cashflows(&self, include: Option<bool>) {
        *self.include_todays_cashflows.write().unwrap_or_else(|p| p.into_inner()) = include;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observable::Observer;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    struct CountingObserver(AtomicU32);
    impl CountingObserver {
        fn new() -> Self {
            Self(AtomicU32::new(0))
        }
        fn count(&self) -> u32 {
            self.0.load(Ordering::SeqCst)
        }
    }
    impl Observer for CountingObserver {
        fn update(&self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn settings_singleton() {
        let s1 = Settings::instance();
        let s2 = Settings::instance();
        // Same pointer
        assert!(std::ptr::eq(s1, s2));
    }

    #[test]
    fn settings_evaluation_date() {
        let s = Settings::instance();
        // Set a date and verify
        s.set_evaluation_date_serial(45000);
        assert_eq!(s.evaluation_date_serial(), Some(45000));
        s.clear_evaluation_date();
    }

    #[test]
    fn settings_notifies_on_date_change() {
        let s = Settings::instance();
        let obs: Arc<dyn Observer> = Arc::new(CountingObserver::new());
        s.observable.register_observer(&obs);

        s.set_evaluation_date_serial(45001);

        let counting = unsafe { &*(Arc::as_ptr(&obs) as *const CountingObserver) };
        assert!(counting.count() >= 1);

        // Cleanup
        s.clear_evaluation_date();
    }
}
