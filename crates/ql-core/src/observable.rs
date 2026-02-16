//! Observer / Observable pattern for dependency tracking and notification.
//!
//! When an observable (e.g., a market quote) changes, all registered observers
//! (e.g., term structures, instruments) are notified so they can invalidate
//! their cached results.

use std::sync::{Arc, RwLock, Weak};

/// Unique identifier for an observer registration.
pub type ObserverId = u64;

/// An entity that can be notified when a dependency changes.
pub trait Observer: Send + Sync {
    /// Called when an observed object has changed.
    fn update(&self);
}

/// Internal bookkeeping for observer registrations.
pub struct ObservableState {
    pub(crate) next_id: u64,
    pub(crate) observers: Vec<(ObserverId, Weak<dyn Observer>)>,
}

impl ObservableState {
    /// Create a new, empty observable state.
    pub fn new() -> Self {
        Self {
            next_id: 0,
            observers: Vec::new(),
        }
    }
}

impl Default for ObservableState {
    fn default() -> Self {
        Self::new()
    }
}

/// An entity that notifies registered observers when its state changes.
///
/// Implementors must provide access to shared [`ObservableState`] wrapped in
/// an `RwLock`.
pub trait Observable {
    /// Return a reference to the internal observer registry.
    fn observable_state(&self) -> &RwLock<ObservableState>;

    /// Register an observer. Returns an ID that can be used to unregister.
    fn register_observer(&self, observer: &Arc<dyn Observer>) -> ObserverId {
        let mut state = self.observable_state().write().unwrap();
        let id = state.next_id;
        state.next_id += 1;
        state.observers.push((id, Arc::downgrade(observer)));
        id
    }

    /// Unregister an observer by its ID.
    fn unregister_observer(&self, id: ObserverId) {
        let mut state = self.observable_state().write().unwrap();
        state.observers.retain(|(oid, _)| *oid != id);
    }

    /// Notify all registered observers that this observable has changed.
    ///
    /// Observers whose `Arc` has been dropped (only `Weak` remains) are
    /// silently removed.
    fn notify_observers(&self) {
        let observers: Vec<Arc<dyn Observer>> = {
            let mut state = self.observable_state().write().unwrap();
            // Purge dead weak references while collecting live ones.
            let mut live = Vec::new();
            let mut upgraded = Vec::new();
            for (id, weak) in state.observers.drain(..) {
                if let Some(strong) = weak.upgrade() {
                    live.push((id, Arc::downgrade(&strong)));
                    upgraded.push(strong);
                }
            }
            state.observers = live;
            upgraded
        };
        // Notify outside the lock to prevent deadlocks.
        for observer in &observers {
            observer.update();
        }
    }
}

/// A simple, concrete observable that can be embedded in other types.
pub struct SimpleObservable {
    state: RwLock<ObservableState>,
}

impl SimpleObservable {
    /// Create a new `SimpleObservable`.
    pub fn new() -> Self {
        Self {
            state: RwLock::new(ObservableState::new()),
        }
    }
}

impl Default for SimpleObservable {
    fn default() -> Self {
        Self::new()
    }
}

impl Observable for SimpleObservable {
    fn observable_state(&self) -> &RwLock<ObservableState> {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A test observer that counts how many times it was notified.
    struct CountingObserver {
        count: AtomicU32,
    }

    impl CountingObserver {
        fn new() -> Self {
            Self {
                count: AtomicU32::new(0),
            }
        }

        fn count(&self) -> u32 {
            self.count.load(Ordering::SeqCst)
        }
    }

    impl Observer for CountingObserver {
        fn update(&self) {
            self.count.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn register_and_notify() {
        let observable = SimpleObservable::new();
        let observer: Arc<dyn Observer> = Arc::new(CountingObserver::new());

        observable.register_observer(&observer);
        observable.notify_observers();
        observable.notify_observers();

        // Downcast to check count
        let counting = observer
            .as_ref()
            .as_any_counting()
            .expect("should be CountingObserver");
        assert_eq!(counting, 2);
    }

    #[test]
    fn multiple_observers() {
        let observable = SimpleObservable::new();
        let obs1: Arc<dyn Observer> = Arc::new(CountingObserver::new());
        let obs2: Arc<dyn Observer> = Arc::new(CountingObserver::new());

        observable.register_observer(&obs1);
        observable.register_observer(&obs2);
        observable.notify_observers();

        // Both should have received one notification
        // Access count via raw pointer since we know the concrete type
        let c1 = unsafe { &*(Arc::as_ptr(&obs1) as *const CountingObserver) };
        let c2 = unsafe { &*(Arc::as_ptr(&obs2) as *const CountingObserver) };
        assert_eq!(c1.count(), 1);
        assert_eq!(c2.count(), 1);
    }

    #[test]
    fn unregister_observer() {
        let observable = SimpleObservable::new();
        let observer: Arc<dyn Observer> = Arc::new(CountingObserver::new());

        let id = observable.register_observer(&observer);
        observable.notify_observers();
        observable.unregister_observer(id);
        observable.notify_observers(); // should NOT reach observer

        let c = unsafe { &*(Arc::as_ptr(&observer) as *const CountingObserver) };
        assert_eq!(c.count(), 1); // only the first notification
    }

    #[test]
    fn dropped_observer_is_purged() {
        let observable = SimpleObservable::new();
        let observer: Arc<dyn Observer> = Arc::new(CountingObserver::new());
        observable.register_observer(&observer);

        // Drop the only strong reference
        drop(observer);

        // Should not panic — dead weak refs are purged
        observable.notify_observers();

        // Verify the dead observer was removed
        let state = observable.observable_state().read().unwrap();
        assert_eq!(state.observers.len(), 0);
    }

    // Helper trait to safely downcast in tests
    trait AsAnyCounting {
        fn as_any_counting(&self) -> Option<u32>;
    }

    impl AsAnyCounting for dyn Observer {
        fn as_any_counting(&self) -> Option<u32> {
            // We know in tests these are CountingObserver
            let ptr = self as *const dyn Observer as *const CountingObserver;
            Some(unsafe { &*ptr }.count())
        }
    }
}
