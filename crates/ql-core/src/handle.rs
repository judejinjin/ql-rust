//! Relinkable handle pattern — shared, swappable references.
//!
//! A [`Handle<T>`] is a cheaply-cloneable smart pointer. All clones share the
//! same inner link, so when the owner calls [`RelinkableHandle::link_to`],
//! every `Handle` clone immediately sees the new object.

use std::sync::{Arc, RwLock};

use crate::errors::{QLError, QLResult};
use crate::observable::{ObservableState, ObserverId, Observer};

/// Internal link shared by all clones of a Handle.
struct Link<T: ?Sized> {
    inner: Option<Arc<T>>,
    observable_state: ObservableState,
}

/// A shared, read-only reference to a (possibly absent) object.
///
/// All clones of a `Handle` point to the same underlying [`Link`] and see the
/// same object. The object can only be changed via the corresponding
/// [`RelinkableHandle`].
pub struct Handle<T: ?Sized> {
    link: Arc<RwLock<Link<T>>>,
}

impl<T: ?Sized> Handle<T> {
    /// Create a handle pointing to the given object.
    pub fn new(obj: Arc<T>) -> Self {
        Self {
            link: Arc::new(RwLock::new(Link {
                inner: Some(obj),
                observable_state: ObservableState::new(),
            })),
        }
    }

    /// Create an empty handle (not yet linked to any object).
    pub fn empty() -> Self {
        Self {
            link: Arc::new(RwLock::new(Link {
                inner: None,
                observable_state: ObservableState::new(),
            })),
        }
    }

    /// Get the underlying object. Returns an error if the handle is empty.
    pub fn get(&self) -> QLResult<Arc<T>> {
        self.link
            .read()
            .unwrap()
            .inner
            .as_ref()
            .cloned()
            .ok_or(QLError::EmptyHandle)
    }

    /// Whether this handle is empty (not linked to any object).
    pub fn is_empty(&self) -> bool {
        self.link.read().unwrap().inner.is_none()
    }

    /// Register an observer to be notified when this handle is relinked.
    pub fn register_observer(&self, observer: &Arc<dyn Observer>) -> ObserverId {
        let mut link = self.link.write().unwrap();
        let id = link.observable_state.next_id;
        link.observable_state.next_id += 1;
        link.observable_state
            .observers
            .push((id, Arc::downgrade(observer)));
        id
    }
}

impl<T: ?Sized> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self {
            link: Arc::clone(&self.link),
        }
    }
}

/// The owning side of a relinkable handle.
///
/// Only the holder of a `RelinkableHandle` can call [`link_to`](Self::link_to).
/// All `Handle` clones will immediately see the new object.
pub struct RelinkableHandle<T: ?Sized> {
    handle: Handle<T>,
}

impl<T: ?Sized> RelinkableHandle<T> {
    /// Create a relinkable handle initially pointing to `obj`.
    pub fn new(obj: Arc<T>) -> Self {
        Self {
            handle: Handle::new(obj),
        }
    }

    /// Create an empty relinkable handle.
    pub fn empty() -> Self {
        Self {
            handle: Handle::empty(),
        }
    }

    /// Get a shareable, read-only [`Handle`] clone.
    pub fn handle(&self) -> Handle<T> {
        self.handle.clone()
    }

    /// Relink all associated handles to a new object and notify observers.
    pub fn link_to(&self, obj: Arc<T>) {
        {
            let mut link = self.handle.link.write().unwrap();
            link.inner = Some(obj);
        }
        // Notify observers outside the write lock
        self.notify_observers_internal();
    }

    fn notify_observers_internal(&self) {
        let observers: Vec<Arc<dyn Observer>> = {
            let mut link = self.handle.link.write().unwrap();
            let mut live = Vec::new();
            let mut upgraded = Vec::new();
            for (id, weak) in link.observable_state.observers.drain(..) {
                if let Some(strong) = weak.upgrade() {
                    live.push((id, Arc::downgrade(&strong)));
                    upgraded.push(strong);
                }
            }
            link.observable_state.observers = live;
            upgraded
        };
        for observer in &observers {
            observer.update();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    struct DummyObj(f64);

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
    fn handle_new_and_get() {
        let obj = Arc::new(DummyObj(42.0));
        let handle = Handle::new(obj);

        assert!(!handle.is_empty());
        assert_eq!(handle.get().unwrap().0, 42.0);
    }

    #[test]
    fn handle_empty() {
        let handle: Handle<DummyObj> = Handle::empty();
        assert!(handle.is_empty());
        assert!(handle.get().is_err());
    }

    #[test]
    fn handle_clone_shares_same_object() {
        let obj = Arc::new(DummyObj(10.0));
        let h1 = Handle::new(obj);
        let h2 = h1.clone();

        assert_eq!(h1.get().unwrap().0, 10.0);
        assert_eq!(h2.get().unwrap().0, 10.0);
    }

    #[test]
    fn relinkable_handle_relink() {
        let obj1 = Arc::new(DummyObj(1.0));
        let obj2 = Arc::new(DummyObj(2.0));

        let rh = RelinkableHandle::new(obj1);
        let h = rh.handle();

        assert_eq!(h.get().unwrap().0, 1.0);

        rh.link_to(obj2);
        assert_eq!(h.get().unwrap().0, 2.0);
    }

    #[test]
    fn relinkable_handle_notifies_observers() {
        let obj1 = Arc::new(DummyObj(1.0));
        let rh = RelinkableHandle::new(obj1);
        let h = rh.handle();

        let obs: Arc<dyn Observer> = Arc::new(CountingObserver::new());
        h.register_observer(&obs);

        rh.link_to(Arc::new(DummyObj(2.0)));
        rh.link_to(Arc::new(DummyObj(3.0)));

        let counting = unsafe { &*(Arc::as_ptr(&obs) as *const CountingObserver) };
        assert_eq!(counting.count(), 2);
    }

    #[test]
    fn relinkable_handle_empty_then_link() {
        let rh: RelinkableHandle<DummyObj> = RelinkableHandle::empty();
        let h = rh.handle();

        assert!(h.is_empty());

        rh.link_to(Arc::new(DummyObj(99.0)));
        assert!(!h.is_empty());
        assert_eq!(h.get().unwrap().0, 99.0);
    }
}
