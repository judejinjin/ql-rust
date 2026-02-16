//! Lazy evaluation and cached value patterns.
//!
//! These provide interior mutability for on-demand computation and caching,
//! mirroring QuantLib's `LazyObject` pattern.

use std::cell::{Cell, RefCell};

/// A dirty-flag mixin for lazy calculation with caching.
///
/// Tracks whether a cached result is still valid and supports
/// freezing (preventing invalidation).
pub struct LazyCache {
    calculated: Cell<bool>,
    frozen: Cell<bool>,
}

impl LazyCache {
    /// Create a new `LazyCache` in the not-yet-calculated state.
    pub fn new() -> Self {
        Self {
            calculated: Cell::new(false),
            frozen: Cell::new(false),
        }
    }

    /// Whether the cached result is currently valid.
    pub fn is_calculated(&self) -> bool {
        self.calculated.get()
    }

    /// Mark the cached result as invalid (dirty).
    ///
    /// Has no effect if the cache is frozen.
    pub fn invalidate(&self) {
        if !self.frozen.get() {
            self.calculated.set(false);
        }
    }

    /// Execute `f` only if the cache is invalid (and not frozen).
    ///
    /// Sets the calculated flag *before* calling `f` to prevent infinite
    /// recursion in cyclic dependency graphs.
    pub fn ensure_calculated<F: FnOnce()>(&self, f: F) {
        if !self.calculated.get() {
            self.calculated.set(true);
            f();
        }
    }

    /// Freeze the cache — subsequent calls to [`invalidate`](Self::invalidate)
    /// will be ignored.
    pub fn freeze(&self) {
        self.frozen.set(true);
    }

    /// Unfreeze the cache — invalidation will work again.
    pub fn unfreeze(&self) {
        self.frozen.set(false);
    }

    /// Whether the cache is currently frozen.
    pub fn is_frozen(&self) -> bool {
        self.frozen.get()
    }
}

impl Default for LazyCache {
    fn default() -> Self {
        Self::new()
    }
}

/// A lazily-computed, cached value.
///
/// The value is computed on first access (or after invalidation) and then
/// stored for subsequent reads.
pub struct Cached<T> {
    value: RefCell<Option<T>>,
    valid: Cell<bool>,
}

impl<T> Cached<T> {
    /// Create a new, empty cached value.
    pub fn new() -> Self {
        Self {
            value: RefCell::new(None),
            valid: Cell::new(false),
        }
    }

    /// Get the cached value, computing it with `f` if necessary.
    ///
    /// Returns a `Ref<T>` guard. The value is recomputed only when the cache
    /// has been invalidated.
    pub fn get_or_compute<F: FnOnce() -> T>(&self, f: F) -> std::cell::Ref<'_, T> {
        if !self.valid.get() {
            *self.value.borrow_mut() = Some(f());
            self.valid.set(true);
        }
        std::cell::Ref::map(self.value.borrow(), |v| v.as_ref().unwrap())
    }

    /// Mark the cached value as invalid — the next access will recompute.
    pub fn invalidate(&self) {
        self.valid.set(false);
    }

    /// Whether the cached value is currently valid.
    pub fn is_valid(&self) -> bool {
        self.valid.get()
    }
}

impl<T> Default for Cached<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn lazy_cache_basic() {
        let cache = LazyCache::new();
        assert!(!cache.is_calculated());

        let call_count = Cell::new(0u32);
        cache.ensure_calculated(|| {
            call_count.set(call_count.get() + 1);
        });
        assert!(cache.is_calculated());
        assert_eq!(call_count.get(), 1);

        // Second call should not re-execute
        cache.ensure_calculated(|| {
            call_count.set(call_count.get() + 1);
        });
        assert_eq!(call_count.get(), 1);
    }

    #[test]
    fn lazy_cache_invalidate() {
        let cache = LazyCache::new();
        let call_count = Cell::new(0u32);

        cache.ensure_calculated(|| call_count.set(call_count.get() + 1));
        assert_eq!(call_count.get(), 1);

        cache.invalidate();
        assert!(!cache.is_calculated());

        cache.ensure_calculated(|| call_count.set(call_count.get() + 1));
        assert_eq!(call_count.get(), 2);
    }

    #[test]
    fn lazy_cache_freeze() {
        let cache = LazyCache::new();
        let call_count = Cell::new(0u32);

        cache.ensure_calculated(|| call_count.set(call_count.get() + 1));
        cache.freeze();
        cache.invalidate(); // should be ignored
        assert!(cache.is_calculated());

        cache.unfreeze();
        cache.invalidate();
        assert!(!cache.is_calculated());

        cache.ensure_calculated(|| call_count.set(call_count.get() + 1));
        assert_eq!(call_count.get(), 2);
    }

    #[test]
    fn cached_value_compute_once() {
        let cached = Cached::<f64>::new();
        let call_count = Cell::new(0u32);

        let val = cached.get_or_compute(|| {
            call_count.set(call_count.get() + 1);
            42.0
        });
        assert_eq!(*val, 42.0);
        assert_eq!(call_count.get(), 1);
        drop(val);

        let val = cached.get_or_compute(|| {
            call_count.set(call_count.get() + 1);
            99.0
        });
        assert_eq!(*val, 42.0); // still cached
        assert_eq!(call_count.get(), 1);
    }

    #[test]
    fn cached_value_invalidate_recompute() {
        let cached = Cached::<f64>::new();

        let val = cached.get_or_compute(|| 1.0);
        assert_eq!(*val, 1.0);
        drop(val);

        cached.invalidate();
        assert!(!cached.is_valid());

        let val = cached.get_or_compute(|| 2.0);
        assert_eq!(*val, 2.0);
    }
}
