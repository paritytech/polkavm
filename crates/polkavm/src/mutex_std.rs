pub struct Mutex<T>(std::sync::Mutex<T>)
where
    T: ?Sized;

impl<T> Mutex<T> {
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(std::sync::Mutex::new(value))
    }

    #[inline]
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub fn lock(&self) -> std::sync::MutexGuard<T> {
        match self.0.lock() {
            Ok(mutable) => mutable,
            Err(poison) => poison.into_inner(),
        }
    }
}
