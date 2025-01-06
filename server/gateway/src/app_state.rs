use crate::redis_client::RedisClient;
use jwt_simple::{
    prelude::HS256Key,
    reexports::rand::{thread_rng, RngCore},
};

/// Application state holding shared resources.
pub struct AppState {
    /// Redis client for database interactions.
    redis_client: Box<dyn RedisClient>,
    /// Key for signing JWTs.
    jwt_key: HS256Key,
    /// A salt for password hashing.
    salt: u32,
}

impl AppState {
    /// Creates a new `AppState`.
    ///
    /// # Arguments
    ///
    /// * `redis_client` - A Redis client instance.
    pub fn new(redis_client: Box<dyn RedisClient>) -> Self {
        let jwt_key = HS256Key::generate();
        let salt = thread_rng().next_u32();
        AppState {
            redis_client,
            jwt_key,
            salt,
        }
    }

    /// Returns a reference to the Redis client.
    pub fn redis_client_mut(&self) -> &dyn RedisClient {
        self.redis_client.as_ref()
    }

    /// Returns a reference to the JWT signing key.
    pub fn jwt_key(&self) -> &HS256Key {
        &self.jwt_key
    }

    /// Returns the salt for password hashing.
    pub fn salt(&self) -> u32 {
        self.salt
    }
}
