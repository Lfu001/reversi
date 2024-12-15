use jwt_simple::prelude::HS256Key;
use redis::Client;

/// Application state holding shared resources.
pub struct AppState {
    /// Redis client for database interactions.
    redis_client: Client,
    /// Key for signing JWTs.
    jwt_key: HS256Key,
}

impl AppState {
    /// Creates a new `AppState`.
    ///
    /// # Arguments
    ///
    /// * `redis_client` - A Redis client instance.
    /// * `jwt_key` - A key for JWT signing.
    pub fn new(redis_client: Client, jwt_key: HS256Key) -> Self {
        AppState {
            redis_client,
            jwt_key,
        }
    }

    /// Returns a reference to the Redis client.
    pub fn redis_client(&self) -> &Client {
        &self.redis_client
    }

    /// Returns a reference to the JWT signing key.
    pub fn jwt_key(&self) -> &HS256Key {
        &self.jwt_key
    }
}
