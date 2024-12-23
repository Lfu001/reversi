use redis::{Client, Commands, RedisError};

/// Redis client interface for easier mocking.
pub trait RedisClient {
    /// Sets a key-value pair in Redis.
    ///
    /// # Arguments
    ///
    /// * `key` - A key to set.
    /// * `value` - A value to set.
    fn set(&self, key: &str, value: &str) -> Result<(), RedisError>;

    /// Expires a key in Redis.
    ///
    /// # Arguments
    ///
    /// * `key` - A key to expire.
    /// * `seconds` - An expiration time in seconds.
    fn expire(&self, key: &str, seconds: i64) -> Result<(), RedisError>;
}

/// Real Redis client implementation.
pub struct RealRedisClient {
    client: Client,
}

impl RealRedisClient {
    /// Creates a new `RealRedisClient` instance.
    ///
    /// # Arguments
    ///
    /// * `client` - A Redis client instance.
    pub fn new(client: Client) -> Self {
        RealRedisClient { client }
    }
}

impl RedisClient for RealRedisClient {
    fn set(&self, key: &str, value: &str) -> Result<(), RedisError> {
        let mut conn = self.client.get_connection()?;
        let _: () = conn.set(key, value)?;
        Ok(())
    }

    fn expire(&self, key: &str, seconds: i64) -> Result<(), RedisError> {
        let mut conn = self.client.get_connection()?;
        let _: () = conn.expire(key, seconds)?;
        Ok(())
    }
}
