use redis::{Client, Commands, JsonCommands, RedisError};
use redis_macros::FromRedisValue;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A game table.
#[derive(Serialize, Deserialize, FromRedisValue)]
pub struct GameTable {
    /// A list of player IDs
    players: Vec<String>,
    /// A game state
    game_state: String,
}

impl GameTable {
    /// Creates a new `GameTable` instance.
    pub fn new(players: Vec<String>, game_state: String) -> Self {
        GameTable {
            players,
            game_state,
        }
    }

    /// Returns a reference to the players.
    pub fn players(&self) -> &Vec<String> {
        &self.players
    }
}

/// Redis client interface for easier mocking.
pub trait RedisClient {
    /// Sets a key-value pair in Redis.
    ///
    /// # Arguments
    ///
    /// * `key` - A key to set.
    /// * `value` - A value to set.
    fn set(&self, key: &str, value: &str) -> Result<(), RedisError>;

    /// Gets a JSON value from Redis.
    ///
    /// # Arguments
    ///
    /// * `key` - A key to get.
    /// * `path` - A JSON path to get.
    fn json_get(&self, key: &str, path: &str) -> Result<GameTable, RedisError>;

    /// Sets a JSON value in Redis.
    ///
    /// # Arguments
    ///
    /// * `key` - A key to set.
    /// * `path` - A JSON path to set.
    /// * `value` - A value to set.
    fn json_set(&self, key: &str, path: &str, value: &Value) -> Result<(), RedisError>;

    /// Appends a value to a JSON array in Redis.
    ///
    /// # Arguments
    ///
    /// * `key` - A key to append to.
    /// * `path` - A JSON path to append to.
    /// * `value` - A value to append.
    fn json_arr_append(&self, key: &str, path: &str, value: &str) -> Result<(), RedisError>;

    /// Checks if a key exists in Redis.
    ///
    /// # Arguments
    ///
    /// * `key` - A key to check.
    fn exists(&self, key: &str) -> Result<bool, RedisError>;

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

    fn json_get(&self, key: &str, path: &str) -> Result<GameTable, RedisError> {
        let mut conn = self.client.get_connection()?;
        let game_table: GameTable = conn.json_get(key, path)?;
        Ok(game_table)
    }

    fn json_set(&self, key: &str, path: &str, value: &Value) -> Result<(), RedisError> {
        let mut conn = self.client.get_connection()?;
        let _: () = conn.json_set(key, path, value)?;
        Ok(())
    }

    fn json_arr_append(&self, key: &str, path: &str, value: &str) -> Result<(), RedisError> {
        let mut conn = self.client.get_connection()?;
        let _: () = conn.json_arr_append(key, path, &value)?;
        Ok(())
    }

    fn exists(&self, key: &str) -> Result<bool, RedisError> {
        let mut conn = self.client.get_connection()?;
        let exists: bool = conn.exists(key)?;
        Ok(exists)
    }

    fn expire(&self, key: &str, seconds: i64) -> Result<(), RedisError> {
        let mut conn = self.client.get_connection()?;
        let _: () = conn.expire(key, seconds)?;
        Ok(())
    }
}
