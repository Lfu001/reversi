use redis::{Client, Commands, JsonCommands, RedisError};
use redis_macros::FromRedisValue;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A game table.
#[cfg_attr(test, derive(Clone))]
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

    // /// Search for the first occurrence of a JSON value in an array in Redis.
    // ///
    // /// # Arguments
    // ///
    // /// * `key` - A key to search in.
    // /// * `path` - A JSON path to search in.
    // /// * `value` - A value to search for.
    // ///
    // /// # Returns
    // ///
    // /// The index of the first occurrence of the value in the array. -1 if not found.
    // fn json_arr_index(&self, key: &str, path: &str, value: &str) -> Result<i32, RedisError>;

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

    // fn json_arr_index(&self, key: &str, path: &str, value: &str) -> Result<i32, RedisError> {
    //     let mut conn = self.client.get_connection()?;
    //     let index: i32 = conn.json_arr_index(key, path, &value)?;
    //     Ok(index)
    // }

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

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::services::redis_client::GameTable;
    use redis_test::{MockCmd, MockRedisConnection};
    use serde_json::Value;

    /// Mock implementation of RedisClient.
    pub struct MockRedisClient {
        pub set_result: String,
        pub json_get_result: String,
        pub json_set_result: String,
        pub json_arr_append_result: String,
        // pub json_arr_index_result: String,
        pub exists_result: String,
        pub expire_result: String,
    }

    impl Default for MockRedisClient {
        fn default() -> Self {
            Self {
                set_result: String::from("1"),
                json_get_result: serde_json::to_string(&GameTable {
                    players: vec![],
                    game_state: String::from(""),
                })
                .unwrap(),
                json_set_result: String::from("1"),
                json_arr_append_result: String::from("1"),
                // json_arr_index_result: String::from("-1"),
                exists_result: String::from("0"),
                expire_result: String::from("1"),
            }
        }
    }

    impl RedisClient for MockRedisClient {
        fn set(&self, key: &str, value: &str) -> Result<(), RedisError> {
            let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("SET").arg(key).arg(value),
                Ok(self.set_result.to_owned()),
            )]);
            let _: () = mock_conn.set(key, value)?;
            Ok(())
        }

        fn json_get(&self, key: &str, path: &str) -> Result<GameTable, RedisError> {
            let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("JSON.GET").arg(key).arg(path),
                Ok(self.json_get_result.to_owned()),
            )]);
            let game_table: GameTable = mock_conn.json_get(key, path)?;
            Ok(game_table)
        }

        fn json_set(&self, key: &str, path: &str, value: &Value) -> Result<(), RedisError> {
            let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("JSON.SET")
                    .arg(key)
                    .arg(path)
                    .arg(serde_json::to_string(value)?),
                Ok(self.json_set_result.to_owned()),
            )]);
            let _: () = mock_conn.json_set(key, path, &value)?;
            Ok(())
        }

        fn json_arr_append(&self, key: &str, path: &str, value: &str) -> Result<(), RedisError> {
            let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("JSON.ARRAPPEND")
                    .arg(key)
                    .arg(path)
                    .arg(serde_json::to_string(value)?),
                Ok(self.json_arr_append_result.to_owned()),
            )]);
            let _: () = mock_conn.json_arr_append(key, path, &value)?;
            Ok(())
        }

        // fn json_arr_index(&self, key: &str, path: &str, value: &str) -> Result<i32, RedisError> {
        //     let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
        //         redis::cmd("JSON.ARRINDEX")
        //             .arg(key)
        //             .arg(path)
        //             .arg(serde_json::to_string(value)?),
        //         Ok(self.json_arr_index_result.to_owned()),
        //     )]);
        //     let index: i32 = mock_conn.json_arr_index(key, path, &value)?;
        //     Ok(index)
        // }

        fn exists(&self, key: &str) -> Result<bool, RedisError> {
            let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("EXISTS").arg(key),
                Ok(self.exists_result.to_owned()),
            )]);
            let exists: bool = mock_conn.exists(key)?;
            Ok(exists)
        }

        fn expire(&self, key: &str, seconds: i64) -> Result<(), RedisError> {
            let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("EXPIRE").arg(key).arg(seconds),
                Ok(self.expire_result.to_owned()),
            )]);
            let _: () = mock_conn.expire(key, seconds)?;
            Ok(())
        }
    }
}
