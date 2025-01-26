use redis::{aio::ConnectionManager, AsyncCommands, Client, JsonAsyncCommands, RedisError};
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
#[async_trait::async_trait]
pub trait RedisClient: Send {
    /// Sets a key-value pair in Redis.
    ///
    /// # Arguments
    ///
    /// * `key` - A key to set.
    /// * `value` - A value to set.
    async fn set(&mut self, key: &str, value: &str) -> Result<(), RedisError>;

    /// Gets a JSON value from Redis.
    ///
    /// # Arguments
    ///
    /// * `key` - A key to get.
    /// * `path` - A JSON path to get.
    async fn json_get(&mut self, key: &str, path: &str) -> Result<GameTable, RedisError>;

    /// Sets a JSON value in Redis.
    ///
    /// # Arguments
    ///
    /// * `key` - A key to set.
    /// * `path` - A JSON path to set.
    /// * `value` - A value to set.
    async fn json_set(&mut self, key: &str, path: &str, value: &Value) -> Result<(), RedisError>;

    /// Appends a value to a JSON array in Redis.
    ///
    /// # Arguments
    ///
    /// * `key` - A key to append to.
    /// * `path` - A JSON path to append to.
    /// * `value` - A value to append.
    async fn json_arr_append(
        &mut self,
        key: &str,
        path: &str,
        value: &str,
    ) -> Result<(), RedisError>;

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
    async fn exists(&mut self, key: &str) -> Result<bool, RedisError>;

    /// Expires a key in Redis.
    ///
    /// # Arguments
    ///
    /// * `key` - A key to expire.
    /// * `seconds` - An expiration time in seconds.
    async fn expire(&mut self, key: &str, seconds: i64) -> Result<(), RedisError>;
}

/// Real Redis client implementation.
#[derive(Clone)]
pub struct RealRedisClient {
    connection_manager: ConnectionManager,
}

impl RealRedisClient {
    /// Creates a new `RealRedisClient` instance.
    ///
    /// # Arguments
    ///
    /// * `client` - A Redis client instance.
    pub async fn new(client: Client) -> Self {
        RealRedisClient {
            connection_manager: client.get_connection_manager().await.unwrap(),
        }
    }
}

#[async_trait::async_trait]
impl RedisClient for RealRedisClient {
    async fn set(&mut self, key: &str, value: &str) -> Result<(), RedisError> {
        let _: () = self.connection_manager.set(key, value).await?;
        Ok(())
    }

    async fn json_get(&mut self, key: &str, path: &str) -> Result<GameTable, RedisError> {
        let game_table: GameTable = self.connection_manager.json_get(key, path).await?;
        Ok(game_table)
    }

    async fn json_set(&mut self, key: &str, path: &str, value: &Value) -> Result<(), RedisError> {
        let _: () = self.connection_manager.json_set(key, path, value).await?;
        Ok(())
    }

    async fn json_arr_append(
        &mut self,
        key: &str,
        path: &str,
        value: &str,
    ) -> Result<(), RedisError> {
        let _: () = self
            .connection_manager
            .json_arr_append(key, path, &value)
            .await?;
        Ok(())
    }

    // fn json_arr_index(&self, key: &str, path: &str, value: &str) -> Result<i32, RedisError> {
    //     let mut conn = self.client.get_connection()?;
    //     let index: i32 = conn.json_arr_index(key, path, &value)?;
    //     Ok(index)
    // }

    async fn exists(&mut self, key: &str) -> Result<bool, RedisError> {
        let exists: bool = self.connection_manager.exists(key).await?;
        Ok(exists)
    }

    async fn expire(&mut self, key: &str, seconds: i64) -> Result<(), RedisError> {
        let _: () = self.connection_manager.expire(key, seconds).await?;
        Ok(())
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use redis_test::{MockCmd, MockRedisConnection};

    /// Mock implementation of RedisClient.
    #[derive(Clone)]
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

    #[async_trait::async_trait]
    impl RedisClient for MockRedisClient {
        async fn set(&mut self, key: &str, value: &str) -> Result<(), RedisError> {
            let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("SET").arg(key).arg(value),
                Ok(self.set_result.to_owned()),
            )]);
            let _: () = mock_conn.set(key, value).await?;
            Ok(())
        }

        async fn json_get(&mut self, key: &str, path: &str) -> Result<GameTable, RedisError> {
            let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("JSON.GET").arg(key).arg(path),
                Ok(self.json_get_result.to_owned()),
            )]);
            let game_table: GameTable = mock_conn.json_get(key, path).await?;
            Ok(game_table)
        }

        async fn json_set(
            &mut self,
            key: &str,
            path: &str,
            value: &Value,
        ) -> Result<(), RedisError> {
            let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("JSON.SET")
                    .arg(key)
                    .arg(path)
                    .arg(serde_json::to_string(value)?),
                Ok(self.json_set_result.to_owned()),
            )]);
            let _: () = mock_conn.json_set(key, path, &value).await?;
            Ok(())
        }

        async fn json_arr_append(
            &mut self,
            key: &str,
            path: &str,
            value: &str,
        ) -> Result<(), RedisError> {
            let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("JSON.ARRAPPEND")
                    .arg(key)
                    .arg(path)
                    .arg(serde_json::to_string(value)?),
                Ok(self.json_arr_append_result.to_owned()),
            )]);
            let _: () = mock_conn.json_arr_append(key, path, &value).await?;
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

        async fn exists(&mut self, key: &str) -> Result<bool, RedisError> {
            let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("EXISTS").arg(key),
                Ok(self.exists_result.to_owned()),
            )]);
            let exists: bool = mock_conn.exists(key).await?;
            Ok(exists)
        }

        async fn expire(&mut self, key: &str, seconds: i64) -> Result<(), RedisError> {
            let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("EXPIRE").arg(key).arg(seconds),
                Ok(self.expire_result.to_owned()),
            )]);
            let _: () = mock_conn.expire(key, seconds).await?;
            Ok(())
        }
    }
}
