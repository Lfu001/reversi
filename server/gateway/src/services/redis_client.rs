use redis::{aio::ConnectionManager, AsyncCommands, Client, JsonAsyncCommands, RedisError};
use serde_json::Value;

/// A trait for converting a type to a Redis key.
pub trait RedisKey {
    /// Converts the type to a Redis key.
    fn to_key(&self) -> String;
}

/// Redis client interface for easier mocking.
#[async_trait::async_trait]
pub trait RedisClient: Send + Sync + 'static {
    /// Gets a value from Redis.
    ///
    /// # Arguments
    ///
    /// * `key` - A key of the value.
    #[allow(unused)]
    async fn get(&mut self, key: &(dyn RedisKey + Sync)) -> Result<String, RedisError>;

    /// Sets a key-value pair in Redis.
    ///
    /// # Arguments
    ///
    /// * `key` - A key to set.
    /// * `value` - A value to set.
    #[allow(unused)]
    async fn set(&mut self, key: &(dyn RedisKey + Sync), value: &str) -> Result<(), RedisError>;

    /// Gets a JSON value from Redis.
    ///
    /// # Arguments
    ///
    /// * `key` - A key to get.
    /// * `path` - A JSON path to get.
    async fn json_get(
        &mut self,
        key: &(dyn RedisKey + Sync),
        path: &str,
    ) -> Result<Value, RedisError>;

    /// Sets a JSON value in Redis.
    ///
    /// # Arguments
    ///
    /// * `key` - A key to set.
    /// * `path` - A JSON path to set.
    /// * `value` - A value to set.
    async fn json_set(
        &mut self,
        key: &(dyn RedisKey + Sync),
        path: &str,
        value: &Value,
    ) -> Result<(), RedisError>;

    /// Checks if a key exists in Redis.
    ///
    /// # Arguments
    ///
    /// * `key` - A key to check.
    async fn exists(&mut self, key: &(dyn RedisKey + Sync)) -> Result<bool, RedisError>;

    /// Expires a key in Redis.
    ///
    /// # Arguments
    ///
    /// * `key` - A key to expire.
    /// * `seconds` - An expiration time in seconds.
    async fn expire(&mut self, key: &(dyn RedisKey + Sync), seconds: i64)
        -> Result<(), RedisError>;
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
    async fn get(&mut self, key: &(dyn RedisKey + Sync)) -> Result<String, RedisError> {
        self.connection_manager.get(key.to_key()).await
    }

    async fn set(&mut self, key: &(dyn RedisKey + Sync), value: &str) -> Result<(), RedisError> {
        let _: () = self.connection_manager.set(key.to_key(), value).await?;
        Ok(())
    }

    async fn json_get(
        &mut self,
        key: &(dyn RedisKey + Sync),
        path: &str,
    ) -> Result<Value, RedisError> {
        let json: String = self.connection_manager.json_get(key.to_key(), path).await?;
        let json = serde_json::from_str(&json).unwrap();
        Ok(json)
    }

    async fn json_set(
        &mut self,
        key: &(dyn RedisKey + Sync),
        path: &str,
        value: &Value,
    ) -> Result<(), RedisError> {
        let _: () = self
            .connection_manager
            .json_set(key.to_key(), path, value)
            .await?;
        Ok(())
    }

    async fn exists(&mut self, key: &(dyn RedisKey + Sync)) -> Result<bool, RedisError> {
        let exists: bool = self.connection_manager.exists(key.to_key()).await?;
        Ok(exists)
    }

    async fn expire(
        &mut self,
        key: &(dyn RedisKey + Sync),
        seconds: i64,
    ) -> Result<(), RedisError> {
        let _: () = self
            .connection_manager
            .expire(key.to_key(), seconds)
            .await?;
        Ok(())
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use common::Table;
    use redis_test::{MockCmd, MockRedisConnection};

    /// Mock implementation of RedisClient.
    #[derive(Clone)]
    pub struct MockRedisClient {
        #[allow(unused)]
        pub get_result: String,
        #[allow(unused)]
        pub set_result: String,
        pub json_get_result: String,
        pub json_set_result: String,
        pub exists_result: String,
        pub expire_result: String,
    }

    impl Default for MockRedisClient {
        fn default() -> Self {
            Self {
                get_result: String::from("1"),
                set_result: String::from("1"),
                json_get_result: serde_json::to_string(&Table::default()).unwrap(),
                json_set_result: String::from("1"),
                exists_result: String::from("0"),
                expire_result: String::from("1"),
            }
        }
    }

    #[async_trait::async_trait]
    impl RedisClient for MockRedisClient {
        async fn get(&mut self, key: &(dyn RedisKey + Sync)) -> Result<String, RedisError> {
            let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("GET").arg(key.to_key()),
                Ok(self.get_result.to_owned()),
            )]);
            let value: String = mock_conn.get(key.to_key()).await?;
            Ok(value)
        }

        async fn set(
            &mut self,
            key: &(dyn RedisKey + Sync),
            value: &str,
        ) -> Result<(), RedisError> {
            let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("SET").arg(key.to_key()).arg(value),
                Ok(self.set_result.to_owned()),
            )]);
            let _: () = mock_conn.set(key.to_key(), value).await?;
            Ok(())
        }

        async fn json_get(
            &mut self,
            key: &(dyn RedisKey + Sync),
            path: &str,
        ) -> Result<Value, RedisError> {
            let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("JSON.GET").arg(key.to_key()).arg(path),
                Ok(self.json_get_result.to_owned()),
            )]);
            let json: String = mock_conn.json_get(key.to_key(), path).await?;
            let json = serde_json::from_str(&json).unwrap();
            Ok(json)
        }

        async fn json_set(
            &mut self,
            key: &(dyn RedisKey + Sync),
            path: &str,
            value: &Value,
        ) -> Result<(), RedisError> {
            let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("JSON.SET")
                    .arg(key.to_key())
                    .arg(path)
                    .arg(serde_json::to_string(value)?),
                Ok(self.json_set_result.to_owned()),
            )]);
            let _: () = mock_conn.json_set(key.to_key(), path, &value).await?;
            Ok(())
        }

        async fn exists(&mut self, key: &(dyn RedisKey + Sync)) -> Result<bool, RedisError> {
            let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("EXISTS").arg(key.to_key()),
                Ok(self.exists_result.to_owned()),
            )]);
            let exists: bool = mock_conn.exists(key.to_key()).await?;
            Ok(exists)
        }

        async fn expire(
            &mut self,
            key: &(dyn RedisKey + Sync),
            seconds: i64,
        ) -> Result<(), RedisError> {
            let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("EXPIRE").arg(key.to_key()).arg(seconds),
                Ok(self.expire_result.to_owned()),
            )]);
            let _: () = mock_conn.expire(key.to_key(), seconds).await?;
            Ok(())
        }
    }
}
