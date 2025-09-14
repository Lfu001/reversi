use crate::services::redis_client::RedisClient;
use crate::websocket::server::GameSessionManagerHandle;
use jwt_simple::prelude::HS256Key;
use std::sync::Arc;
use tokio::sync::{Mutex, MutexGuard};

/// Application state holding shared resources.
pub struct AppState {
    /// Redis client for database interactions.
    redis_client: Arc<Mutex<dyn RedisClient>>,
    /// Key for signing JWTs.
    jwt_key: HS256Key,
    /// A salt for password hashing.
    salt: u32,
    /// Game server handle.
    game_server: GameSessionManagerHandle,
}

impl AppState {
    /// Creates a new `AppState`.
    ///
    /// # Arguments
    ///
    /// * `redis_client` - A Redis client instance.
    /// * `jwt_key` - Key for signing JWTs.
    /// * `salt` - A salt for password hashing.
    /// * `game_server` - A game server handle.
    pub fn new(
        redis_client: impl RedisClient + 'static,
        jwt_key: HS256Key,
        salt: u32,
        game_server: GameSessionManagerHandle,
    ) -> Self {
        AppState {
            redis_client: Arc::new(Mutex::new(redis_client)),
            jwt_key,
            salt,
            game_server,
        }
    }

    /// Returns a reference to the Redis client.
    pub async fn redis_client(&self) -> MutexGuard<'_, dyn RedisClient> {
        self.redis_client.lock().await
    }

    /// Returns a reference to the JWT signing key.
    pub fn jwt_key(&self) -> &HS256Key {
        &self.jwt_key
    }

    /// Returns the salt for password hashing.
    pub fn salt(&self) -> u32 {
        self.salt
    }

    /// Returns a reference to the game server handle.
    pub fn game_server(&self) -> &GameSessionManagerHandle {
        &self.game_server
    }
}
