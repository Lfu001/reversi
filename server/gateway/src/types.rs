use crate::services::redis_client::RedisKey;
use serde::{Deserialize, Serialize};
use std::ops::Deref;
use uuid::Uuid;

/// A player ID, represented as a UUID.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct PlayerId {
    /// The UUID of the player ID.
    pub id: Uuid,
}

impl PlayerId {
    /// Creates a new [`PlayerId`].
    pub fn new() -> Self {
        Self { id: Uuid::new_v4() }
    }
}

impl Deref for PlayerId {
    type Target = Uuid;

    fn deref(&self) -> &Self::Target {
        &self.id
    }
}

impl RedisKey for PlayerId {
    fn to_key(&self) -> String {
        format!("player:{}", self.id)
    }
}

/// A table ID, represented as a string.
#[derive(Eq, Hash, PartialEq, Clone, Debug)]
pub struct TableId {
    /// The ID of the table.
    id: String,
}

impl TableId {
    /// Creates a new [`TableId`].
    pub fn new(id: String) -> Self {
        Self { id }
    }
}

impl Deref for TableId {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.id
    }
}

impl RedisKey for TableId {
    fn to_key(&self) -> String {
        format!("table:{}", self.id)
    }
}

/// A connection ID, represented as a UUID.
#[derive(Clone, Eq, Hash, PartialEq)]
pub struct ConnectionId {
    /// The UUID of the connection ID.
    id: Uuid,
}

impl ConnectionId {
    /// Creates a new [`ConnectionId`].
    pub fn new() -> Self {
        Self { id: Uuid::new_v4() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn player_id_redis_key() {
        let player_id = PlayerId::new();
        let key = player_id.to_key();
        assert!(key.starts_with("player:"));
    }

    #[test]
    fn table_id_redis_key() {
        let table_id = TableId::new("foo".to_string());
        let key = table_id.to_key();
        assert!(key.starts_with("table:"));
    }
}
