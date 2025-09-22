use crate::services::redis_client::RedisKey;
use common::{DiskColor, JudgeResult, Position, Table};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, ops::Deref};
use uuid::Uuid;

/// A player ID, represented as a UUID.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
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

impl Serialize for PlayerId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.id.to_string())
    }
}

impl<'de> Deserialize<'de> for PlayerId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let id = Uuid::parse_str(&s).map_err(serde::de::Error::custom)?;
        Ok(PlayerId { id })
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

/// A player profile, which includes the name and avatar of the player.
#[derive(Clone, Serialize, Deserialize)]
pub struct PlayerProfile {
    /// The name of the player.
    pub name: String,
    /// The avatar URL of the player.
    pub avatar_url: String,
}

/// A table state, represented as a JSON object.
#[derive(Serialize, Deserialize)]
pub struct TableState {
    /// The table.
    table: Table,
    /// The player roles.
    roles: HashMap<PlayerId, DiskColor>,
    /// The puttable positions.
    puttable_positions: Vec<Position>,
    /// The judge result.
    #[serde(skip_serializing_if = "Option::is_none")]
    judge_result: Option<JudgeResult>,
}

impl TableState {
    /// Creates a new [`TableState`].
    pub fn new(
        table: Table,
        roles: HashMap<PlayerId, DiskColor>,
        puttable_positions: Vec<Position>,
        judge_result: Option<JudgeResult>,
    ) -> Self {
        Self {
            table,
            roles,
            puttable_positions,
            judge_result,
        }
    }

    /// Returns a reference to the table of this [`TableState`].
    pub fn table(&self) -> &Table {
        &self.table
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
