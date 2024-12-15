use crate::app_state::AppState;
use actix_web::{web, HttpResponse, Responder};
use base64::{engine::general_purpose::URL_SAFE, Engine as _};
use jwt_simple::prelude::*;
use redis::{Commands, ConnectionLike, JsonCommands};
use redis_macros::FromRedisValue;
use serde::{Deserialize, Serialize};
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake128,
};
use uuid::Uuid;

/// A join request message from the client.
#[derive(Serialize, Deserialize)]
pub(crate) struct JoinRequestMessage {
    // A password to join the table
    password: String,
}

/// A join response message to the client.
#[derive(Serialize, Deserialize)]
struct JoinResponseMessage {
    // A table ID
    table_id: String,
    // A JWT
    token: String,
}

/// A claim about a player in a JWT.
#[derive(Serialize, Deserialize)]
struct PlayerClaims {
    /// A player ID
    player_id: String,
}

/// A game table.
#[derive(Serialize, Deserialize, FromRedisValue)]
struct GameTable {
    /// A list of player IDs
    players: Vec<String>,
    /// A game state
    game_state: String,
}

// A handler for join table.
///
/// If the table does not exist, new table is created before joining. Otherwise, join the table.
///
/// # Arguments
///
/// * `req` - A request message from the client.
///
/// # Returns
///
/// A JSON object containing the table ID and JWT.
pub async fn join_table(
    req: web::Form<JoinRequestMessage>,
    app_state: web::Data<AppState>,
) -> impl Responder {
    // Save the player ID and table ID to Redis.
    let table_id = password_to_table_id(&req.password);
    let player_id = Uuid::new_v4().to_string();
    let mut conn = app_state.redis_client().get_connection().unwrap();
    match write_player_id_to_table(&mut conn, &table_id, &player_id) {
        Ok(()) => (),
        Err(()) => return HttpResponse::BadRequest().finish(),
    };

    // Generate a JWT including the player ID.
    let token = generate_jwt(app_state.jwt_key(), &player_id);

    HttpResponse::Ok().json(JoinResponseMessage { table_id, token })
}

/// Converts a password to a table ID.
///
/// # Arguments
///
/// * `password` - A password to join the table.
fn password_to_table_id(password: &str) -> String {
    // Hash the password by Shake128 (length = 10).
    let mut hasher = Shake128::default();
    hasher.update(password.as_bytes());
    let mut reader = hasher.finalize_xof();
    let mut hash = [0u8; 10];
    reader.read(&mut hash);

    // URL safe base64 encode the hash.
    URL_SAFE.encode(hash)
}

/// Writes a player ID to a table. If the table does not exist, a new table is created.
///
/// # Arguments
///
/// * `conn` - A Redis connection.
/// * `table_id` - A table ID.
/// * `player_id` - A player ID.
fn write_player_id_to_table<C: ConnectionLike>(
    conn: &mut C,
    table_id: &str,
    player_id: &str,
) -> Result<(), ()> {
    let exists: bool = conn.exists(table_id).unwrap();
    if exists {
        let game_table: GameTable = conn.json_get(table_id, "$").unwrap();
        if game_table.players.len() >= 2 {
            return Err(());
        }
        let _: () = conn
            .json_arr_append(table_id, "$.players", &player_id)
            .unwrap();
    } else {
        let response = ureq::get("http://127.0.0.1:8080/new")
            .call()
            .unwrap()
            .into_string()
            .unwrap();

        let json = serde_json::json!(&GameTable {
            players: vec![player_id.to_string()],
            game_state: response,
        });
        let _: () = conn.json_set(table_id, "$", &json).unwrap();
        let _: () = conn.expire(table_id, 1800).unwrap();
    }

    Ok(())
}

/// Generates a JWT including a player ID.
///
/// # Arguments
///
/// * `key` - A JWT secret key.
/// * `player_id` - A player ID.
fn generate_jwt(key: &HS256Key, player_id: &str) -> String {
    let custom_claims = PlayerClaims {
        player_id: player_id.to_owned(),
    };
    let claims = Claims::with_custom_claims(custom_claims, Duration::from_mins(30));
    key.authenticate(claims).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_password_to_table_id() {
        assert_eq!(password_to_table_id("foo"), "-E6Vy1-9IDiGOg==");
        assert_eq!(password_to_table_id("bar"), "BJ0wAaBTWD-yfg==");
        assert_eq!(password_to_table_id("baz"), "LQafpDw6uyP28g==");
    }

    #[test]
    fn test_generate_jwt() {
        let key = HS256Key::generate();
        let player_id = String::from("foo");
        let jwt = generate_jwt(&key, &player_id);

        let claims = key.verify_token::<PlayerClaims>(&jwt, None).unwrap();
        assert_eq!(claims.custom.player_id, player_id);
    }
}
