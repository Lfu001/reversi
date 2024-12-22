use crate::app_state::AppState;
use crate::redis_client::RedisClient;
use actix_web::{web, HttpResponse, Responder};
use jwt_simple::prelude::*;
use redis::RedisError;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// An expiration time of a JWT and a player ID.
/// TODO: Make this configurable via environment variables.
const EXPIRE_TIME_SECONDS: u64 = 3 * 60 * 60;

/// A player registration request message.
#[derive(Serialize, Deserialize)]
pub struct PlayerRegistrationRequest {
    /// A name of the player.
    name: String,
}

/// A player registration response message.
#[derive(Serialize, Deserialize)]
pub struct PlayerRegistrationResponse {
    /// A JWT including the player's ID.
    token: String,
}

/// A claim about a player in a JWT.
#[derive(Serialize, Deserialize)]
struct PlayerClaims {
    /// A player ID
    player_id: String,
}

/// Registers a player and publishes a JWT.
///
/// # Arguments
///
/// * `name` - A name of the player.
///
/// # Returns
///
/// A JWT including the player's ID.
pub async fn register_player(
    req: web::Form<PlayerRegistrationRequest>,
    app_state: web::Data<AppState>,
) -> impl Responder {
    // Check name length.
    let name = req.name.as_str();
    if name.len() > 10 {
        return HttpResponse::BadRequest().json(serde_json::json!({ "error": "Name exceeds maximum length of 10 characters.", "error_code": "NAME_TOO_LONG"}));
    }

    // Write player ID and name to redis.
    let player_id = Uuid::new_v4().to_string();
    let res = write_player_id_name(app_state.redis_client_mut(), &player_id, name);
    if let Err(err) = res {
        log::error!("Failed to write player ID and name to redis: {}", err);
        return HttpResponse::InternalServerError().finish();
    }

    let token = match generate_jwt(app_state.jwt_key(), &player_id) {
        Ok(token) => token,
        Err(err) => {
            log::error!("Failed to generate JWT: {}", err);
            return HttpResponse::InternalServerError().finish();
        }
    };

    HttpResponse::Created().json(serde_json::json!(PlayerRegistrationResponse { token }))
}

/// Writes a player ID and name to redis.
///
/// # Arguments
///
/// * `client` - A Redis client.
/// * `player_id` - A player ID.
/// * `name` - A name of the player.
fn write_player_id_name(
    client: &dyn RedisClient,
    player_id: &str,
    name: &str,
) -> Result<(), RedisError> {
    let key = format!("player:{}", player_id);
    client.set(&key, name)?;
    client.expire(&key, EXPIRE_TIME_SECONDS as i64)?;

    Ok(())
}

/// Generates a JWT including a player ID.
///
/// # Arguments
///
/// * `key` - A JWT secret key.
/// * `player_id` - A player ID.
fn generate_jwt(key: &HS256Key, player_id: &str) -> Result<String, jwt_simple::Error> {
    let custom_claims = PlayerClaims {
        player_id: player_id.to_owned(),
    };
    let claims =
        Claims::with_custom_claims(custom_claims, Duration::from_secs(EXPIRE_TIME_SECONDS));
    key.authenticate(claims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use redis::Commands;
    use redis_test::{MockCmd, MockRedisConnection};

    /// Mock implementation of RedisClient.
    pub struct MockRedisClient {}

    impl MockRedisClient {
        pub fn new() -> Self {
            MockRedisClient {}
        }
    }

    impl RedisClient for MockRedisClient {
        fn set(&self, key: &str, value: &str) -> Result<(), RedisError> {
            let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("SET").arg(key).arg(value),
                Ok("1"),
            )]);
            let _: () = mock_conn.set(key, value)?;
            Ok(())
        }

        fn expire(&self, key: &str, seconds: i64) -> Result<(), RedisError> {
            let mut mock_conn = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("EXPIRE").arg(key).arg(EXPIRE_TIME_SECONDS),
                Ok("1"),
            )]);
            let _: () = mock_conn.expire(key, seconds)?;
            Ok(())
        }
    }

    mod endpoint_test {
        use super::*;
        use actix_web::{http::StatusCode, test, web, App};

        #[actix_web::test]
        async fn test_register_player() {
            let player_name = "foo";

            // Mock Redis setup
            let mock_redis_client = MockRedisClient::new();

            // AppState setup
            let jwt_key = HS256Key::generate();
            let app_state = AppState::new(Box::new(mock_redis_client), jwt_key.clone());

            // Test app
            let app = test::init_service(
                App::new()
                    .app_data(web::Data::new(app_state))
                    .route("/players", web::post().to(register_player)),
            )
            .await;

            // Test request
            let req = test::TestRequest::post()
                .uri("/players")
                .set_form(PlayerRegistrationRequest {
                    name: String::from(player_name),
                })
                .to_request();

            let resp = test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::CREATED);

            // Validate token
            let body: PlayerRegistrationResponse = test::read_body_json(resp).await;
            let claims = jwt_key
                .verify_token::<PlayerClaims>(&body.token, None)
                .unwrap();
            assert_eq!(claims.custom.player_id.len(), 36); // UUID length
        }
    }

    #[test]
    fn test_write_player_id_name() {
        let player_id = Uuid::new_v4().to_string();
        let name = "foo";
        let client = MockRedisClient::new();
        let res = write_player_id_name(&client, &player_id, name);
        assert!(res.is_ok());
    }

    #[test]
    fn test_generate_jwt() {
        let key = HS256Key::generate();
        let player_id = String::from("foo");
        let jwt = generate_jwt(&key, &player_id).unwrap();

        let claims = key.verify_token::<PlayerClaims>(&jwt, None).unwrap();
        assert_eq!(claims.custom.player_id, player_id);
    }
}
