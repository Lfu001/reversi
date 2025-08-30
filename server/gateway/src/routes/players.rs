use crate::{
    app_state::AppState,
    authentication::{generate_jwt, EXPIRE_TIME_SECONDS},
    services::redis_client::RedisClient,
    types::PlayerId,
};
use actix_web::{web, HttpResponse, Responder};
use redis::RedisError;
use serde::{Deserialize, Serialize};

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
    let player_id = PlayerId::new();
    let res = write_player_id_name(&mut *app_state.redis_client().await, &player_id, name).await;
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
async fn write_player_id_name(
    client: &mut (impl RedisClient + ?Sized),
    player_id: &PlayerId,
    name: &str,
) -> Result<(), RedisError> {
    client.set(player_id, name).await?;
    client.expire(player_id, EXPIRE_TIME_SECONDS as i64).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::redis_client::test::MockRedisClient;

    mod endpoint_test {
        use super::*;
        use crate::{authentication::PlayerClaims, websocket::server::GameSessionManager};
        use actix_web::{http::StatusCode, test, web, App};
        use jwt_simple::{prelude::*, reexports::rand::prelude::*};

        #[actix_web::test]
        async fn test_register_player() {
            let player_name = "foo";

            // Mock Redis setup
            let mock_redis_client = MockRedisClient::default();

            // Auth setup
            let jwt_key = HS256Key::generate();
            let salt = thread_rng().next_u32();

            // GameServer setup
            let (_, server_handle) = GameSessionManager::new(mock_redis_client.clone());

            // AppState setup
            let app_state = AppState::new(mock_redis_client, jwt_key, salt, server_handle.clone());
            let jwt_key = app_state.jwt_key().to_owned();

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
            jwt_key
                .verify_token::<PlayerClaims>(&body.token, None)
                .unwrap();
        }
    }

    #[tokio::test]
    async fn test_write_player_id_name() {
        let player_id = PlayerId::new();
        let name = "foo";
        let mut client = MockRedisClient::default();
        let res = write_player_id_name(&mut client, &player_id, name).await;
        assert!(res.is_ok());
    }
}
