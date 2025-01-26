use crate::{
    app_state::AppState,
    authentication::{extract_bearer_token, validate_jwt},
    services::redis_client::RedisClient,
    websocket::handler,
};
use actix_web::{rt, web, Error, HttpRequest, HttpResponse};
use serde::Deserialize;

/// A path parameters.
#[derive(Deserialize)]
pub struct Info {
    /// A table ID.
    table_id: String,
}

/// Handshake and start WebSocket handler with heartbeats.
///
/// # Arguments
///
/// * `req` - The HTTP request of the WebSocket handshake.
/// * `stream` - The payload of the WebSocket.
/// * `info` - The path parameters.
/// * `app_state` - The application state.
pub async fn game_ws(
    req: HttpRequest,
    stream: web::Payload,
    info: web::Path<Info>,
    app_state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    // Validate JWT and extract player ID.
    let jwt = extract_bearer_token(&req);
    if let Err(err) = jwt {
        log::error!("{}", err);
        return Ok(HttpResponse::Unauthorized().finish());
    }
    let jwt = jwt.unwrap();
    let player_id = validate_jwt(app_state.jwt_key(), &jwt);
    if let Err(err) = player_id {
        log::error!("{}", err);
        return Ok(HttpResponse::Unauthorized().finish());
    }
    let player_id = player_id.unwrap();

    //  Check if the table exists.
    match is_table_exist(&mut *app_state.redis_client().await, &info.table_id).await {
        Ok(exists) => {
            if !exists {
                log::error!("Table \"{}\" does not exist.", info.table_id);
                return Ok(HttpResponse::NotFound().finish());
            }
        }
        Err(err) => {
            log::error!("{}", err);
            return Ok(HttpResponse::InternalServerError().finish());
        }
    }

    // Spawn WebSocket handler.
    let (res, session, stream) = actix_ws::handle(&req, stream)?;
    rt::spawn(handler::game_ws(
        player_id,
        info.table_id.clone(),
        (**app_state).game_server().clone(),
        session,
        stream,
    ));

    Ok(res)
}

/// Checks if the table exists.
///
/// # Arguments
///
/// * `client` - The Redis client.
/// * `table_id` - The table ID.
async fn is_table_exist(
    client: &mut (impl RedisClient + ?Sized),
    table_id: &str,
) -> Result<bool, String> {
    match client.exists(table_id).await {
        Ok(exists) => Ok(exists),
        Err(err) => Err(err.to_string()),
    }
}

// /// Checks if the player is in the table.
// ///
// /// # Arguments
// ///
// /// * `client` - The Redis client.
// /// * `player_id` - The player ID.
// /// * `table_id` - The table ID.
// fn is_player_in_table(
//     client: &dyn RedisClient,
//     player_id: &str,
//     table_id: &str,
// ) -> Result<bool, String> {
//     match client.json_arr_index(table_id, "$.players", player_id) {
//         Ok(index) => Ok(index >= 0),
//         Err(err) => Err(err.to_string()),
//     }
// }
