use crate::{
    app_state::AppState, authentication::validate_jwt, services::redis_client::RedisClient,
    types::TableId, websocket::handler,
};
use actix_web::{rt, web, Error, HttpRequest, HttpResponse};
use serde::Deserialize;

/// A path parameters.
#[derive(Deserialize)]
pub struct Info {
    /// A table ID.
    table_id: String,
}

#[derive(Deserialize)]
pub struct GameQuery {
    token: String,
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
    query: web::Query<GameQuery>,
    app_state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    // Validate JWT and extract player ID.
    // WARNING:
    // Note that it is insecure to extract or send a token in the query parameter.
    // It is impossible to send the token in the authorization header because this endpoint is a WebSocket handshake,
    // and on the client WebSocket() JavaScript interface does not provide setting the authorization header.
    // For development purposes, we allow the token to be sent as a query parameter as a workaround.
    // However, in a production environment, it is crucial to implement secure logic to send the token
    // in the first message after the WebSocket connection is established.
    // This ensures that the token is transmitted securely and cannot be easily intercepted or tampered with.
    // TODO: Implement secure logic to send the token in the first message after the WebSocket connection is established.
    let jwt = &query.token;
    let player_id = validate_jwt(app_state.jwt_key(), jwt);
    if let Err(err) = player_id {
        log::error!("{}", err);
        return Ok(HttpResponse::Unauthorized().finish());
    }
    let player_id = player_id.unwrap();

    //  Check if the table exists.
    let table_id = TableId::new(info.table_id.to_owned());
    match is_table_exist(&mut *app_state.redis_client().await, &table_id).await {
        Ok(exists) => {
            if !exists {
                log::error!("Table \"{}\" does not exist.", *table_id);
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
        table_id,
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
    table_id: &TableId,
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
