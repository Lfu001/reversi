use crate::{
    app_state::AppState, services::redis_client::RedisClient, types::TableId, websocket::handler,
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
