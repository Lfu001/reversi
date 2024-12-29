use crate::players::PlayerClaims;
use crate::redis_client::GameTable;
use crate::{app_state::AppState, redis_client::RedisClient};
use actix_web::{http::header, web, HttpRequest, HttpResponse, Responder};
use base64::{engine::general_purpose::URL_SAFE, Engine as _};
use jwt_simple::prelude::*;
use serde::{Deserialize, Serialize};
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake128,
};

/// A join table request message.
#[derive(Serialize, Deserialize)]
pub struct JoinTableRequest {
    /// A password of the table.
    password: String,
}

/// Joins a player to a table.
pub async fn join_table(
    req: HttpRequest,
    form: web::Form<JoinTableRequest>,
    app_state: web::Data<AppState>,
) -> impl Responder {
    // Extract JWT from Authorization header.
    let res = extract_bearer_token(&req);
    if let Err(err) = res {
        log::error!("{}", err);
        return HttpResponse::Unauthorized().finish();
    }
    let jwt = res.unwrap();

    // Validate JWT and extract player ID.
    let res = validate_jwt(app_state.jwt_key(), &jwt);
    if let Err(err) = res {
        log::error!("{}", err);
        return HttpResponse::Unauthorized().finish();
    }
    let player_id = res.unwrap();

    // Convert the password to table ID.
    let table_id = convert_password_to_table_id(&form.password, app_state.salt());

    // Add the player to the table.
    let res = add_player_to_table(app_state.redis_client_mut(), &player_id, &table_id);
    if let Err(err) = res {
        log::error!("{}", err);
        return HttpResponse::InternalServerError().finish();
    }

    HttpResponse::Created()
        .insert_header((header::LOCATION, format!("/{}", table_id)))
        .finish()
}

/// Extracts a Bearer token from a Authorization header.
///
/// # Arguments
///
/// * `req` - A HTTP request.
///
/// # Returns
///
/// A Bearer token if it exists, or an error message otherwise.
fn extract_bearer_token(req: &HttpRequest) -> Result<String, String> {
    match req.headers().get(header::AUTHORIZATION) {
        Some(bearer) => match bearer.to_str().unwrap().strip_prefix("Bearer ") {
            Some(bearer) => Ok(String::from(bearer)),
            None => Err(String::from("Invalid Bearer token.")),
        },
        None => Err(String::from("Missing Bearer token.")),
    }
}

/// Validates a JWT and returns the player ID.
///
/// # Arguments
///
/// * `key` - A JWT secret key.
/// * `jwt` - A JWT.
///
/// # Returns
///
/// A player ID if the JWT is valid, or an error message otherwise.
fn validate_jwt(key: &HS256Key, jwt: &str) -> Result<String, String> {
    match key.verify_token::<PlayerClaims>(jwt, None) {
        Ok(claims) => Ok(claims.custom.player_id().to_owned()),
        Err(err) => Err(format!("JWT validation failed: {}", err)),
    }
}

/// Converts a password to a table ID.
///
/// # Arguments
///
/// * `password` - A password of the table.
/// * `salt` - A salt for password hashing.
///
/// # Returns
///
/// A table ID.
fn convert_password_to_table_id(password: &str, salt: u32) -> String {
    let data = format!("{}{}", password, salt);

    let mut hasher = Shake128::default();
    hasher.update(data.as_bytes());
    let mut reader = hasher.finalize_xof();
    let mut digest = [0u8; 10];
    reader.read(&mut digest);

    URL_SAFE.encode(digest)
}

/// Add a player to a table. If the table is not found, a new table is created.
///
/// # Arguments
///
/// * `client` - A Redis client.
/// * `player_id` - A player ID.
/// * `table_id` - A table ID.
fn add_player_to_table(
    client: &dyn RedisClient,
    player_id: &str,
    table_id: &str,
) -> Result<(), String> {
    let key = format!("table:{}", table_id);
    let exists = client.exists(&key);
    if let Err(err) = exists {
        return Err(err.to_string());
    }
    let exists = exists.unwrap();
    if exists {
        let game_table = client.json_get(&key, "$").unwrap();
        if game_table.players().len() >= 2 {
            return Err(format!("Table {} is full.", table_id));
        }
        if let Err(err) = client.json_arr_append(table_id, "$.players", player_id) {
            return Err(err.to_string());
        }
    } else {
        let response = ureq::get("http://127.0.0.1:8080/new")
            .call()
            .unwrap()
            .into_string()
            .unwrap();
        let json = serde_json::json!(&GameTable::new(vec![player_id.to_string()], response));
        let _: () = client.json_set(table_id, "$", &json).unwrap();
        let _: () = client.expire(table_id, 1800).unwrap();
    }

    Ok(())
}
