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
use std::env;

/// Environment variable name for the URL of the `/new` api endpoint of the game server.
const NEW_GAME_URL_ENV_VAR: &str = "NEW_GAME_API_URL";

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
        let url = env::var(NEW_GAME_URL_ENV_VAR).unwrap_or("http://127.0.0.1:8080/new".to_string());
        let response = ureq::get(&url).call().unwrap().into_string().unwrap();
        let json = serde_json::json!(&GameTable::new(vec![player_id.to_string()], response));
        let _: () = client.json_set(table_id, "$", &json).unwrap();
        let _: () = client.expire(table_id, 1800).unwrap();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::redis_client::test::MockRedisClient;
    use serial_test::serial;

    #[derive(Serialize, Deserialize)]
    struct CustomClaims {
        player_id: String,
    }

    mod endpoint_test {
        use super::*;
        use actix_web::{
            http::{self, header},
            test, App,
        };

        #[actix_web::test]
        #[serial]
        async fn test_join() {
            let mock_redis_client = MockRedisClient::default();
            let app_state = AppState::new(Box::new(mock_redis_client));
            let jwt_key = app_state.jwt_key().to_owned();
            let app = test::init_service(
                App::new()
                    .app_data(web::Data::new(app_state))
                    .service(web::resource("/join").route(web::post().to(join_table))),
            )
            .await;

            let custom_claims = CustomClaims {
                player_id: String::from("foo"),
            };
            let claims = Claims::with_custom_claims(custom_claims, Duration::from_secs(60));
            let jwt = jwt_key.authenticate(claims).unwrap();

            let mut server = mockito::Server::new_async().await;
            let mock = setup_mock_async(&mut server).await;

            let req = test::TestRequest::post()
                .uri("/join")
                .insert_header((header::AUTHORIZATION, format!("Bearer {}", jwt)))
                .insert_header((header::CONTENT_TYPE, "application/x-www-form-urlencoded"))
                .set_form([("password", "bar")])
                .to_request();

            let resp = test::call_service(&app, req).await;
            mock.assert_async().await;
            assert_eq!(resp.status(), http::StatusCode::CREATED);
        }

        #[test]
        async fn test_extract_bearer_token() {
            let req = test::TestRequest::default()
                .insert_header((header::AUTHORIZATION, "Bearer test_token"))
                .to_http_request();

            let token = extract_bearer_token(&req).unwrap();
            assert_eq!(token, "test_token");
        }

        #[test]
        async fn test_extract_bearer_token_missing() {
            let req = test::TestRequest::default().to_http_request();
            assert!(extract_bearer_token(&req).is_err());
        }

        #[test]
        async fn test_extract_bearer_token_invalid() {
            let req = test::TestRequest::default()
                .insert_header((header::AUTHORIZATION, "foo bar"))
                .to_http_request();
            assert!(extract_bearer_token(&req).is_err());
        }
    }

    #[test]
    fn test_validate_jwt() {
        // Prepare test JWT.
        let key = HS256Key::generate();
        let player_id = "player1";

        let claims = Claims::with_custom_claims(
            CustomClaims {
                player_id: player_id.to_owned(),
            },
            Duration::from_secs(10),
        );
        let jwt = key.authenticate(claims).unwrap();

        // Valid JWT.
        let result = validate_jwt(&key, &jwt).unwrap();
        assert_eq!(result, player_id);

        // Invalid JWT.
        let invalid_jwt = "invalid_jwt";
        let result = validate_jwt(&key, invalid_jwt);
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_password_to_table_id() {
        let password = "password";
        let salt = 12345;
        let table_id = convert_password_to_table_id(password, salt);
        assert_eq!(table_id, "03NHKRfdHqCrvw==");

        let password = "foo";
        let salt = 98765;
        let table_id = convert_password_to_table_id(password, salt);
        assert_eq!(table_id, "GkElquPDNXk6sA==");
    }

    #[test]
    #[serial]
    fn test_add_player_to_table_create_new() {
        let client = MockRedisClient {
            exists_result: String::from("0"),
            ..MockRedisClient::default()
        };
        let player_id = "player1";
        let table_id = "table1";
        let mut server = mockito::Server::new();
        let mock = setup_mock(&mut server);

        let result = add_player_to_table(&client, player_id, table_id);
        assert!(result.is_ok());
        mock.assert();
    }

    #[test]
    fn test_add_player_to_table_existing() {
        let table_id = "table1";
        // 1 player already joining the table.
        {
            let client = MockRedisClient {
                exists_result: String::from("1"),
                json_get_result: String::from("{\"players\":[\"player1\"], \"game_state\": \"\"}"),
                ..MockRedisClient::default()
            };
            let player_id = "player2";

            let result = add_player_to_table(&client, player_id, table_id);
            assert!(result.is_ok());
        }
        // 2 player already joining the table.
        {
            let client = MockRedisClient {
                exists_result: String::from("1"),
                json_get_result: String::from(
                    "{\"players\":[\"player1\", \"player2\"], \"game_state\": \"\"}",
                ),
                ..MockRedisClient::default()
            };
            let player_id = "player3";

            let result = add_player_to_table(&client, player_id, table_id);
            assert!(result.is_err());
        }
    }

    fn _setup_mock(server: &mut mockito::Server) -> mockito::Mock {
        env::set_var(NEW_GAME_URL_ENV_VAR, format!("{}/new", server.url()));
        server
        .mock("GET", "/new")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body("{\"table\":{\"board\":[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,\"Light\",\"Dark\",null,null,null,null,null,null,\"Dark\",\"Light\",null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\"turn\":\"Dark\",\"history\":[]},\"puttable_positions\":[{\"row\":\"Three\",\"column\":\"D\"},{\"row\":\"Four\",\"column\":\"C\"},{\"row\":\"Five\",\"column\":\"F\"},{\"row\":\"Six\",\"column\":\"E\"}]}}")
    }

    fn setup_mock(server: &mut mockito::Server) -> mockito::Mock {
        _setup_mock(server).create()
    }

    async fn setup_mock_async(server: &mut mockito::ServerGuard) -> mockito::Mock {
        _setup_mock(server).create_async().await
    }
}
