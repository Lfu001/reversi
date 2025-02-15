use crate::{
    app_state::AppState,
    authentication::{extract_bearer_token, validate_jwt},
    services::redis_client::{GameTable, RedisClient},
    types::{PlayerId, TableId},
};
use actix_web::{http::header, web, HttpRequest, HttpResponse, Responder};
use base64::{engine::general_purpose::URL_SAFE, Engine as _};
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
    let res =
        add_player_to_table(&mut *app_state.redis_client().await, &player_id, &table_id).await;
    if let Err(err) = res {
        log::error!("{}", err);
        return HttpResponse::InternalServerError().finish();
    }

    HttpResponse::Created()
        .insert_header((header::LOCATION, format!("/{}", *table_id)))
        .finish()
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
fn convert_password_to_table_id(password: &str, salt: u32) -> TableId {
    let data = format!("{}{}", password, salt);

    let mut hasher = Shake128::default();
    hasher.update(data.as_bytes());
    let mut reader = hasher.finalize_xof();
    let mut digest = [0u8; 10];
    reader.read(&mut digest);

    TableId::new(URL_SAFE.encode(digest))
}

/// Add a player to a table. If the table is not found, a new table is created.
///
/// # Arguments
///
/// * `client` - A Redis client.
/// * `player_id` - A player ID.
/// * `table_id` - A table ID.
async fn add_player_to_table(
    client: &mut (impl RedisClient + ?Sized),
    player_id: &PlayerId,
    table_id: &TableId,
) -> Result<(), String> {
    let exists = client.exists(table_id).await;
    if let Err(err) = exists {
        return Err(err.to_string());
    }
    let exists = exists.unwrap();
    if exists {
        let game_table = client.json_get(table_id, "$").await.unwrap();
        if game_table.players().len() >= 2 {
            return Err(format!("Table {:?} is full.", table_id));
        }
        if let Err(err) = client
            .json_arr_append(table_id, "$.players", &player_id.to_string())
            .await
        {
            return Err(err.to_string());
        }
    } else {
        let url = env::var(NEW_GAME_URL_ENV_VAR).unwrap_or("http://127.0.0.1:8080/new".to_string());
        let response = ureq::get(&url).call().unwrap().into_string().unwrap();
        let json = serde_json::json!(&GameTable::new(vec![player_id.to_string()], response));
        let _: () = client.json_set(table_id, "$", &json).await.unwrap();
        let _: () = client.expire(table_id, 1800).await.unwrap();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::redis_client::test::MockRedisClient;
    use serial_test::serial;

    mod endpoint_test {
        use super::*;
        use crate::{authentication::PlayerClaims, websocket::server::GameSessionManager};
        use actix_web::{http, test, App};
        use jwt_simple::prelude::*;

        #[actix_web::test]
        #[serial]
        async fn test_join() {
            let mock_redis_client = MockRedisClient::default();
            let (_, server_handle) = GameSessionManager::new(mock_redis_client.clone());
            let app_state = AppState::new(mock_redis_client, server_handle.clone());
            let jwt_key = app_state.jwt_key().to_owned();
            let app = test::init_service(
                App::new()
                    .app_data(web::Data::new(app_state))
                    .service(web::resource("/join").route(web::post().to(join_table))),
            )
            .await;

            let custom_claims = PlayerClaims::new(PlayerId::new());
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
    }

    #[test]
    fn test_convert_password_to_table_id() {
        let password = "password";
        let salt = 12345;
        let table_id = convert_password_to_table_id(password, salt);
        assert_eq!(*table_id, "03NHKRfdHqCrvw==");

        let password = "foo";
        let salt = 98765;
        let table_id = convert_password_to_table_id(password, salt);
        assert_eq!(*table_id, "GkElquPDNXk6sA==");
    }

    #[tokio::test]
    #[serial]
    async fn test_add_player_to_table_create_new() {
        let mut client = MockRedisClient {
            exists_result: String::from("0"),
            ..MockRedisClient::default()
        };
        let player_id = PlayerId::new();
        let table_id = TableId::new(String::from("table1"));
        let mut server = mockito::Server::new_async().await;
        let mock = setup_mock_async(&mut server).await;

        let result = add_player_to_table(&mut client, &player_id, &table_id).await;
        assert!(result.is_ok());
        mock.assert();
    }

    #[tokio::test]
    async fn test_add_player_to_table_existing() {
        let table_id = TableId::new(String::from("table1"));
        let player1 = PlayerId::new();
        let player2 = PlayerId::new();
        // 1 player already joining the table.
        {
            let mut client = MockRedisClient {
                exists_result: String::from("1"),
                json_get_result: format!(
                    "{{\"players\":[\"{}\"], \"game_state\": \"\"}}",
                    *player1
                ),
                ..MockRedisClient::default()
            };

            let result = add_player_to_table(&mut client, &player2, &table_id).await;
            assert!(result.is_ok());
        }
        // 2 player already joining the table.
        {
            let mut client = MockRedisClient {
                exists_result: String::from("1"),
                json_get_result: format!(
                    "{{\"players\":[\"{}\", \"{}\"], \"game_state\": \"\"}}",
                    *player1, *player2
                ),
                ..MockRedisClient::default()
            };
            let player3 = PlayerId::new();

            let result = add_player_to_table(&mut client, &player3, &table_id).await;
            assert!(result.is_err());
        }
    }

    async fn setup_mock_async(server: &mut mockito::ServerGuard) -> mockito::Mock {
        env::set_var(NEW_GAME_URL_ENV_VAR, format!("{}/new", server.url()));
        server
            .mock("GET", "/new")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("{\"table\":{\"board\":[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,\"Light\",\"Dark\",null,null,null,null,null,null,\"Dark\",\"Light\",null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\"turn\":\"Dark\",\"history\":[]},\"puttable_positions\":[{\"row\":\"Three\",\"column\":\"D\"},{\"row\":\"Four\",\"column\":\"C\"},{\"row\":\"Five\",\"column\":\"F\"},{\"row\":\"Six\",\"column\":\"E\"}]}}")
            .create_async()
            .await
    }
}
