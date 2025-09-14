use crate::{
    app_state::AppState,
    authentication::{extract_bearer_token, validate_jwt},
    services::redis_client::RedisClient,
    types::TableId,
};
use actix_web::{http::header, web, HttpRequest, HttpResponse, Responder};
use base64::{engine::general_purpose::URL_SAFE, Engine as _};
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
    let jwt = match extract_bearer_token(&req) {
        Ok(jwt) => jwt,
        Err(err) => {
            log::error!("{}", err);
            return HttpResponse::Unauthorized().finish();
        }
    };

    // Validate JWT.
    match validate_jwt(app_state.jwt_key(), &jwt) {
        Ok(player_id) => player_id,
        Err(err) => {
            log::error!("{}", err);
            return HttpResponse::Unauthorized().finish();
        }
    };

    // Convert the password to table ID.
    let table_id = convert_password_to_table_id(&form.password, app_state.salt());

    // Create the table.
    match create_table_if_not_exist(&mut *app_state.redis_client().await, &table_id).await {
        Ok(_) => {}
        Err(err) => {
            log::error!("{}", err);
            return HttpResponse::InternalServerError().finish();
        }
    }

    HttpResponse::Created()
        .insert_header((header::LOCATION, format!("/{}", *table_id)))
        .insert_header((
            header::ACCESS_CONTROL_EXPOSE_HEADERS,
            header::LOCATION.to_string(),
        ))
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

/// Creates a new table if it does not exist. If already exists, does nothing.
///
/// # Arguments
///
/// * `client` - A Redis client.
/// * `table_id` - A table ID.
async fn create_table_if_not_exist(
    client: &mut (impl RedisClient + ?Sized),
    table_id: &TableId,
) -> Result<(), String> {
    let exists = match client.exists(table_id).await {
        Ok(value) => value,
        Err(err) => return Err(err.to_string()),
    };
    if !exists {
        let _: () = client
            .json_set(table_id, "$", &serde_json::json!({}))
            .await
            .unwrap();
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
        use crate::{
            authentication::PlayerClaims, types::PlayerId, websocket::server::GameSessionManager,
        };
        use actix_web::{http, test, App};
        use jwt_simple::{prelude::*, reexports::rand::prelude::*};

        #[actix_web::test]
        #[serial]
        async fn test_join() {
            let mock_redis_client = MockRedisClient::default();
            let jwt_key = HS256Key::generate();
            let salt = thread_rng().next_u32();
            let (_, server_handle) =
                GameSessionManager::new(mock_redis_client.clone(), jwt_key.clone());
            let app_state = AppState::new(mock_redis_client, jwt_key, salt, server_handle.clone());
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

            let req = test::TestRequest::post()
                .uri("/join")
                .insert_header((header::AUTHORIZATION, format!("Bearer {}", jwt)))
                .insert_header((header::CONTENT_TYPE, "application/x-www-form-urlencoded"))
                .set_form([("password", "bar")])
                .to_request();

            let resp = test::call_service(&app, req).await;
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
    async fn test_create_table_not_exist() {
        let mut client = MockRedisClient {
            exists_result: String::from("0"),
            ..MockRedisClient::default()
        };
        let table_id = TableId::new(String::from("table1"));

        let result = create_table_if_not_exist(&mut client, &table_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_create_table_already_exist() {
        let table_id = TableId::new(String::from("table1"));

        let mut client = MockRedisClient {
            exists_result: String::from("1"),
            ..MockRedisClient::default()
        };

        let result = create_table_if_not_exist(&mut client, &table_id).await;
        assert!(result.is_ok());
    }
}
