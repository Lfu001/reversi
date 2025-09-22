use crate::{
    authentication::{extract_bearer_token, validate_jwt},
    AppState,
};
use actix_web::{web, HttpRequest, HttpResponse, Responder};
use serde::{Deserialize, Serialize};

/// A request to update a player's profile.
#[derive(Serialize, Deserialize)]
pub struct UpdateProfileRequest {
    /// The player's new avatar URL.
    avatar_url: String,
}

/// Updates a player's profile.
///
/// # Arguments
///
/// * `req` - An HTTP request.
/// * `form` - A form containing the updated profile information.
/// * `app_state` - An application state containing a Redis client and a JWT key.
pub async fn update_profile(
    req: HttpRequest,
    form: web::Form<UpdateProfileRequest>,
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
    let player_id = match validate_jwt(app_state.jwt_key(), &jwt) {
        Ok(player_id) => player_id,
        Err(err) => {
            log::error!("{}", err);
            return HttpResponse::Unauthorized().finish();
        }
    };

    // Update player profile.
    let _: () = app_state
        .redis_client()
        .await
        .json_set(
            &player_id,
            ".avatar_url",
            &serde_json::json!(form.avatar_url),
        )
        .await
        .unwrap();

    HttpResponse::Ok().finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        authentication::PlayerClaims, services::redis_client::test::MockRedisClient,
        types::PlayerId, websocket::server::GameSessionManager,
    };
    use actix_web::{
        http::{header, StatusCode},
        test, App,
    };
    use jwt_simple::{prelude::*, reexports::rand::prelude::*};

    #[actix_web::test]
    async fn test_update_profile() {
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
                .service(web::resource("/player_profile").route(web::put().to(update_profile))),
        )
        .await;

        let custom_claims = PlayerClaims::new(PlayerId::new());
        let claims = Claims::with_custom_claims(custom_claims, Duration::from_secs(60));
        let jwt = jwt_key.authenticate(claims).unwrap();

        let form = UpdateProfileRequest {
            avatar_url: "bar".to_string(),
        };

        let req = test::TestRequest::put()
            .uri("/player_profile")
            .insert_header((header::AUTHORIZATION, format!("Bearer {}", jwt)))
            .insert_header((header::CONTENT_TYPE, "application/x-www-form-urlencoded"))
            .set_form(form)
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
