use crate::types::PlayerId;
use actix_web::{http::header, HttpRequest};
use jwt_simple::prelude::*;
use serde::{Deserialize, Serialize};

/// An expiration time of a JWT and a player ID.
/// TODO: Make this configurable via environment variables.
pub const EXPIRE_TIME_SECONDS: u64 = 3 * 60 * 60;

/// A claim about a player in a JWT.
#[derive(Serialize, Deserialize)]
pub struct PlayerClaims {
    /// A player ID
    player_id: PlayerId,
}

impl PlayerClaims {
    /// Creates a new [`PlayerClaims`].
    pub fn new(player_id: PlayerId) -> Self {
        PlayerClaims { player_id }
    }

    /// Returns a reference to the player id of this [`PlayerClaims`].
    pub fn player_id(&self) -> &PlayerId {
        &self.player_id
    }
}

/// Generates a JWT including a player ID.
///
/// # Arguments
///
/// * `key` - A JWT secret key.
/// * `player_id` - A player ID.
pub fn generate_jwt(key: &HS256Key, player_id: &PlayerId) -> Result<String, jwt_simple::Error> {
    let custom_claims = PlayerClaims::new(player_id.to_owned());
    let claims =
        Claims::with_custom_claims(custom_claims, Duration::from_secs(EXPIRE_TIME_SECONDS));
    key.authenticate(claims)
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
pub fn extract_bearer_token(req: &HttpRequest) -> Result<String, String> {
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
pub fn validate_jwt(key: &HS256Key, jwt: &str) -> Result<PlayerId, String> {
    match key.verify_token::<PlayerClaims>(jwt, None) {
        Ok(claims) => Ok(claims.custom.player_id().to_owned()),
        Err(err) => Err(format!("JWT validation failed: {}", err)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    mod async_test {
        use super::*;
        use actix_web::test;

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
    fn test_generate_jwt() {
        let key = HS256Key::generate();
        let player_id = PlayerId::new();
        let jwt = generate_jwt(&key, &player_id).unwrap();

        let claims = key.verify_token::<PlayerClaims>(&jwt, None).unwrap();
        assert_eq!(claims.custom.player_id, player_id);
    }

    #[test]
    fn test_validate_jwt() {
        // Prepare test JWT.
        let key = HS256Key::generate();
        let player_id = PlayerId::new();

        let claims = Claims::with_custom_claims(
            PlayerClaims::new(player_id.to_owned()),
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
}
