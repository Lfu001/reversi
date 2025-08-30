mod app_state;
mod authentication;
mod config;
mod routes;
mod services;
mod types;
mod websocket;

use crate::{
    app_state::AppState, config::config, services::redis_client::RealRedisClient,
    websocket::server::GameSessionManager,
};
use actix_web::{middleware, web, App, HttpServer};
use env_logger::Env;
use jwt_simple::{prelude::HS256Key, reexports::rand::prelude::*};
use std::env;
use tokio::{spawn, try_join};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(Env::default().default_filter_or("info"));

    let host = env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = env::var("PORT")
        .unwrap_or_else(|_| "8081".to_string())
        .parse::<u16>()
        .expect("PORT must be a valid u16");
    let redis_url = env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

    let redis_client = redis::Client::open(redis_url).expect("Failed to connect to Redis");
    let redis_client = RealRedisClient::new(redis_client).await;
    let jwt_key = HS256Key::generate();
    let salt = thread_rng().next_u32();
    let (game_server, server_handle) = GameSessionManager::new(redis_client.clone());
    let game_server = spawn(game_server.run());
    let app_state = web::Data::new(AppState::new(redis_client, jwt_key, salt, server_handle));

    let http_server = HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .configure(config)
            .wrap(middleware::Logger::default())
    })
    .bind((host, port))?
    .run();

    try_join!(http_server, async move { game_server.await.unwrap() })?;

    Ok(())
}
