mod api;
mod app_state;
mod authentication;
mod routes;
mod services;
mod types;
mod websocket;

use crate::{
    app_state::AppState, routes::config, services::redis_client::RealRedisClient,
    websocket::server::GameSessionManager,
};
use actix_web::{middleware, web, App, HttpServer};
use env_logger::Env;
use jwt_simple::{prelude::HS256Key, reexports::rand::prelude::*};
use tokio::{spawn, try_join};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(Env::default().default_filter_or("info"));

    let client = redis::Client::open("redis://127.0.0.1:6379").unwrap();
    let client = RealRedisClient::new(client).await;
    let jwt_key = HS256Key::generate();
    let salt = thread_rng().next_u32();
    let (game_server, server_handle) = GameSessionManager::new(client.clone());
    let game_server = spawn(game_server.run());

    let http_server = HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(AppState::new(
                client.clone(),
                jwt_key.clone(),
                salt,
                server_handle.clone(),
            )))
            .configure(config)
            .wrap(middleware::Logger::default())
    })
    .bind(("127.0.0.1", 8081))?
    .run();

    try_join!(http_server, async move { game_server.await.unwrap() })?;

    Ok(())
}
