mod app_state;
mod config;
mod redis_client;
mod routes;

use actix_web::{middleware, web, App, HttpServer};
use app_state::AppState;
use config::config;
use env_logger::Env;
use redis_client::RealRedisClient;
use std::env;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(Env::default().default_filter_or("info"));

    let host = env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = env::var("PORT")
        .unwrap_or_else(|_| "8081".to_string())
        .parse::<u16>()
        .expect("PORT must be a valid u16");
    let redis_url = env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(AppState::new(Box::new(
                RealRedisClient::new(redis::Client::open(redis_url.clone()).unwrap()),
            ))))
            .configure(config)
            .wrap(middleware::Logger::default())
    })
    .bind((host, port))?
    .run()
    .await
}
