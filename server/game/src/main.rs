mod config;
mod messages;
mod routes;

use crate::config::config;
use actix_web::{middleware, App, HttpServer};
use env_logger::Env;
use std::env;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(Env::default().default_filter_or("info"));

    let host = env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = env::var("PORT")
        .unwrap_or_else(|_| "8080".to_string())
        .parse::<u16>()
        .expect("PORT must be a valid u16");

    HttpServer::new(|| {
        App::new()
            .configure(config)
            .wrap(middleware::Logger::default())
    })
    .bind((host, port))?
    .run()
    .await
}
