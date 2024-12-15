mod app_state;
mod join;
mod routes;

use actix_web::{middleware, web, App, HttpServer};
use app_state::AppState;
use env_logger::Env;
use jwt_simple::prelude::HS256Key;
use routes::config;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(Env::default().default_filter_or("info"));
    HttpServer::new(|| {
        App::new()
            .app_data(web::Data::new(AppState::new(
                redis::Client::open("redis://127.0.0.1:6379").unwrap(),
                HS256Key::generate(),
            )))
            .configure(config)
            .wrap(middleware::Logger::default())
    })
    .bind(("127.0.0.1", 8081))?
    .run()
    .await
}
