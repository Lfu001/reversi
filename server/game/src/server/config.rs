use super::routes::{health::health, new::create_new_table, step::step_table};
use actix_web::web;

/// Configure routes.
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(web::resource("/health").route(web::get().to(health)));
    cfg.service(web::resource("/new").route(web::get().to(create_new_table)));
    cfg.service(web::resource("/step").route(web::post().to(step_table)));
}
