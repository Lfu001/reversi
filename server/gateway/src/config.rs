use crate::routes::health::health;
use crate::routes::join::join_table;
use crate::routes::players::register_player;
use actix_web::web;

/// Configure routes.
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(web::resource("/health").route(web::get().to(health)));
    cfg.service(web::resource("/players").route(web::post().to(register_player)));
    cfg.service(web::resource("/join").route(web::post().to(join_table)));
}
