use crate::players::register_player;
use actix_web::web;

/// Configure routes.
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(web::resource("/players").route(web::post().to(register_player)));
}
