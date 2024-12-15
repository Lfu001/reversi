use crate::join::join_table;
use actix_web::web;

/// Configure routes.
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(web::resource("/join").route(web::post().to(join_table)));
}
