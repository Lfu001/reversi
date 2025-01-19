use crate::api::{game::game_ws, join::join_table, players::register_player};
use actix_web::web;

/// Configure routes.
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(web::resource("/players").route(web::post().to(register_player)));
    cfg.service(web::resource("/join").route(web::post().to(join_table)));
    cfg.service(web::resource("/{table_id}").route(web::get().to(game_ws)));
}
