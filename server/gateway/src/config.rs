use crate::routes::{
    game::game_ws, health::health, join::join_table, player_profile::update_profile,
    players::register_player,
};
use actix_web::web;

/// Configure routes.
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(web::resource("/health").route(web::get().to(health)));
    cfg.service(web::resource("/players").route(web::post().to(register_player)));
    cfg.service(web::resource("/playerProfile").route(web::put().to(update_profile)));
    cfg.service(web::resource("/join").route(web::post().to(join_table)));
    cfg.service(web::resource("/{table_id}").route(web::get().to(game_ws)));
}
