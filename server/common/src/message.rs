use crate::{
    action::Action, disk::DiskColor, judge::JudgeResult, position::Position, table::Table,
};
use serde::{Deserialize, Serialize};

/// A request message to execute an action on the table.
#[derive(Serialize, Deserialize)]
pub struct StepRequestMessage {
    /// A table of the game.
    table: Table,
    /// An action to execute.
    action: Action,
}

impl StepRequestMessage {
    /// Creates a new [`StepRequestMessage`].
    pub fn new(table: Table, action: Action) -> Self {
        Self { table, action }
    }

    /// Returns a reference to the table of this [`StepRequestMessage`].
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Returns a reference to the action of this [`StepRequestMessage`].
    pub fn action(&self) -> &Action {
        &self.action
    }
}

/// A response message from the game endpoint.
#[derive(Serialize, Deserialize, Clone)]
pub struct StateResponseMessage {
    /// A table of the game.
    table: Table,
    /// Next puttable positions.
    puttable_positions: Vec<Position>,
    /// A player colors.
    #[serde(skip_serializing_if = "Option::is_none")]
    player_colors: Option<Vec<DiskColor>>,
    /// A result of the game.
    #[serde(skip_serializing_if = "Option::is_none")]
    judge_result: Option<JudgeResult>,
}

impl StateResponseMessage {
    /// Creates a new [`StateResponseMessage`].
    pub fn new(
        table: Table,
        puttable_positions: Vec<Position>,
        player_colors: Option<Vec<DiskColor>>,
        judge_result: Option<JudgeResult>,
    ) -> Self {
        Self {
            table,
            puttable_positions,
            player_colors,
            judge_result,
        }
    }

    /// Returns a reference to the table of this [`StateResponseMessage`].
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Returns a reference to the puttable positions of this [`StateResponseMessage`].
    pub fn puttable_positions(&self) -> &Vec<Position> {
        &self.puttable_positions
    }

    /// Returns a reference to the judge result of this [`StateResponseMessage`].
    pub fn judge_result(&self) -> &Option<JudgeResult> {
        &self.judge_result
    }

    /// Returns a reference to the player colors of this [`StateResponseMessage`].
    pub fn player_colors(&self) -> &Option<Vec<DiskColor>> {
        &self.player_colors
    }

    /// Sets the player colors of this [`StateResponseMessage`].
    pub fn set_player_colors(&mut self, player_colors: Vec<DiskColor>) {
        self.player_colors = Some(player_colors);
    }
}
