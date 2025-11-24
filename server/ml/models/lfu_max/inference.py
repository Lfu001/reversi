from typing import Any, Callable, override

from inference_kit.api_model import DiskColor, Position, SuggestedPosition, TableState
from inference_kit.base_inference_handler import BaseInferenceHandler
from reversi import ReversiEnvironment


class InferenceHandler(BaseInferenceHandler):
    @override
    def load_model(self) -> Callable[..., Any]:
        return lambda x: x

    @override
    def prepare_input(self, table_state: TableState) -> TableState:
        return table_state

    @override
    def predict(self, data: TableState, model: Callable[..., Any]) -> TableState:
        return data

    @override
    def output(self, prediction: TableState) -> list[SuggestedPosition]:
        if not prediction.puttable_positions:
            return []

        best_score = -1
        best_positions = []

        # Convert to numpy for reversi library
        # prediction.to_numpy() returns (1, 4, 8, 8), we need (4, 8, 8)
        current_state_np = prediction.to_numpy()[0]

        is_dark_turn = prediction.turn == DiskColor.DARK

        for pos in prediction.puttable_positions:
            action = pos.row_idx * 8 + pos.col_idx

            # Get next state using reversi library
            try:
                next_state = ReversiEnvironment.get_next_state(current_state_np, action)

                # Count own stones in the next state
                # Channel 0: Dark, Channel 1: Light
                # If I was Dark, I want to count Dark stones (channel 0)
                # If I was Light, I want to count Light stones (channel 1)
                target_channel = 0 if is_dark_turn else 1
                own_count = next_state[target_channel].sum()

                if own_count > best_score:
                    best_score = own_count
                    best_positions = [pos]
                elif own_count == best_score:
                    best_positions.append(pos)
            except ValueError as e:
                # If the move is invalid for some reason, skip it
                # This shouldn't happen if puttable_positions is correct
                print(
                    f"Warning: Skipping invalid move {pos.row_idx},{pos.col_idx}: {e}"
                )
                continue

        # If no valid moves found (shouldn't happen), return empty list
        if not best_positions:
            return []

        # Select the first best position
        selected_pos = best_positions[0]

        return [
            SuggestedPosition(
                position=Position(row=pos.row, column=pos.column),
                confidence=1.0
                if (
                    pos.row_idx == selected_pos.row_idx
                    and pos.col_idx == selected_pos.col_idx
                )
                else 0.0,
            )
            for pos in prediction.puttable_positions
        ]
