<template>
	<div class="board">
		<div class="board-inner">
			<div
				v-for="(row, rowIndex) in board"
				:key="'row-' + rowIndex"
				class="row"
			>
				<div
					v-for="(cell, colIndex) in row"
					:key="'cell-' + rowIndex + '-' + colIndex"
					class="cell"
					:class="{ dot: isDotPosition(rowIndex, colIndex), dark: cell === 1, light: cell === 2}"
					@click="placeStone(rowIndex, colIndex)"
				></div>
			</div>
		</div>
	</div>
</template>

<script setup lang="ts">
import { reactive } from 'vue';

const board = reactive(
	Array(8)
	.fill(null)
	.map(() => Array(8).fill(null))
);

// initial placement
board[3][3] = 2;	// light
board[3][4] = 1;	// dark
board[4][3] = 1;	// dark
board[4][4] = 2;	// light

// dot position
const isDotPosition = (row: number, col: number) => {
	const dotPositions = [
		[2, 2],
		[2, 6],
		[6, 2],
		[6, 6],
	];
	return dotPositions.some(([dotRow, dotCol]) => dotRow === row && dotCol === col);
};

const placeStone = (row: number, col: number) => {
	// filled cell
	if (board[row][col] != 0) return;

	// TODO : apply logic
	// anyway put dark stone
	board[row][col] = 1;
}
</script>

<style scoped>
.board {
	background-color: #4a3222;
	padding: 30px;
	border-radius: 15px;
	box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);
	display: flex;
	justify-content: center;
	align-items: center;
	width: 336px;	/* (1cell:40px + 2border:2px) * 8 */
	height: 336px;	/* (1cell:40px + 2border:2px) * 8 */
}

.board-inner {
	background-color:green;
	padding: 10px;
	border-radius: 10px;
	display: grid;
	grid-template-rows: repeat(8, 1fr);
	width: 336px;	/* (1cell:40px + 2border:2px) * 8 */
	height: 336px;	/* (1cell:40px + 2border:2px) * 8 */
}

.row {
	display: grid;
	grid-template-columns: repeat(8, 1fr);
}

.cell {
	width: 40px;
	height: 40px;
	display: flex;
	align-items: center;
	justify-content: center;
	position: relative;
	border: 1px solid black;
}

.cell::before {
	content: '';
	position: absolute;
	width: 6px;
	height: 6px;
	border-radius: 50%;
	background-color: black;
	top: -1px;
	left: -1px;
	transform: translate(-50%, -50%);
	opacity: 0;	/* invisible by default */
}

.cell.dot::before {
	opacity: 1;	/* visible */
}

.cell.dark::before,
.cell.light::before {
	content: '';
	width: 80%;
	height: 80%;
	border-radius: 50%;
	position: absolute;
}

.cell.dark::before {
	background-color: black;
}

.cell.light::before {
	background-color: white;
}
</style>