<template>
  <div class="board-container">
    <div class="board-inner">
      <div class="board-grid">
        <div
          v-for="(c, index) in props.board"
          :key="index"
          class="square"
          :class="{ dot: isDotPosition(index) }"
        >
          <DiskComponent
            :color="c"
            @click="placeStone(index)"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { DiskColor } from '~/types/DiskColor'

const props = defineProps<{
  board: Array<DiskColor | null>
}>()

const isDotPosition = (index: number) => {
  const dotPositions = [
    18,
    22,
    50,
    54,
  ]
  return dotPositions.some(i => i === index)
}

// put down disc
const placeStone = (index: number) => {
  // filled cell
  if (props.board[index] != null) return

  // TODO : apply logic
}
</script>

<style scoped>
.board-container {
  background-color: #4a3222;
  padding: 30px;
  border-radius: 15px;
  box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);
  display: flex;
  justify-content: center;
  align-items: center;
  width: 380px;  /* (1cell:40px + 2border:2px) * 8 */
  height: 380px;  /* (1cell:40px + 2border:2px) * 8 */
  /* disable selection */
  user-select: none;
  -webkit-user-select: none;  /* Chrome, Safari */
  -moz-user-select: none;    /* Firefox */
}

.board-inner {
  background: radial-gradient(#50aa50, #2d642d);
  padding: 10px;
  border-radius: 10px;
}

.board-grid {
  border: 1px solid black;
  display: grid;
  grid-template-rows: repeat(8, 1fr);
  grid-template-columns: repeat(8, 1fr);
  width: 320px;  /* (1cell:40px + 2border:2px) * 8 */
  height: 320px;  /* (1cell:40px + 2border:2px) * 8 */
}

.square {
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  border: 1px solid black;
}

.square.dot::before {
  content: '';
  position: absolute;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background-color: black;
  top: -1px;
  left: -1px;
  transform: translate(-50%, -50%);
  opacity: 1;
}
</style>
