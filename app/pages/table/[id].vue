<template>
  <div class="flex min-h-screen flex-col bg-[#e9c983] p-4">
    <div class="container mx-auto flex flex-1 flex-col gap-8 md:flex-row">
      <!-- Left side - Board -->
      <div class="flex flex-1 items-center justify-center">
        <BoardContainer
          :board="boardStore.board"
          @on-square-click="onSquareClick"
        />
      </div>

      <!-- Right side - Sidebar placeholder -->
      <div class="w-full rounded-lg bg-white/50 p-4 shadow-lg backdrop-blur-sm md:w-80">
        <div class="flex h-full items-center justify-center text-gray-600">
          <p>Sidebar content will go here</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { Position } from '~/types/Position'

const boardStore = useBoardStore()
const webSocketStore = useWebSocketStore()

/**
 * Sends a "step" message to the WebSocket with the current player and the given board index.
 * @param index - The index of the square on the board.
 */
const onSquareClick = (index: number) => {
  const position = Position.fromIndex(index)
  if (!boardStore.puttablePositions.some(p => p.equals(position))) {
    // If the position is not puttable, do nothing
    return
  }
  const json = JSON.stringify({
    Step: {
      PutDisk: {
        color: boardStore.currentPlayer,
        position: position.toJson(),
      },
    },
  })
  webSocketStore.send(json)
}
</script>
