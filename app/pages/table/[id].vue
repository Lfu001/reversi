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
const websocket = useWebsocket()

websocket.setOnMessageHandler((event: MessageEvent) => {
  const data = JSON.parse(event.data)
  const keys = Object.keys(data)
  const rootKey = keys[0]

  if (rootKey === 'Connected') { // If someone joins the table
    const connected = data[rootKey]
    console.log('Connected:', connected)
  }
  else if (rootKey === 'Disconnected') { // If someone leaves the table
    const disconnected = data[rootKey]
    console.log('Disconnected:', disconnected)
  }
  else if (rootKey === 'GameState') { // If the game state is updated
    const gameState = data[rootKey]
    console.log('GameState:', gameState)
    boardStore.setStateFromServer(gameState)

    if (gameState.judge_result) { // If the game is over
      console.log('Game over:', gameState.judge_result)
      return
    }

    if (gameState.puttable_positions.length === 0) { // If the current player has no puttable positions
      websocket.send(
        JSON.stringify({
          Step: {
            action: {
              PassTurn: {
                color: boardStore.currentPlayer,
              },
            },
          },
        }),
      )
    }
  }
  else if (rootKey === 'InternalServerError') { // If the server returns an error
    const internalServerError = data[rootKey]
    console.error('InternalServerError:', internalServerError)
  }
})

/**
 * Sends a "step" message to the websocket with the current player and the given board index.
 * @param index - The index of the square on the board.
 */
const onSquareClick = (index: number) => {
  if (!boardStore.puttablePositions.includes(Position.fromIndex(index))) {
    // If the position is not puttable, do nothing
    return
  }
  websocket.send(
    JSON.stringify({
      Step: {
        action: {
          PutDisk: {
            color: boardStore.currentPlayer,
            position: Position.fromIndex(index),
          },
        },
      },
    }),
  )
}
</script>
