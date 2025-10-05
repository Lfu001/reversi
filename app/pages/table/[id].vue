<template>
  <div class="flex min-h-screen flex-col gap-4 bg-[#e9c983] p-4 dark:bg-[#28231C]">
    <!-- Header Bar -->
    <header class="container mx-auto w-full">
      <IconButton
        :icon="HomeIcon"
        @click="handleBackToMenuClick"
      />
    </header>

    <!-- Main Content -->
    <div class="container mx-auto flex flex-1 flex-col items-stretch justify-center gap-8 lg:flex-row">
      <!-- Left side - Board Area -->
      <div class="game-container">
        <div class="player-a">
          <PlayerAvatar
            :src="getAvatarUrl(0)"
            :name="getAvatarName(0)"
            :disk-color="getAvatarColor(0)"
            :is-my-turn="getIsMyTurn(0)"
            class="h-16 w-16 lg:h-24 lg:w-24"
          />
        </div>
        <div class="board-area w-full min-w-[320px] sm:min-w-[480px] md:min-w-[480px] lg:min-w-[600px]">
          <BoardContainer
            :board="boardStore.board"
            :puttable-positions="boardStore.puttablePositions"
            :suggested-positions="boardStore.filteredSuggestions.map(s => s.position)"
            :position-guide-color="boardStore.currentPlayer"
            @on-square-click="onSquareClick"
          />
        </div>
        <div class="player-b">
          <PlayerAvatar
            :src="getAvatarUrl(1)"
            :name="getAvatarName(1)"
            :disk-color="getAvatarColor(1)"
            :is-my-turn="getIsMyTurn(1)"
            class="h-14 w-14 lg:h-20 lg:w-20"
          />
        </div>
      </div>

      <!-- Right side - Game Controls -->
      <div class="w-full rounded-lg bg-white/50 p-4 shadow-lg backdrop-blur-sm lg:w-80">
        <div class="space-y-4">
          <div>
            <h3 class="text-lg font-semibold text-gray-800">
              Game Controls
            </h3>
          </div>

          <!-- Suggestion Button -->
          <button
            :disabled="!canRequestSuggestion"
            class="w-full rounded-lg bg-blue-500 px-4 py-2 text-white transition-colors hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
            @click="requestSuggestion"
          >
            Get Suggestion
          </button>

          <!-- Game Status -->
          <div class="text-sm text-gray-600">
            <p>Current Player: {{ boardStore.currentPlayer }}</p>
            <p>Puttable Positions: {{ boardStore.puttablePositions.length }}</p>
          </div>

          <!-- Scores -->
          <div class="text-sm text-gray-600">
            <p>Dark: {{ boardStore.scores.Dark }}</p>
            <p>Light: {{ boardStore.scores.Light }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { HomeIcon } from '@heroicons/vue/24/solid'
import type { DiskColor } from '~/types/DiskColor'
import { Position } from '~/types/Position'
import type { SuggestionRequest } from '~/types/Suggestion'

/** The board store. */
const boardStore = useBoardStore()
/** The WebSocket store. */
const webSocketStore = useWebSocketStore()

/**
 * Returns the avatar URL of the player at the given index.
 *
 * If the index is out of bounds, an empty string is returned.
 *
 * @param index The index of the player.
 */
const getAvatarUrl = (index: number): string => {
  if (0 <= index && index < boardStore.players.length) {
    return boardStore.players[index].avatarUrl
  }
  return ''
}

/**
 * Returns the name of the player at the given index.
 *
 * If the index is out of bounds, an empty string is returned.
 *
 * @param index The index of the player.
 */
const getAvatarName = (index: number): string => {
  if (0 <= index && index < boardStore.players.length) {
    return boardStore.players[index].name
  }
  return ''
}

/**
 * Returns the color of the player at the given index.
 *
 * If the index is out of bounds, undefined is returned.
 *
 * @param index The index of the player.
 */
const getAvatarColor = (index: number): DiskColor | undefined => {
  if (0 <= index && index < boardStore.players.length) {
    return boardStore.players[index].color
  }
  return undefined
}

/**
 * Returns whether it is the turn of the player at the given index.
 *
 * If the index is out of bounds, false is returned.
 *
 * @param index The index of the player.
 */
const getIsMyTurn = (index: number): boolean => {
  if (0 <= index && index < boardStore.players.length) {
    return boardStore.currentPlayer === boardStore.players[index].color
  }
  return false
}

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

/**
 * Closes the WebSocket connection, resets the board store, and navigates to the menu page.
 */
const handleBackToMenuClick = () => {
  webSocketStore.close()
  boardStore.reset()
  navigateTo('/menu')
}

/**
 * Requests a suggestion for the current game state.
 */
const requestSuggestion = () => {
  if (!boardStore.hasGameStarted) {
    console.warn('Cannot request suggestion: game has not started')
    return
  }

  const requestId = crypto.randomUUID()
  boardStore.setLastSuggestionRequestId(requestId)

  const suggestionRequest: SuggestionRequest = {
    model: 'lfu_random',
    request_id: requestId,
  }
  const json = JSON.stringify({ SuggestionRequest: suggestionRequest })
  webSocketStore.send(json)
}

/**
 * Returns whether a suggestion can be requested.
 */
const canRequestSuggestion = computed(() => {
  return boardStore.hasGameStarted && webSocketStore.isConnected && boardStore.puttablePositions.length > 0
})
</script>

<style scoped>
.game-container {
  display: grid;
  align-items: center;
}

.board-area {
  grid-area: board;
}
.player-a {
  grid-area: player-a;
}
.player-b {
  grid-area: player-b;
}

/* Wide screen layout: [A, Board, B] */
@media (min-width: 1024px) {
  .game-container {
    grid-template-columns: auto 1fr auto;
    grid-template-areas: "player-a board player-b";
    gap: 1rem;
  }
  .player-a {
    align-self: end;
    padding-bottom: 2rem;
  }
  .player-b {
    align-self: start;
    padding-top: 2rem;
  }
}

/* Narrow screen layout: [Board] over [A, Spacer, B] */
@media (max-width: 1023px) {
  .game-container {
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr auto;
    grid-template-areas:
      "board board"
      "player-a player-b";
    align-items: center;
    justify-items: stretch;
    gap: 1.5rem;
  }
  .board-area {
    justify-self: center;
  }
  .player-a {
    justify-self: start;
  }
  .player-b {
    justify-self: end;
  }
}
</style>
