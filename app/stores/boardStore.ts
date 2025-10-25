import { DiskColor } from '~/types/DiskColor'
import type { RawPosition } from '~/types/Position'
import { Position } from '~/types/Position'
import type { SuggestedPosition, SuggestionResponse } from '~/types/Suggestion'

type Board = Array<DiskColor | null>

/**
 * The state of the board in bit board format.
 */
interface BitBoard {
  /**
   * A bit board string representing the positions of the dark disks on the board.
   * Each bit in the number corresponds to a position on the board, with 1 indicating a dark disk and 0 indicating no disk.
   */
  dark_plane: string
  /**
   * A bit board string representing the positions of the light disks on the board.
   * Each bit in the number corresponds to a position on the board, with 1 indicating a light disk and 0 indicating no disk.
   */
  light_plane: string
}

const Winner = {
  Draw: 'Draw',
  Dark: 'Dark',
  Light: 'Light',
} as const
type Winner = typeof Winner[keyof typeof Winner]

/** The state of the game on the server. */
interface ServerGameState {
  /** The current state of the table. */
  table: {
    /** The current state of the board. */
    board: BitBoard
    /** The current player. */
    turn: DiskColor
    /** The history of the game. */
    history: Array<{ position: string } | 'Pass'>
  }
  /** The positions that the current player can put disks on. */
  puttable_positions: RawPosition[]
  /** The roles of the players. */
  player_colors?: Array<DiskColor>
  /** The result of the game if the game is over. */
  judge_result?: {
    /** The number of dark disks on the board. */
    dark_count: number
    /** The number of light disks on the board. */
    light_count: number
    /** The winner of the game. */
    winner: Winner
  }
}

/**
   * Parses a bit board object into a board object.
   *
   * @param bitBoard - The bit board object to parse.
   * @returns The parsed board object.
   */
function parseBitBoard(bitBoard: BitBoard): Board {
  const board: Board = []
  const darkPlane = BigInt(bitBoard.dark_plane)
  const lightPlane = BigInt(bitBoard.light_plane)
  console.log(darkPlane, lightPlane)
  console.log(bitBoard)
  for (let i = 0; i < 64; i++) {
    const mask = BigInt(1) << BigInt(63 - i)
    const dark = darkPlane & mask
    const light = lightPlane & mask
    if (dark != 0n) {
      board.push(DiskColor.Dark)
    }
    else if (light != 0n) {
      board.push(DiskColor.Light)
    }
    else {
      board.push(null)
    }
  }
  return board
}

/**
 * Player interface
 */
interface Player {
  /**
   * The name of the player.
   */
  name: string
  /**
   * The avatar URL of the player.
   */
  avatarUrl: string
  /**
   * The color of the player.
   */
  color?: DiskColor
}

/**
 * A store that holds the state of the game on the client.
 *
 * This store is a replica of the state on the server. It is always synced with the server state.
 *
 * You can use this store to access the current state of the game,
 * including the current state of the board, the current player,
 * the positions that the current player can put disks on,
 * and the winner of the game.
 *
 * This store should not be updated directly.
 * Instead, you should dispatch actions to update the server state.
 */
export const useBoardStore = defineStore('board', () => {
  /**
   * The current state of the board.
   * It is an array of 64 elements, where each element is either
   * `null` (for an empty square), `DiskColor.Dark` (for a dark disk),
   * or `DiskColor.Light` (for a light disk).
   */
  const board = ref<Board>(Array(64).fill(null))
  /**
   * The current player.
   * It is either `DiskColor.Dark` (for dark player) or `DiskColor.Light` (for light player).
   */
  const currentPlayer = ref<DiskColor>(DiskColor.Dark)
  /**
   * The positions that the current player can put disks on.
   */
  const puttablePositions = ref<Position[]>([])
  /**
   * The winner of the game.
   * It is either `DiskColor.Dark` (for dark player), `DiskColor.Light` (for light player), or `null` (if the game is not over).
   */
  const winner = ref<Winner | null>(null)
  /**
   * The players of the game.
   * It is an array of objects, each representing a player with a `name` property.
   */
  const players = ref<Player[]>([])
  /**
   * The suggested positions to highlight on the board.
   */
  const suggestions = ref<SuggestedPosition[]>([])
  /**
   * The ID of the last suggestion request sent to the server.
   */
  const lastSuggestionRequestId = ref<string | null>(null)
  /**
   * Whether the game has started.
   */
  const hasGameStarted = ref(false)

  /**
   * The scores of the game.
   * It is an object with two properties: `Dark` and `Light`,
   * representing the number of dark and light disks on the board, respectively.
   */
  const scores = computed(() => {
    const darkCount = board.value.filter(cell => cell === DiskColor.Dark).length
    const lightCount = board.value.filter(cell => cell === DiskColor.Light).length
    return {
      Dark: darkCount,
      Light: lightCount,
    }
  })

  /**
   * The suggestions with a confidence score above a certain threshold.
   */
  const filteredSuggestions = computed(() => {
    const threshold = 0.75
    return suggestions.value.filter(s => s.confidence >= threshold)
  })

  /**
   * Resets the state of the board to its initial state.
   */
  function reset() {
    board.value = Array(64).fill(null)
    currentPlayer.value = DiskColor.Dark
    puttablePositions.value = []
    winner.value = null
    players.value = []
    suggestions.value = []
    hasGameStarted.value = false
    lastSuggestionRequestId.value = null
  }

  /**
   * Sets the ID of the last suggestion request.
   *
   * @param requestId - The ID of the request.
   */
  function setLastSuggestionRequestId(requestId: string) {
    lastSuggestionRequestId.value = requestId
  }

  /**
   * Updates the state of the board with the state received from the server.
   *
   * @param serverState - The state of the game on the server.
   */
  function setStateFromServer(serverState: ServerGameState) {
    board.value = parseBitBoard(serverState.table.board)
    currentPlayer.value = serverState.table.turn
    puttablePositions.value = serverState.puttable_positions
      .map(({ row, column }) => Position.fromString(row, column))
      .filter(position => position !== null)
    if (serverState.player_colors && players.value.length === serverState.player_colors.length) {
      for (let i = 0; i < serverState.player_colors.length; i++) {
        players.value[i].color = serverState.player_colors[i]
      }
    }
    if (serverState.judge_result) {
      winner.value = serverState.judge_result.winner
    }
    else {
      winner.value = null
    }

    // Reset suggestions since the game state has changed
    suggestions.value = []
    lastSuggestionRequestId.value = null
  }

  /**
   * Handles incoming messages from the WebSocket.
   *
   * @param event - The message event.
   */
  function handleWebsocketMessage(event: MessageEvent) {
    const data = JSON.parse(event.data)
    const keys = Object.keys(data)
    const rootKey = keys[0]

    if (rootKey === 'Players') { // If someone joined or left the table
      const playerNames: { name: string, avatar_url: string }[] = data[rootKey]
      players.value = playerNames.map(({ name, avatar_url }) => ({ name, avatarUrl: avatar_url }))
      console.log('Players:', players.value)
    }
    else if (rootKey === 'GameState') { // If the game state is updated
      const gameState: ServerGameState = data[rootKey]
      console.log('GameState:', gameState)
      setStateFromServer(gameState)
      hasGameStarted.value = true

      if (gameState.judge_result) { // If the game is over
        console.log('Game over:', gameState.judge_result)
        return
      }

      if (gameState.puttable_positions.length === 0) { // If the current player has no puttable positions
        useWebSocketStore().send(
          JSON.stringify({
            Step: {
              PassTurn: currentPlayer.value,
            },
          }),
        )
      }
    }
    else if (rootKey === 'SuggestionResponse') { // If the server sends a suggestion
      const suggestionResponse: SuggestionResponse = data[rootKey]
      if (suggestionResponse.request_id !== lastSuggestionRequestId.value) {
        console.log('Ignoring stale suggestion:', suggestionResponse)
        return
      }

      console.log('SuggestionResponse:', suggestionResponse)

      suggestions.value = suggestionResponse.positions
        .map(p => ({
          ...p,
          position: Position.fromString(p.position.row, p.position.column),
        }))
        .filter(p => p.position !== null) as SuggestedPosition[]
    }
    else if (rootKey === 'InternalServerError') { // If the server returns an error
      const internalServerError = data[rootKey]
      console.error('InternalServerError:', internalServerError)
    }
    else {
      console.error('Unknown message:', data)
    }
  }

  return {
    board,
    currentPlayer,
    puttablePositions,
    winner,
    players,
    suggestions,
    filteredSuggestions,
    scores,
    hasGameStarted,
    reset,
    setStateFromServer,
    handleWebsocketMessage,
    setLastSuggestionRequestId,
  }
})
