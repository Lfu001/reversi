import { DiskColor } from '~/types/DiskColor'
import type { Position } from '~/types/Position'

type Board = Array<DiskColor | null>

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
    board: Board
    /** The current player. */
    turn: DiskColor
    /** The history of the game. */
    history: Array<{ position: Position } | 'Pass'>
  }
  /** The positions that the current player can put disks on. */
  puttable_positions: Position[]
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
   * Updates the state of the board with the state received from the server.
   *
   * @param {ServerGameState} serverState - The state of the game on the server.
   */
  function setStateFromServer(serverState: ServerGameState) {
    board.value = serverState.table.board
    currentPlayer.value = serverState.table.turn
    puttablePositions.value = serverState.puttable_positions

    if (serverState.judge_result) {
      winner.value = serverState.judge_result.winner
    }
    else {
      winner.value = null
    }
  }

  return {
    board,
    currentPlayer,
    puttablePositions,
    winner,
    scores,
    setStateFromServer,
  }
})
