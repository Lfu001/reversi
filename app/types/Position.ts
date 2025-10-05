const Row = {
  One: 0,
  Two: 1,
  Three: 2,
  Four: 3,
  Five: 4,
  Six: 5,
  Seven: 6,
  Eight: 7,
} as const

type Row = typeof Row[keyof typeof Row]

const Column = {
  A: 0,
  B: 1,
  C: 2,
  D: 3,
  E: 4,
  F: 5,
  G: 6,
  H: 7,
} as const

type Column = typeof Column[keyof typeof Column]

class Position {
  row: Row
  column: Column

  constructor(row: Row, column: Column) {
    this.row = row
    this.column = column
  }

  /**
   * Converts this position to string
   * @returns a string representation of the position
   */
  toString(): string {
    const colKey = Object.keys(Column).find(k => Column[k as keyof typeof Column] === this.column)
    const rowNumber = this.row + 1
    return `${colKey}${rowNumber}`
  }

  /**
   * Check if the given Position is equal to this one.
   * @param other - The other Position to compare with.
   * @returns true if the given Position is equal to this one, false otherwise.
   */
  equals(other: Position): boolean {
    return this.row === other.row && this.column === other.column
  }

  /**
   * Creates a Position from a 0-indexed number.
   * @param index - The index of the position. Must be in the range 0-63.
   * @returns The Position corresponding to the given index, or null if the index is out of bounds.
   */
  static fromIndex(index: number) {
    const row = Math.floor(index / 8)
    const column = index % 8
    return new Position(row as Row, column as Column)
  }

  /**
   * Creates a Position from a letter-number pair.
   * @param row - The letter of the row (One-Eight).
   * @param column - The letter of the column (A-H).
   * @returns The Position corresponding to the given row and column, or null if the row or column is out of bounds.
   */
  static fromString(row: string, column: string) {
    let r: Row | null = null
    let c: Column | null = null
    for (const [key, value] of Object.entries(Row)) {
      if (key === row) {
        r = value as Row
      }
    }
    for (const [key, value] of Object.entries(Column)) {
      if (key === column) {
        c = value as Column
      }
    }
    if (r !== null && c !== null) {
      return new Position(r, c)
    }
    return null
  }

  /**
   * Converts this Position to a string.
   * @returns A string of the form `{row:<Row>,column:<Column>}`, or null if the row or column is out of bounds.
   */
  toJson() {
    let r: string | null = null
    let c: string | null = null
    for (const [key, value] of Object.entries(Row)) {
      if (value === this.row) {
        r = key
      }
    }
    for (const [key, value] of Object.entries(Column)) {
      if (value === this.column) {
        c = key
      }
    }
    if (r !== null && c !== null) {
      return { row: r, column: c }
    }
    return null
  }
}

/**
 * A raw position object as received from the server.
 */
export interface RawPosition {
  /** The row of the position. */
  row: string
  /** The column of the position. */
  column: string
}

export { Position }
