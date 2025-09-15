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
   * Creates a Position from a 0-indexed number.
   * @param index - The index of the position. Must be in the range 0-63.
   * @returns The Position corresponding to the given index, or null if the index is out of bounds.
   */
  static fromIndex(index: number) {
    const row = Math.floor(index / 8)
    const column = index % 8
    return new Position(row as Row, column as Column)
  }
}

export { Position }
