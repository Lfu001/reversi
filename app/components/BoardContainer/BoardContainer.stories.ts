import type { Meta, StoryObj } from '@storybook/vue3'

import BoardContainer from './BoardContainer.vue'

const meta: Meta<typeof BoardContainer> = {
  title: 'BoardContainer',
  component: BoardContainer,
}

export default meta
type Story = StoryObj<typeof BoardContainer>

export const Empty: Story = {
  args: {
    board: Array(64).fill(null),
  },
}

export const InitialState: Story = {
  args: {
    board: (() => {
      const board = Array(64).fill(null)
      board[27] = 'Dark'
      board[28] = 'Light'
      board[35] = 'Light'
      board[36] = 'Dark'
      return board
    })(),
  },
}

export const MidGame: Story = {
  args: {
    board: (() => {
      const board = Array(64).fill(null)
      const blackPositions = [19, 20, 27, 28, 35, 36, 43, 44]
      const whitePositions = [26, 29, 34, 37, 42, 45]
      blackPositions.forEach(p => board[p] = 'Dark')
      whitePositions.forEach(p => board[p] = 'Light')
      return board
    })(),
  },
}

export const EndGame: Story = {
  args: {
    board: (() => {
      const board = Array(64).fill('Dark')
      const whitePositions = [10, 11, 12, 13, 14, 15, 16, 17]
      whitePositions.forEach(i => board[i] = 'Light')
      return board
    })(),
  },
}
