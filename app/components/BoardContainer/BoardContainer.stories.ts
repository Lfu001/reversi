import type { Meta, StoryObj } from '@storybook/vue3'

import BoardContainer from './BoardContainer.vue'
import { DiskColor } from '~/types/DiskColor'
import { Position } from '~/types/Position'

const meta: Meta<typeof BoardContainer> = {
  title: 'BoardContainer',
  component: BoardContainer,
}

export default meta
type Story = StoryObj<typeof BoardContainer>

export const Empty: Story = {
  args: {
    board: Array(64).fill(null),
    puttablePositions: [],
    positionGuideColor: DiskColor.Dark,
  },
}

export const InitialState1: Story = {
  args: {
    board: (() => {
      const board = Array(64).fill(null)
      board[27] = DiskColor.Dark
      board[28] = DiskColor.Light
      board[35] = DiskColor.Light
      board[36] = DiskColor.Dark
      return board
    })(),
    puttablePositions: [
      Position.fromIndex(20),
      Position.fromIndex(29),
      Position.fromIndex(34),
      Position.fromIndex(43),
    ],
    positionGuideColor: DiskColor.Dark,
  },
}

export const InitialState2: Story = {
  args: {
    board: (() => {
      const board = Array(64).fill(null)
      board[27] = DiskColor.Dark
      board[28] = DiskColor.Dark
      board[29] = DiskColor.Dark
      board[35] = DiskColor.Light
      board[36] = DiskColor.Dark
      return board
    })(),
    puttablePositions: [
      Position.fromIndex(19),
      Position.fromIndex(21),
      Position.fromIndex(37),
    ],
    positionGuideColor: DiskColor.Light,
  },
}

export const MidGame: Story = {
  args: {
    board: (() => {
      const board = Array(64).fill(null)
      const blackPositions = [19, 20, 27, 28, 35, 36, 43, 44]
      const whitePositions = [26, 29, 34, 37, 42, 45]
      blackPositions.forEach(p => (board[p] = DiskColor.Dark))
      whitePositions.forEach(p => (board[p] = DiskColor.Light))
      return board
    })(),
    puttablePositions: [
      Position.fromIndex(17),
      Position.fromIndex(22),
      Position.fromIndex(25),
      Position.fromIndex(30),
      Position.fromIndex(33),
      Position.fromIndex(38),
      Position.fromIndex(41),
      Position.fromIndex(46),
      Position.fromIndex(49),
      Position.fromIndex(54),
    ],
    positionGuideColor: DiskColor.Dark,
  },
}

export const EndGame: Story = {
  args: {
    board: (() => {
      const board = Array(64).fill(DiskColor.Dark)
      const whitePositions = [10, 11, 12, 13, 14, 15, 16, 17]
      whitePositions.forEach(i => (board[i] = DiskColor.Light))
      return board
    })(),
    puttablePositions: [],
    positionGuideColor: DiskColor.Dark,
  },
}
