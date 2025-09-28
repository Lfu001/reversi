import type { Meta, StoryObj } from '@storybook/vue3'
import RightSideBar from './RightSideBar.vue'
import { Position } from '~/types/Position'

const meta: Meta<typeof RightSideBar> = {
  component: RightSideBar,
}

export default meta
type Story = StoryObj<typeof RightSideBar>

const sampleTurns = [
  { player: 'すずき', position: new Position(3, 4) },
  { player: 'たかはし', position: new Position(4, 3) },
  { player: 'すずき', position: new Position(3, 5) },
  { player: 'たかはし', position: new Position(4, 2) },
  { player: 'すずき', position: new Position(3, 6) },
  { player: 'たかはし', position: new Position(2, 1) },
  { player: 'すずき', position: null },
  { player: 'たかはし', position: new Position(3, 5) },
]

export const GameStart: Story = {
  args: {
    darkScore: 4,
    lightScore: 1,
    darkName: 'くろだ',
    lightName: 'しらい',
    turns: sampleTurns,
  },
}
