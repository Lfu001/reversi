import type { Meta, StoryObj } from '@storybook/vue3'
import HistoryPanel from './HistoryPanel.vue'
import { Position } from '~/types/Position'

const meta: Meta<typeof HistoryPanel> = {
  title: 'HistoryPanel',
  component: HistoryPanel,
}

export default meta
type Story = StoryObj<typeof HistoryPanel>

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

export const empty: Story = {
  args: {
    turns: [],
  },
}

export const SomeTurns: Story = {
  args: {
    turns: sampleTurns,
  },
}
