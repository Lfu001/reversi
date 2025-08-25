import type { Meta, StoryObj } from '@storybook/vue3'
import HistoryPanel from './HistoryPanel.vue'

const meta: Meta<typeof HistoryPanel> = {
  title: 'HistoryPanel',
  component: HistoryPanel,
}

export default meta
type Story = StoryObj<typeof HistoryPanel>

const sampleTurns = [
  { player: 'すずき', move: 'E4' },
  { player: 'たかはし', move: 'D5' },
  { player: 'すずき', move: 'F4' },
  { player: 'たかはし', move: 'C5' },
  { player: 'すずき', move: 'G4' },
  { player: 'たかはし', move: 'B3' },
  { player: 'すずき', move: 'パス' },
  { player: 'たかはし', move: 'F4' },
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
