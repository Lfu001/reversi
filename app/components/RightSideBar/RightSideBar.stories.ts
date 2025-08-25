import type { Meta, StoryObj } from '@storybook/vue3'
import RightSideBar from './RightSideBar.vue'

const meta: Meta<typeof RightSideBar> = {
  component: RightSideBar,
}

export default meta
type Story = StoryObj<typeof RightSideBar>

const sampleTurns = [
  { player: 'くろだ', move: 'E4' },
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
