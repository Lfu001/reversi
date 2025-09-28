import type { Meta, StoryObj } from '@storybook/vue3'
import ScoreBoard from './ScoreBoard.vue'

const meta: Meta<typeof ScoreBoard> = {
  title: 'ScoreBoard',
  component: ScoreBoard,
}

export default meta
type Story = StoryObj<typeof ScoreBoard>

export const Default: Story = {}

export const CloseMatch: Story = {
  args: {
    darkScore: 30,
    darkName: 'くろだ',
    lightScore: 31,
    lightName: 'しらい',
  },
}
