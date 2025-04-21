import type { Meta, StoryObj } from '@storybook/vue3'

import BoardContainer from './BoardContainer.vue'

const meta: Meta<typeof BoardContainer> = {
  title: 'BoardContainer',
  component: BoardContainer,
}

export default meta
type Story = StoryObj<typeof BoardContainer>

export const Primary: Story = {
  args: {},
}
