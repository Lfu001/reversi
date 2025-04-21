import type { Meta, StoryObj } from '@storybook/vue3'

import DiskComponent from './DiskComponent.vue'

const meta: Meta<typeof DiskComponent> = {
  title: 'DiskComponent',
  component: DiskComponent,
}

export default meta
type Story = StoryObj<typeof DiskComponent>

export const Primary: Story = {
  args: {
    color: null,
  },
}

export const DarkStone: Story = {
  args: {
    color: 'Dark',
  },
}

export const LightStone: Story = {
  args: {
    color: 'Light',
  },
}
