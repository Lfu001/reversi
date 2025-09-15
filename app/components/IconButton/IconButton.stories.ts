import type { Meta, StoryObj } from '@storybook/vue3'

import { HomeIcon } from '@heroicons/vue/20/solid'
import IconButton from './IconButton.vue'

const meta = {
  title: 'IconButton',
  component: IconButton,
  tags: ['autodocs'],
  argTypes: {
    icon: { control: false },
    disabled: {
      control: 'boolean',
      description: 'Whether the button is disabled',
    },
  },
  args: {
    icon: HomeIcon,
    disabled: false,
  },
} satisfies Meta<typeof IconButton>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {}

export const Disabled: Story = {
  args: {
    disabled: true,
  },
}
