import type { Meta, StoryObj } from '@storybook/vue3'
import Button from './Button.vue'

const meta: Meta<typeof Button> = {
  component: Button,
}

export default meta
type Story = StoryObj<typeof Button>

export const Enabled: Story = {
  args: {
    label: 'Button',
    disabled: false,
  },
  render: args => ({
    components: { Button },
    setup() {
      return { args }
    },
    template: '<Button :label="args.label" :disabled="args.disabled"/>',
  }),
}

export const Disabled: Story = {
  args: {
    label: 'Button',
    disabled: true,
  },
  render: args => ({
    components: {
      Button,
    },
    setup() {
      return { args }
    },
    template: '<Button :label="args.label" :disabled="args.disabled"/>',
  }),
}
