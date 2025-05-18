import type { Meta, StoryObj } from '@storybook/vue3'
import Button from './Button.vue'

const meta: Meta<typeof Button> = {
  component: Button,
}

export default meta
type Story = StoryObj<typeof Button>

export const Fill: Story = {
  args: {
    label: 'Button',
    variant: 'fill',
    disabled: false,
  },
  render: args => ({
    components: { Button },
    setup() {
      return { args }
    },
    template: '<Button v-bind="args"/>',
  }),
}

export const FillDisabled: Story = {
  args: {
    label: 'Button',
    variant: 'fill',
    disabled: true,
  },
  render: args => ({
    components: { Button },
    setup() {
      return { args }
    },
    template: '<Button v-bind="args"/>',
  }),
}

export const Outlined: Story = {
  args: {
    label: 'Button',
    variant: 'outlined',
    disabled: false,
  },
  render: args => ({
    components: { Button },
    setup() {
      return { args }
    },
    template: '<Button v-bind="args"/>',
  }),
}

export const OutlinedDisabled: Story = {
  args: {
    label: 'Button',
    variant: 'outlined',
    disabled: true,
  },
  render: args => ({
    components: { Button },
    setup() {
      return { args }
    },
    template: '<Button v-bind="args"/>',
  }),
}
