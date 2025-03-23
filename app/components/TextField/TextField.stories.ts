import type { Meta, StoryObj } from '@storybook/vue3'
import TextField from './TextField.vue'

const meta: Meta<typeof TextField> = {
  component: TextField,
}

export default meta
type Story = StoryObj<typeof TextField>

export const Basic: Story = {
  render: args => ({
    components: { TextField },
    setup() {
      return { args }
    },
    template: '<TextField :placeholder="args.placeholder"/>',
  }),
  args: {
    placeholder: 'placeholder',
  },
}
