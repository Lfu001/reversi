import type { Meta, StoryObj } from '@storybook/vue3'
import CopyableCode from './CopyableCode.vue'

const meta: Meta<typeof CopyableCode> = {
  component: CopyableCode,
  argTypes: {
    value: { control: 'text' },
    label: { control: 'text' },
    copyButtonText: { control: 'text' },
    copiedMessage: { control: 'text' },
    id: { control: 'text' },
  },
}

export default meta
type Story = StoryObj<typeof CopyableCode>

export const Default: Story = {
  args: {
    value: 'ABCD1234',
    label: 'Invite Code',
    copyButtonText: 'Copy to clipboard',
    copiedMessage: 'Copied!',
    id: 'invite-code',
  },
  render: args => ({
    components: { CopyableCode },
    setup() {
      return { args }
    },
    template: '<CopyableCode v-bind="args" />',
  }),
}

export const NoLabel: Story = {
  ...Default,
  args: {
    ...Default.args,
    label: '',
  },
}

export const CustomMessages: Story = {
  ...Default,
  args: {
    ...Default.args,
    copyButtonText: 'クリップボードにコピー',
    copiedMessage: 'コピーしました！',
  },
}
