import type { Meta, StoryObj } from '@storybook/vue3'
import StatusMessage from './StatusMessage.vue'

const meta: Meta<typeof StatusMessage> = {
  component: StatusMessage,
  argTypes: {
    isReady: {
      control: 'boolean',
      description: 'Whether the status is in a ready state',
    },
    statusText: {
      control: 'text',
      description: 'Text to display when not in ready state',
    },
    readyText: {
      control: 'text',
      description: 'Text to display when in ready state',
    },
  },
  args: {
    isReady: false,
    statusText: '処理中...',
    readyText: '準備ができました！',
  },
}

export default meta
type Story = StoryObj<typeof StatusMessage>

export const Loading: Story = {
  args: {
    isReady: false,
    statusText: '参加者を待っています...',
  },
  render: args => ({
    components: { StatusMessage },
    setup() {
      return { args }
    },
    template: '<StatusMessage v-bind="args" />',
  }),
}

export const Ready: Story = {
  args: {
    isReady: true,
    readyText: '準備が完了しました！',
  },
  render: args => ({
    components: { StatusMessage },
    setup() {
      return { args }
    },
    template: '<StatusMessage v-bind="args" />',
  }),
}

export const CustomText: Story = {
  args: {
    isReady: false,
    statusText: 'カスタムメッセージを表示中...',
  },
  render: args => ({
    components: { StatusMessage },
    setup() {
      return { args }
    },
    template: '<StatusMessage v-bind="args" />',
  }),
}
