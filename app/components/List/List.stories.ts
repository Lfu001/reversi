import type { Meta, StoryObj } from '@storybook/vue3'
import List from './List.vue'

const meta: Meta<typeof List> = {
  component: List,
  argTypes: {
    items: { control: 'object' },
    title: { control: 'text' },
    maxItems: { control: 'number' },
    showCount: { control: 'boolean' },
    emptyMessage: { control: 'text' },
  },
  parameters: {
    docs: {
      description: {
        component:
          'A flexible list component that can display items with custom rendering.',
      },
    },
  },
}

export default meta
type Story = StoryObj<typeof List>

const defaultItems = ['Item 1', 'Item 2', 'Item 3']

export const Default: Story = {
  args: {
    items: defaultItems,
    title: 'Items',
    maxItems: 5,
    showCount: true,
    emptyMessage: 'No items',
  },
  render: args => ({
    components: { List },
    setup() {
      return { args }
    },
    template: '<List v-bind="args" />',
  }),
}
export const Empty: Story = {
  ...Default,
  args: {
    ...Default.args,
    items: [],
    emptyMessage: 'No items found',
  },
}

export const WithoutCount: Story = {
  ...Default,
  args: {
    ...Default.args,
    title: 'Items',
    showCount: false,
  },
}
