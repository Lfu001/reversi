import type { Meta, StoryObj } from '@storybook/vue3'
import ModalAvatarSelect from './ModalAvatarSelect.vue'

const meta: Meta<typeof ModalAvatarSelect> = {
  title: 'Modal/ModalAvatarSelect',
  component: ModalAvatarSelect,
  tags: ['autodocs'],
  argTypes: {
    isOpen: { control: 'boolean' },
    onClose: { action: 'closed' },
    onSelect: { action: 'selected' },
  },
}

export default meta
type Story = StoryObj<typeof ModalAvatarSelect>

export const Default: Story = {
  args: {
    isOpen: true,
  },
}
