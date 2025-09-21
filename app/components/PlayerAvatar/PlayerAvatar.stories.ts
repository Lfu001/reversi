import type { Meta, StoryObj } from '@storybook/vue3'
import PlayerAvatar from './PlayerAvatar.vue'
import { DiskColor } from '~/types/DiskColor'

const meta: Meta<typeof PlayerAvatar> = {
  title: 'PlayerAvatar',
  component: PlayerAvatar,
  tags: ['autodocs'],
  argTypes: {
    src: { control: 'text' },
    name: { control: 'text' },
    diskColor: {
      control: 'select',
      options: [DiskColor.Dark, DiskColor.Light, undefined],
    },
    size: { control: 'number' },
    isMyTurn: { control: 'boolean' },
  },
}

export default meta
type Story = StoryObj<typeof PlayerAvatar>

const defaultImage = '/images/avatars/autumn-leaves.png'

export const Default: Story = {
  args: {
    src: defaultImage,
    name: 'すずき',
    diskColor: DiskColor.Dark,
    size: 80,
    isMyTurn: false,
  },
}

export const MyTurn: Story = {
  args: {
    src: defaultImage,
    name: 'すずき',
    diskColor: DiskColor.Dark,
    size: 80,
    isMyTurn: true,
  },
}

export const LightDisk: Story = {
  args: {
    src: defaultImage,
    name: 'さとう',
    diskColor: DiskColor.Light,
    size: 80,
  },
}

export const NoName: Story = {
  args: {
    src: defaultImage,
    name: undefined,
    diskColor: DiskColor.Dark,
    size: 120,
  },
}

export const NoDiskColor: Story = {
  args: {
    src: defaultImage,
    name: 'たかはし',
    diskColor: undefined,
    size: 80,
  },
}

export const NoNameNoDiskColor: Story = {
  args: {
    src: defaultImage,
    name: undefined,
    diskColor: undefined,
    size: 120,
  },
}

export const SmallSize: Story = {
  args: {
    src: defaultImage,
    name: 'こばやし',
    diskColor: DiskColor.Dark,
    size: 40,
  },
}
