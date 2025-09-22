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
    isMyTurn: { control: 'boolean' },
  },
}

export default meta
type Story = StoryObj<typeof PlayerAvatar>

const defaultImage = useAvatars().choose()

export const Default: Story = {
  render: args => ({
    components: { PlayerAvatar },
    setup() {
      return { args }
    },
    template: '<PlayerAvatar v-bind="args" class="h-24 w-24" />',
  }),
  args: {
    src: defaultImage,
    name: 'すずき',
    diskColor: DiskColor.Dark,
    isMyTurn: false,
  },
}

export const MyTurn: Story = {
  render: args => ({
    components: { PlayerAvatar },
    setup() {
      return { args }
    },
    template: '<PlayerAvatar v-bind="args" class="h-24 w-24" />',
  }),
  args: {
    src: defaultImage,
    name: 'すずき',
    diskColor: DiskColor.Dark,
    isMyTurn: true,
  },
}

export const LightDisk: Story = {
  render: args => ({
    components: { PlayerAvatar },
    setup() {
      return { args }
    },
    template: '<PlayerAvatar v-bind="args" class="h-24 w-24" />',
  }),
  args: {
    src: defaultImage,
    name: 'さとう',
    diskColor: DiskColor.Light,
  },
}

export const NoName: Story = {
  render: args => ({
    components: { PlayerAvatar },
    setup() {
      return { args }
    },
    template: '<PlayerAvatar v-bind="args" class="h-24 w-24" />',
  }),
  args: {
    src: defaultImage,
    name: undefined,
    diskColor: DiskColor.Dark,
  },
}

export const NoDiskColor: Story = {
  render: args => ({
    components: { PlayerAvatar },
    setup() {
      return { args }
    },
    template: '<PlayerAvatar v-bind="args" class="h-24 w-24" />',
  }),
  args: {
    src: defaultImage,
    name: 'たかはし',
    diskColor: undefined,
  },
}

export const NoNameNoDiskColor: Story = {
  render: args => ({
    components: { PlayerAvatar },
    setup() {
      return { args }
    },
    template: '<PlayerAvatar v-bind="args" class="h-24 w-24" />',
  }),
  args: {
    src: defaultImage,
    name: undefined,
    diskColor: undefined,
  },
}
