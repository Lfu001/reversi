import type { Meta, StoryObj } from '@storybook/vue3'

import { DiskColor } from '~/types/DiskColor'
import DiskComponent from './DiskComponent.vue'

const meta: Meta<typeof DiskComponent> = {
  title: 'DiskComponent',
  component: DiskComponent,
}

export default meta
type Story = StoryObj<typeof DiskComponent>

export const Empty: Story = {
  args: {
    color: null,
    guideColor: null,
  },
}

export const DarkStone: Story = {
  args: {
    color: DiskColor.Dark,
    guideColor: null,
  },
}

export const LightStone: Story = {
  args: {
    color: DiskColor.Light,
    guideColor: null,
  },
}

export const GuideDark: Story = {
  args: {
    color: null,
    guideColor: DiskColor.Dark,
  },
}

export const GuideLight: Story = {
  args: {
    color: null,
    guideColor: DiskColor.Light,
  },
}
