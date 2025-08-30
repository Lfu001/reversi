import type { Meta, StoryObj } from '@storybook/vue3'
import BackgroundWaterDrop from './BackgroundWaterDrop.vue'

const meta: Meta<typeof BackgroundWaterDrop> = {
  component: BackgroundWaterDrop,
}

export default meta
type Story = StoryObj<typeof BackgroundWaterDrop>

export const Default: Story = {
  render: args => ({
    components: { BackgroundWaterDrop },
    setup() {
      return { args }
    },
    template: '<BackgroundWaterDrop />',
  }),
}
