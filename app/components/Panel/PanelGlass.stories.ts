import type { Meta, StoryObj } from '@storybook/vue3'
import PanelGlass from './PanelGlass.vue'

const meta: Meta<typeof PanelGlass> = {
  component: PanelGlass,
}

export default meta
type Story = StoryObj<typeof PanelGlass>

export const Default: Story = {
  render: () => ({
    components: { PanelGlass },
    template: `<PanelGlass />`,
  }),
}
