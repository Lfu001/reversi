import type { Meta, StoryObj } from '@storybook/vue3'
import ModalJoin from './ModalJoin.vue'

const meta: Meta<typeof ModalJoin> = {
  component: ModalJoin,
}

export default meta
type ModalJoinStory = StoryObj<typeof ModalJoin>

export const Default: ModalJoinStory = {
  args: {},
  render: args => ({
    components: { ModalJoin },
    setup() {
      return { args }
    },
    template: '<ModalJoin @close="() => {}" @join="(password) => {}" />',
  }),
}
