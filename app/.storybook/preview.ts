import type { Preview } from '@storybook/vue3'

const preview: Preview = {
  parameters: {
    actions: { argTypesRegex: '^on[A-Z].*' },
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/,
      },
    },
  },
  tags: ['autodocs'],
}

// Workaround to avoid the __VUE_HMR_RUNTIME__  error
// TODO: Remove when [this issue](https://github.com/storybookjs/storybook/issues/31010) is fixed
// @ts-expect-error __VUE_HMR_RUNTIME__ is not defined in the window object
window.__VUE_HMR_RUNTIME__ = {
  createRecord: () => {},
  reload: () => {
    window.location.reload()
  },
  rerender: () => {
    window.location.reload()
  },
}

export default preview
