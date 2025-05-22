<template>
  <button
    :class="[
      'rounded-full px-4 py-2 font-bold transition-colors duration-150 select-none focus:outline-none',
      props.variant === 'outlined'
        ? [
          'border-2',
          props.disabled
            ? 'border-gray-400 text-gray-400'
            : 'cursor-pointer border-indigo-500 text-indigo-500 hover:bg-indigo-50 active:bg-indigo-100',
        ]
        : [
          'text-white',
          props.disabled
            ? 'bg-gray-400'
            : 'cursor-pointer bg-indigo-500 hover:bg-indigo-600 focus:ring-0 active:bg-indigo-700',
        ],
    ]"
    @click="handleClick"
  >
    {{ props.label }}
  </button>
</template>

<script setup lang="ts">
const props = withDefaults(defineProps<{
  /**
   * The text to display on the button.
   */
  label: string
  /**
   * Whether the button is disabled.
   */
  disabled?: boolean
  /**
   * The style of the button.
   */
  variant?: 'fill' | 'outlined'
}>(), {
  disabled: false,
  variant: 'fill',
})

const emit = defineEmits<{
  /**
   * Emitted when the button is clicked, if not disabled.
   */
  (e: 'click'): void
}>()

/**
 * Handles the click event for the button.
 * Emits a 'click' event if the button is not disabled.
 */
const handleClick = () => {
  if (!props.disabled) {
    emit('click')
  }
}
</script>
