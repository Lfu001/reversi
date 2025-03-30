<template>
  <button
    :class="[
      'rounded-full px-4 py-2 font-bold text-white opacity-80 transition-colors duration-150 select-none focus:outline-none',
      props.disabled
        ? 'bg-gray-400'
        : 'cursor-pointer bg-indigo-500 hover:bg-indigo-600 focus:ring-0 active:bg-indigo-700',
    ]"
    @click="handleClick"
  >
    {{ props.label }}
  </button>
</template>

<script setup lang="ts">
/**
 * Props for the Button component.
 * @prop {string} label - The text to display on the button.
 * @prop {boolean} [disabled] - Determines if the button is disabled.
 */
const props = defineProps<{
  label: string
  disabled?: boolean
}>()

/**
 * Emits events for the Button component.
 * @event click - Emitted when the button is clicked, if not disabled.
 */
const emit = defineEmits<{
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
