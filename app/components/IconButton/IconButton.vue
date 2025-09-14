<template>
  <button
    :class="[
      'flex items-center justify-center rounded-full p-2 transition-colors duration-150 select-none focus:outline-none',
      'text-gray-700 hover:bg-gray-100 active:bg-gray-200',
      'dark:text-gray-300 dark:hover:bg-gray-700 dark:active:bg-gray-600',
      { 'cursor-not-allowed opacity-50': disabled, 'cursor-pointer': !disabled },
    ]"
    :disabled="disabled"
    @click="handleClick"
  >
    <component
      :is="icon"
      class="h-6 w-6"
    />
  </button>
</template>

<script setup lang="ts">
import type { FunctionalComponent, HTMLAttributes } from 'vue'

type IconComponent = FunctionalComponent<HTMLAttributes & { class?: string }>

const props = withDefaults(defineProps<{
  /**
   * The icon component to display.
   */
  icon: IconComponent
  /**
   * Whether the button is disabled.
   */
  disabled?: boolean
}>(), {
  disabled: false,
})

const emit = defineEmits(['click'])

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
