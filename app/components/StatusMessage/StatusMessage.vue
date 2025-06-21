<template>
  <div
    class="text-center bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 transition-colors duration-300"
  >
    <div
      v-if="!isReady"
      class="flex items-center justify-center space-x-3"
    >
      <div class="animate-pulse flex space-x-2">
        <div
          v-for="delay in [0, 0.2, 0.4]"
          :key="delay"
          class="h-3 w-3 bg-blue-500 rounded-full animate-bounce"
          :style="{ animationDelay: `${delay}s` }"
        />
      </div>
      <p class="text-gray-600 dark:text-gray-300 transition-colors duration-300">
        {{ statusText }}
      </p>
    </div>
    <div
      v-else
      class="text-center"
    >
      <p class="text-green-600 dark:text-green-400 font-medium">
        {{ readyText }}
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  /**
   * Whether the status is in a ready state
   */
  isReady: boolean
  /**
   * Text to display when not in ready state
   */
  statusText?: string
  /**
   * Text to display when in ready state
   */
  readyText?: string
}

withDefaults(defineProps<Props>(), {
  statusText: '処理中...',
  readyText: '準備ができました！',
})
</script>
