<template>
  <div class="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg transition-colors duration-300">
    <div class="relative">
      <div class="flex justify-between items-center mb-2">
        <p
          v-if="label"
          class="text-sm font-medium text-gray-500 dark:text-gray-400"
        >
          {{ label }}
        </p>
        <transition
          enter-active-class="transition-opacity duration-200"
          leave-active-class="transition-opacity duration-200"
          enter-from-class="opacity-0"
          leave-to-class="opacity-0"
        >
          <p
            v-if="showCopied"
            class="text-xs text-green-500 dark:text-green-400"
          >
            {{ copiedMessage }}
          </p>
        </transition>
      </div>
      <div class="flex items-center justify-between bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md px-4 py-2 transition-colors duration-300">
        <span
          :id="id"
          class="font-mono text-lg text-gray-800 dark:text-gray-200"
        >
          {{ value }}
        </span>
        <button
          :title="copyButtonText || 'Copy to clipboard'"
          class="text-blue-500 hover:text-blue-600 dark:text-blue-400 dark:hover:text-blue-300 transition-colors cursor-pointer flex items-center justify-center h-5 w-5"
          @click="copyToClipboard"
        >
          <svg
            v-if="!showCopied"
            xmlns="http://www.w3.org/2000/svg"
            class="h-5 w-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"
            />
          </svg>
          <svg
            v-else
            xmlns="http://www.w3.org/2000/svg"
            class="h-5 w-5 text-green-500 dark:text-green-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M5 13l4 4L19 7"
            />
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  /**
   * The value to be displayed and copied
   */
  value: string
  /**
   * Optional label above the code
   */
  label?: string
  /**
   * Optional ID for the code element
   */
  id?: string
  /**
   * Custom text for the copy button's tooltip
   */
  copyButtonText?: string
  /**
   * Custom message shown after copying
   */
  copiedMessage?: string
}

const props = withDefaults(defineProps<Props>(), {
  label: '',
  id: 'copyable-code',
  copyButtonText: '',
  copiedMessage: 'Copied!',
})

/**
 * Whether the code has been copied to the clipboard
 */
const showCopied = ref(false)

/**
 * Copy the code to the clipboard
 */
const copyToClipboard = async () => {
  try {
    await navigator.clipboard.writeText(props.value)
    showCopied.value = true
    setTimeout(() => {
      showCopied.value = false
    }, 2000)
  }
  catch (err) {
    console.error('Failed to copy text: ', err)
  }
}
</script>
