<template>
  <div class="h-10 w-full">
    <input
      v-model="model"
      type="text"
      :placeholder="placeholder"
      class="rounded-full bg-slate-200/80 px-3 py-2 text-gray-700 transition-colors duration-300 select-none focus:ring-2 focus:ring-blue-500 focus:outline-none dark:bg-slate-600/80 dark:text-gray-200"
    >
  </div>
</template>

<script setup lang="ts">
defineProps({
  /**
   * A placeholder string to show when the input is empty.
   */
  placeholder: String,
})

/**
 * A two-way data binding to the input value.
 *
 * @type {Ref<string>}
 */
const model = defineModel<string>()
</script>
