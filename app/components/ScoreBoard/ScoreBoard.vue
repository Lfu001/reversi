<template>
  <div class="bg-white/70 dark:bg-gray-800/70 rounded-2xl shadow-md p-6 flex justify-between items-end w-full max-w-md mx-auto">
    <!-- Left Player -->
    <div class="text-center">
      <p class="text-6xl font-extrabold text-black dark:text-white">
        {{ darkScore }}
      </p>
      <p class="text-blue-600 dark:text-blue-400 text-xl font-bold mt-2">
        {{ darkName }}
      </p>
    </div>

    <!-- VS -->
    <div class="text-3xl self-center text-gray-400 dark:text-gray-300 font-bold mx-4">
      VS
    </div>

    <!-- Right Player -->
    <div class="text-center">
      <p class="text-6xl font-extrabold text-black dark:text-white stroke-black dark:stroke-white stroke-2">
        {{ lightScore }}
      </p>
      <p class="text-green-600 dark:text-green-400 text-xl font-bold mt-2">
        {{ lightName }}
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  darkScore: number
  lightScore: number
  darkName: string
  lightName: string
}>()
</script>
