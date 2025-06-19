<template>
  <div class="score-board bg-white rounded-2xl shadow-md p-6 flex justify-between items-end w-full max-w-md mx-auto">
    <!-- Left Player -->
    <div class="text-center">
      <p class="text-6xl font-extrabold text-black">
        {{ blackScore }}
      </p>
      <p class="text-blue-600 text-xl font-bold mt-2">
        {{ blackName }}
      </p>
    </div>

    <!-- VS -->
    <div class="text-center text-gray-400 text-3xl font-bold mx-4">
      VS
    </div>

    <!-- Right Player -->
    <div class="text-center">
      <p class="text-6xl font-extrabold text-black stroke-black stroke-2">
        {{ whiteScore }}
      </p>
      <p class="text-green-600 text-xl font-bold mt-2">
        {{ whiteName }}
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  blackScore: number
  whiteScore: number
  blackName: string
  whiteName: string
}>()
</script>
