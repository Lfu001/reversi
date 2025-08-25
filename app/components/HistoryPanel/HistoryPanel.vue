<template>
  <div class="bg-white bg-opacity-70 rounded-2xl p-4 w-64 shadow-md">
    <div class="space-y-4 overflow-y-auto max-h-96 pr-2">
      <div
        v-for="(turn, index) in turns"
        :key="index"
        class="border-b border-gray-300 pb-1"
      >
        <div class="font-bold">
          ターン {{ index + 1 }}
        </div>
        <div>{{ turn.player }} は {{ turn.move }} に置いた</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { Turn } from '~/types/Turn'

defineProps<{
  turns: Turn[]
}>()
</script>
