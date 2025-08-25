<template>
  <div class="flex flex-col w-64 gap-4">
    <ScoreBoard
      :dark-score="props.darkScore"
      :light-score="props.lightScore"
      :dark-name="props.darkName"
      :light-name="props.lightName"
    />
    <HistoryBoard :turns="props.turns" />
  </div>
</template>

<script setup lang="ts">
import ScoreBoard from '../ScoreBoard/ScoreBoard.vue'
import HistoryBoard from '../HistoryPanel/HistoryPanel.vue'
import type { Turn } from '~/types/Turn'

const props = defineProps<{
  darkScore: number
  lightScore: number
  darkName: string
  lightName: string
  turns: Turn[]
}>()
</script>
