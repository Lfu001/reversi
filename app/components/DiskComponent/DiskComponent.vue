<template>
  <div
    :class="[
      'w-[30px] h-[30px] rounded-full',
      colorClass,
    ]"
  />
</template>

<script setup lang="ts">
import type { DiskColor } from '~/types/DiskColor'

const props = defineProps<{
  color: DiskColor | null
}>()

const colorClass = computed(() => {
  if (props.color === 'Dark') return 'bg-black border border-black'
  if (props.color === 'Light') return 'bg-white border border-black'
  return ''
})
</script>
