<template>
  <div class="relative flex w-4/5 aspect-square items-center justify-center">
    <div
      v-if="color"
      class="aspect-square w-full max-w-16 rounded-full"
      :class="colorClass"
    />
    <div
      v-else-if="guideColor"
      class="aspect-square w-1/4 max-w-3 rounded-full"
      :class="guideColorClass"
    />
  </div>
</template>

<script setup lang="ts">
import { DiskColor } from '~/types/DiskColor'

/**
 * Props for the DiskComponent component.
 *
 * @param color The color of the disk, or null if the disk is not drawn.
 * @param guideColor The color of the guide, or undefined if the guide is not drawn.
 */
const props = defineProps<{
  color: DiskColor | null
  guideColor: DiskColor | null
}>()

/**
 * Computes the class string for the disk based on its color.
 */
const colorClass = computed(() => {
  if (props.color === DiskColor.Dark) {
    return 'bg-black border border-black'
  }
  else if (props.color === DiskColor.Light) {
    return 'bg-white border border-black'
  }
  return ''
})

/**
 * Computes the class string for the guide based on its color.
 */
const guideColorClass = computed(() => {
  if (props.guideColor === DiskColor.Dark) {
    return 'bg-black/30'
  }
  else if (props.guideColor === DiskColor.Light) {
    return 'bg-white/30'
  }
  return ''
})
</script>
