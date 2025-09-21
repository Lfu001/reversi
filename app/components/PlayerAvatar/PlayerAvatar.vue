<template>
  <div
    class="relative transition-transform duration-300"
    :class="{ 'scale-110': props.isMyTurn }"
  >
    <div
      v-if="props.name"
      class="absolute top-0 left-1/2 z-10 -translate-x-1/2 -translate-y-1/4 rounded-full bg-gray-800/60 px-3 py-0.5 text-sm whitespace-nowrap text-white backdrop-blur-sm"
    >
      {{ props.name }}
    </div>

    <NuxtImg
      :src="props.src"
      class="h-full w-full rounded-full border-2 border-white object-cover shadow-sm"
      :class="{ 'animate-pulse-glow': props.isMyTurn }"
      format="webp"
    />

    <div
      v-if="props.diskColor"
      class="absolute right-0 bottom-0 z-10 h-1/3 w-1/3 rounded-full"
      :class="diskColorClass"
    />
  </div>
</template>

<script setup lang="ts">
import { DiskColor } from '~/types/DiskColor'

const props = withDefaults(
  defineProps<{
    /**
     * The URL of the image to display.
     */
    src: string
    /**
     * The name of the player.
     * If specified, it will be displayed on the player avatar.
     */
    name?: string
    /**
     * The color of the disk.
     * If specified, a disk will be displayed on the player avatar.
     */
    diskColor?: DiskColor
    /**
     * The size of the player avatar.
     */
    size?: number
    /**
     * Whether it is the player's turn.
     */
    isMyTurn?: boolean
  }>(),
  {
    isMyTurn: false,
  },
)

/**
 * Computes the class string for the disk based on its color.
 */
const diskColorClass = computed(() => {
  if (props.diskColor === DiskColor.Dark) {
    return 'bg-black'
  }
  if (props.diskColor === DiskColor.Light) {
    return 'bg-white border-1 border-gray-500'
  }
  return ''
})
</script>
