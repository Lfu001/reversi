<template>
  <div :class="['disk', colorClass]" />
</template>

<script setup lang="ts">
import type { DiskColor } from '~/types/DiskColor'

const props = defineProps<{
  color: DiskColor | null
}>()

const colorClass = computed(() => {
  if (props.color === 'Dark') return 'dark-disk'
  if (props.color === 'Light') return 'light-disk'
  return ''
})
</script>

<style scoped>
.disk {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  background-color: transparent;
}

.dark-disk {
  background-color: black;
  border: 1px solid #000;
}

.light-disk {
  background-color: white;
  border: 1px solid #000;
}
</style>
