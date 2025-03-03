<template>
  <div class="disc" />
</template>

<script setup lang="ts">
import type { DiskColor } from '~/types/DiskColor'

defineProps<{
  color: DiskColor | null
}>()
</script>
