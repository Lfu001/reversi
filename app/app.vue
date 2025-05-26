<template>
  <div>
    <BoardContainer :board="board" />
  </div>
</template>

<script setup lang="ts">
const board = Array(64).fill(null)
</script>
