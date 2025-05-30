<template>
  <div class="bg-[#4a3222] p-8 rounded-[15px] shadow-lg flex justify-center items-center w-[380px] h-[380px] select-none">
    <div class="bg-[radial-gradient(#50aa50,#2d642d)] p-2.5 rounded-[10px]">
      <div class="grid grid-cols-8 grid-rows-8 border border-black w-[320px] h-[320px]">
        <div
          v-for="(c, index) in board"
          :key="index"
          class="w-10 h-10 flex items-center justify-center relative border border-black"
          :class="{ 'before:content-[\'\'] before:absolute before:w-[6px] before:h-[6px] before:rounded-full before:bg-black before:top-[-1px] before:left-[-1px] before:transform before:-translate-x-1/2 before:-translate-y-1/2 before:opacity-100': isDotPosition(index) }"
        >
          <DiskComponent
            :color="c"
            @click="placeStone(index)"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { DiskColor } from '~/types/DiskColor'

interface Props {
  board?: Array<DiskColor | null>
}

const { board = Array(64).fill(null) } = defineProps<Props>()

const isDotPosition = (index: number) => {
  const dotPositions = [
    18,
    22,
    50,
    54,
  ]
  return dotPositions.some(i => i === index)
}

// put down disc
const placeStone = (index: number) => {
  // filled cell
  if (board[index] != null) return

  // TODO : apply logic
}
</script>
