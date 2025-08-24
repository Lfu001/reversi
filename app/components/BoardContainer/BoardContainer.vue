<template>
  <div class="flex w-full max-w-[90vw] items-center justify-center rounded-[15px] bg-[#4a3222] p-4 shadow-lg select-none sm:max-w-[80vw] sm:p-6 md:max-w-[700px] md:p-8">
    <div class="w-full rounded-[10px] bg-[radial-gradient(#50aa50,#2d642d)] p-2 sm:p-2.5">
      <div class="grid aspect-square w-full grid-cols-8 grid-rows-8 border-r-2 border-b-2 border-black">
        <div
          v-for="(c, index) in board"
          :key="index"
          class="relative h-0 w-full border-t-2 border-l-2 border-black pb-[100%]"
          :class="{
            'before:absolute before:top-[-1px] before:left-[-1px] before:h-[6px] before:w-[6px] before:-translate-x-1/2 before:-translate-y-1/2 before:transform before:rounded-full before:bg-black before:opacity-100 before:content-[\'\']':
              isDotPosition(index),
          }"
        >
          <DiskComponent
            :color="c"
            @click="onSquareClick(index)"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { DiskColor } from '~/types/DiskColor'

/**
 * Props for BoardContainer component.
 * @property An array of 64 disk colors.
 * The index of the array corresponds to the position on the board.
 * A value of `null` indicates an empty square.
 */
interface Props {
  board?: Array<DiskColor | null>
}

const { board = Array(64).fill(null) } = defineProps<Props>()

const emit = defineEmits<{
  (e: 'onSquareClick', index: number): void
}>()

/**
 * Checks if a given index is a position where a dot is drawn.
 * @param index A number from 0 to 63.
 * @returns Whether the position is a dot position.
 */
const isDotPosition = (index: number): boolean => {
  const dotPositions = [18, 22, 50, 54]
  return dotPositions.some(i => i === index)
}

/**
 * Handles the event when a user clicks on a square.
 * @param index The index of the square on the board.
 * If the square is already filled, the function does nothing.
 * Otherwise, it is supposed to apply the logic of placing a stone.
 */
const onSquareClick = (index: number) => {
  if (index < 0 || index >= board.length) { // If the index is invalid
    console.error('Invalid index', index)
    return
  }
  if (board[index] !== null) { // If the square is already filled
    return
  }
  emit('onSquareClick', index)
}
</script>
