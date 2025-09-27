<template>
  <div class="flex w-full items-center justify-center rounded-[15px] bg-[#4a3222] p-4 shadow-lg select-none sm:p-6 md:p-8">
    <div class="w-full rounded-[10px] bg-[radial-gradient(#50aa50,#2d642d)] p-2 sm:p-2.5">
      <div class="relative grid aspect-square w-full grid-cols-8 grid-rows-8 border-r-2 border-b-2 border-black">
        <div
          v-for="(c, index) in props.board"
          :key="index"
          class="flex aspect-square w-full items-center justify-center border-t-2 border-l-2 border-black"
        >
          <DiskComponent
            :color="c"
            :guide-color="isPuttablePosition(index) ? props.positionGuideColor : null"
            @click="onSquareClick(index)"
          />
        </div>

        <div class="absolute top-1/4 left-1/4 mt-px ml-px h-[6px] w-[6px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-black" />
        <div class="absolute top-1/4 left-3/4 mt-px ml-px h-[6px] w-[6px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-black" />
        <div class="absolute top-3/4 left-1/4 mt-px ml-px h-[6px] w-[6px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-black" />
        <div class="absolute top-3/4 left-3/4 mt-px ml-px h-[6px] w-[6px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-black" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { DiskColor } from '~/types/DiskColor'
import { Position } from '~/types/Position'

/**
 * Props for BoardContainer component.
 *
 * @param board An array of 64 disk colors.
 * The index of the array corresponds to the position on the board.
 * A value of `null` indicates an empty square.
 * @param puttablePositions An array of positions where a disk can be placed.
 * @param positionGuideColor The color of the puttable position guide.
 */
interface Props {
  board: Array<DiskColor | null>
  puttablePositions: Array<Position>
  positionGuideColor: DiskColor
}

const props = defineProps<Props>()

const emit = defineEmits<{
  (e: 'onSquareClick', index: number): void
}>()

/**
 * Checks if a given index is a position where a disk can be placed.
 * @param index A number from 0 to 63.
 * @returns Whether the position is a puttable position.
 */
const isPuttablePosition = (index: number): boolean => {
  return props.puttablePositions.some(p => p.equals(Position.fromIndex(index)))
}

/**
 * Handles the event when a user clicks on a square.
 * @param index The index of the square on the board.
 * If the square is already filled, the function does nothing.
 * Otherwise, it is supposed to apply the logic of placing a stone.
 */
const onSquareClick = (index: number) => {
  if (index < 0 || index >= props.board.length) { // If the index is invalid
    console.error('Invalid index', index)
    return
  }
  if (props.board[index] !== null) { // If the square is already filled
    return
  }
  emit('onSquareClick', index)
}
</script>
