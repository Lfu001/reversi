<template>
  <div
    v-if="props.isOpen"
    class="fixed inset-0 z-[101] flex items-center justify-center bg-black/30 backdrop-blur-sm"
    @click.self="$emit('close')"
  >
    <div class="relative w-full max-w-lg rounded-2xl bg-white p-6 shadow-xl dark:bg-gray-800">
      <h2 class="mb-4 text-xl font-bold text-black dark:text-white">
        アバターを選択
      </h2>

      <div class="mb-6 grid max-h-[50vh] grid-cols-4 gap-4 overflow-y-auto p-1 sm:grid-cols-5 md:grid-cols-6">
        <button
          v-for="avatar in avatars"
          :key="avatar"
          class="rounded-full transition-transform hover:scale-105 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:outline-none"
          :class="{
            'ring-2 ring-blue-500 ring-offset-2': selectedAvatar === avatar,
          }"
          @click="selectedAvatar = avatar"
        >
          <PlayerAvatar
            :src="`/images/avatars/${avatar}`"
            class="h-full w-full"
          />
        </button>
      </div>

      <div class="flex justify-end space-x-2">
        <Button
          label="キャンセル"
          variant="outlined"
          @click="$emit('close')"
        />
        <Button
          label="OK"
          :disabled="!selectedAvatar"
          @click="handleSelect"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const props = defineProps<{
  /**
   * Whether the modal is currently open.
   */
  isOpen: boolean
}>()

const emit = defineEmits<{
  /**
   * Emitted when the user closes the modal.
   */
  (e: 'close'): void
  /**
   * Emitted when the user select and confirm an avatar.
   * @param avatarUrl The URL of the selected avatar.
   */
  (e: 'select', avatarUrl: string): void
}>()

/**
 * An url of the selected avatar.
 * Initially set to null, and updated when the user selects an avatar.
 */
const selectedAvatar = ref<string | null>(null)

/**
 * An array of avatar file names.
 */
const avatars = [
  'ancient-oak.png',
  'autumn-leaves.png',
  'bamboo-grove.png',
  'browsing-giraffe.png',
  'camouflaged-octopus.png',
  'crystalline-dragon.png',
  'cutting-marlin.png',
  'delicate-fern.png',
  'forest-spirit-deer.png',
  'giant-squid.png',
  'grazing-sheep.png',
  'humpback-whale.png',
  'leaping-dolphin.png',
  'majestic-stag.png',
  'mythical-griffin.png',
  'penguin-on-ice.png',
  'perched-squirrel.png',
  'sea-serpent.png',
  'seal-on-rock.png',
  'shimmering-phoenix.png',
  'soaring-eagle.png',
  'vibrant-succulent.png',
  'wise-owl.png',
]

/**
 * Handles the select event when the user selects and confirms an avatar.
 */
const handleSelect = () => {
  if (selectedAvatar.value) {
    emit('select', `/images/avatars/${selectedAvatar.value}`)
    emit('close')
  }
}
</script>
