<template>
  <div
    class="fixed inset-0 z-[101] flex items-center justify-center bg-black/30 backdrop-blur-sm"
    @click.self="$emit('close')"
  >
    <div class="relative w-full max-w-sm rounded-2xl bg-white p-6 shadow-xl dark:bg-gray-800">
      <h2 class="mb-4 text-xl font-bold text-black dark:text-white">
        テーブルに参加または作成
      </h2>
      <div class="mb-4">
        <TextField
          v-model="password"
          placeholder="合言葉"
        />
      </div>
      <div class="flex justify-end space-x-2">
        <Button
          label="キャンセル"
          variant="outlined"
          @click="$emit('close')"
        />
        <Button
          label="参加する"
          :disabled="!isPasswordValid"
          @click="joinTable"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const emit = defineEmits<{
  /** Emitted when the modal is closed */
  (e: 'close'): void
  /**
   * Emitted when the user attempts to join a game
   * @param password - The password entered by the user
   */
  (e: 'join', password: string): void
}>()

/**
 * The password input value
 */
const password = ref('')

/**
 * Computed property to check if the password is valid (not empty)
 */
const isPasswordValid = computed(() => password.value.length > 0)

/**
 * Handles the join action when the join button is clicked
 * Only emits the join event if the password is valid
 */
const joinTable = () => {
  if (isPasswordValid.value) {
    emit('join', password.value)
  }
}
</script>
