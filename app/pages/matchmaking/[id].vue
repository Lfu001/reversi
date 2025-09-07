<template>
  <div class="min-h-screen bg-gray-100 dark:bg-gray-900 flex items-center justify-center p-4 transition-colors duration-300">
    <div class="w-full max-w-md space-y-6">
      <!-- Status Message -->
      <StatusMessage
        :is-ready="canStartGame"
        status-text="参加者を待っています..."
      />

      <!-- Room Info Card -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 transition-colors duration-300">
        <h2 class="text-lg font-medium text-gray-700 dark:text-gray-300 mb-4 transition-colors duration-300">
          ルーム情報
        </h2>
        <!-- Room Code -->
        <div class="mb-6">
          <CopyableCode
            :value="roomCode"
            label="合言葉"
          />
        </div>

        <!-- Player List -->
        <div class="border-t border-gray-200 dark:border-gray-700 pt-4">
          <List
            :items="boardStore.players"
            title="参加プレイヤー"
            empty-message="プレイヤーがいません"
            :max-items="2"
          />
        </div>
      </div>

      <!-- Start Game Button -->
      <Button
        v-if="canStartGame"
        :disabled="!isStartButtonEnabled"
        label="ゲームを開始する"
        variant="fill"
        class="w-full py-3 px-4 shadow-md bg-green-600 hover:bg-green-700 transition-colors"
        @click="startGame"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
const boardStore = useBoardStore()
const route = useRoute()

/**
 * Room code
 */
const roomCode = computed(() => useAuthStore().roomPassword ?? '')

/**
 * Whether the game can be started
 */
const canStartGame = computed(() => {
  return boardStore.players.length >= 2
})

/**
 * Whether the start button is enabled
 */
const isStartButtonEnabled = ref(true)

onMounted(() => {
  watch(() => boardStore.hasGameStarted, (hasGameStarted) => {
    if (hasGameStarted) {
      navigateTo(`/table/${route.params.id}`)
    }
  })
})

/**
 * Start the game
 */
const startGame = () => {
  if (canStartGame.value) {
    isStartButtonEnabled.value = false
    useWebSocketStore().send('"Start"')
    navigateTo(`/table/${route.params.id}`)
  }
}
</script>
