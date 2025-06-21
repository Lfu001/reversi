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
            :items="players"
            title="参加プレイヤー"
            empty-message="プレイヤーがいません"
            :max-items="2"
          />
        </div>
      </div>

      <!-- Start Game Button -->
      <Button
        v-if="canStartGame"
        label="ゲームを開始する"
        variant="fill"
        class="w-full py-3 px-4 shadow-md bg-green-600 hover:bg-green-700 transition-colors"
        @click="startGame"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * Player interface
 */
interface Player {
  name: string
}

/**
 * Array of players
 */
const players = ref<Player[]>([
  { name: 'プレイヤー1' },
  { name: 'プレイヤー2' },
])

/**
 * Room code
 */
const roomCode = ref('ABCD12')

/**
 * Whether the game can be started
 */
const canStartGame = computed(() => {
  return players.value.length >= 2
})

/**
 * Start the game
 */
const startGame = () => {
  if (canStartGame.value) {
    // TODO: Implement game start logic with WebSocket
    console.log('Starting game...')
  }
}
</script>
