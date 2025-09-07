<template>
  <div class="h-full">
    <div class="flex h-full flex-col items-center justify-center">
      <div class="flex flex-col space-y-4">
        <Button
          label="コンピュータと対戦"
          @click="startComputerGame"
        />
        <Button
          label="誰かと対戦"
          @click="openJoinModal"
        />
      </div>
    </div>

    <Teleport to="body">
      <div
        v-if="isJoinModalOpen"
        class="fixed inset-0 z-[100]"
      >
        <ModalJoin
          @close="closeJoinModal"
          @join="joinTable"
        />
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
definePageMeta({
  layout: 'entry-flow',
})

/**
 * Starts a game against the computer
 */
const startComputerGame = () => {
  alert('TODO: Computer vs. player game')
}

/**
 * Reactive variable to manage the visibility state of the join modal
 */
const isJoinModalOpen = ref(false)

/**
 * Opens the join modal
 */
const openJoinModal = () => {
  isJoinModalOpen.value = true
}

/**
 * Closes the join modal
 */
const closeJoinModal = () => {
  isJoinModalOpen.value = false
}

/**
 * Joins a table
 *
 * After joining the table, it will navigate to the table page.
 *
 * @param password The password of the table to join
 */
const joinTable = async (password: string) => {
  try {
    const authStore = useAuthStore()
    const jwt = authStore.jwt
    const response = await $fetch.raw('http://127.0.0.1:8081/join', {
      method: 'POST',
      body: new URLSearchParams({ password }),
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': `Bearer ${jwt}`,
      },
    })
    const nextUrl = response.headers.get('location')
    if (nextUrl) {
      authStore.roomPassword = password
      const webSocketStore = useWebSocketStore()
      webSocketStore.connect(`http://127.0.0.1:8081${nextUrl}?token=${jwt}`)
      webSocketStore.setOnMessageHandler(useBoardStore().handleWebsocketMessage)
      navigateTo(`matchmaking${nextUrl}`)
    }
  }
  catch (error) {
    console.error('Join table error:', error)
    alert('テーブルに参加できませんでした。もう一度お試しください。')
  }
}
</script>
