<template>
  <div class="h-full">
    <div class="absolute top-4 right-4 flex items-center gap-4">
      <button
        type="button"
        class="group relative rounded-full transition-transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
        @click="openAvatarModal"
      >
        <PlayerAvatar
          :src="authStore.avatarUrl"
          class="h-16 w-16"
        />
        <div class="absolute inset-0 flex items-center justify-center rounded-full bg-black/50 opacity-0 transition-opacity group-hover:opacity-100">
          <Cog6ToothIcon class="h-8 w-8 text-white" />
        </div>
      </button>
    </div>

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
      <ModalJoin
        :is-open="isJoinModalOpen"
        @close="closeJoinModal"
        @join="joinTable"
      />

      <ModalAvatarSelect
        :is-open="isAvatarModalOpen"
        @close="closeAvatarModal"
        @select="handleAvatarSelect"
      />
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { Cog6ToothIcon } from '@heroicons/vue/24/solid'

definePageMeta({
  layout: 'entry-flow',
})

const authStore = useAuthStore()

/**
 * Starts a game against the computer
 */
const startComputerGame = () => {
  alert('TODO: Computer vs. player game')
}

/**
 * Reactive variable to manage the visibility state of the avatar selection modal
 */
const isAvatarModalOpen = ref(false)

/**
 * Handles the event when an avatar is selected
 */
const handleAvatarSelect = async (avatarUrl: string) => {
  await authStore.setAvatarUrl(avatarUrl)
}

/**
 * Opens the avatar selection modal
 */
const openAvatarModal = () => {
  isAvatarModalOpen.value = true
}

/**
 * Closes the avatar selection modal
 */
const closeAvatarModal = () => {
  isAvatarModalOpen.value = false
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
    const jwt = authStore.jwt
    const response = await $fetch.raw('http://127.0.0.1:8081/join', {
      method: 'POST',
      body: new URLSearchParams({ password }),
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': `Bearer ${jwt!}`,
      },
    })
    const nextUrl = response.headers.get('location')
    if (nextUrl) {
      authStore.roomPassword = password
      const webSocketStore = useWebSocketStore()
      webSocketStore.connect(`http://127.0.0.1:8081${nextUrl}`)
      webSocketStore.setOnOpenHandler(() => {
        webSocketStore.send(JSON.stringify({ Authenticate: jwt! }))
      })
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
