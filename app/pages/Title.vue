<template>
  <div class="flex flex-col items-center justify-center">
    <h1
      class="select-none mb-30 content-center text-5xl font-black text-black dark:text-white"
    >
      AIリバーシ
    </h1>
    <div class="flex flex-col items-center justify-center">
      <TextField
        v-model="guestName"
        class="mb-4"
        placeholder="ゲスト名"
      />
      <Button
        label="次へ →"
        :disabled="!isGuestNameValid"
        @click="registerPlayer"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
definePageMeta({
  layout: 'entry-flow',
})

/**
 * The name of the guest who is currently using the app.
 */
const guestName = ref('')

/**
 * A computed property that checks if the guest name is valid.
 * The name is considered valid if its length is between 1 and 20 characters.
 *
 * @returns {boolean} True if the guest name is valid, false otherwise.
 */
const isGuestNameValid = computed(() => 0 < guestName.value.length && guestName.value.length <= 20)

/**
 * Registers the player by sending a POST request with the guest name.
 * If successful, stores the returned JWT and links to the next scene.
 * If unsuccessful, alerts the user with the error message.
 */
const registerPlayer = async () => {
  await useFetch('/players', {
    method: 'POST',
    body: new URLSearchParams({ name: guestName.value }),
    async onRequest({ options }) {
      options.headers.set('Content-Type', 'application/x-www-form-urlencoded')
      options.headers.set('Accept', 'application/json')
    },
    async onResponse({ response }) {
      const authStore = useAuthStore()
      authStore.jwt = response._data.token

      navigateTo('/menu')
    },
    async onResponseError({ response }) {
      const responseMessage = await response.text()
      alert(`Failed to register: ${responseMessage}`)
    },
  })
}
</script>
