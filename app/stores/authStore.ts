/**
 * The auth store.
 */
export const useAuthStore = defineStore('auth', () => {
  /**
   * The JWT token.
   * It is `null` if the user is not authenticated.
   */
  const jwt = ref<string | null>(null)

  /**
   * The password of the room.
   * It is `null` if the user has not entered the room password yet.
   */
  const roomPassword = ref<string | null>(null)

  /**
   * The URL of the user's avatar.
   */
  const avatarUrl = ref<string>('/images/avatars/autumn-leaves.png')

  /**
   * Sets the URL of the user's avatar and updates it on the server.
   * @param url The new avatar URL.
   */
  async function setAvatarUrl(url: string) {
    if (!jwt.value) {
      console.error('Cannot set avatar without JWT.')
      return
    }

    await $fetch('http://127.0.0.1:8081/playerProfile', {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': `Bearer ${jwt.value}`,
      },
      body: new URLSearchParams({ avatar_url: url }),
      async onResponse({ response }) {
        if (response.status != 200) {
          console.error(`Failed to update avatar: ${response._data}`)
          return
        }
        avatarUrl.value = url
      },
      async onResponseError({ response }) {
        const responseMessage = await response.text()
        console.error(`Failed to update avatar: ${responseMessage}`)
      },
    })
  }

  return {
    jwt,
    roomPassword,
    avatarUrl,
    setAvatarUrl,
  }
})
