/**
 * The auth store.
 * It is used to store the JWT token.
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

  return {
    jwt,
    roomPassword,
  }
})
