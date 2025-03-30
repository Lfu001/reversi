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

  return {
    jwt,
  }
})
