/**
 * Create a reactive WebSocket connection.
 *
 * If the connection is already established, it will be reused.
 * Otherwise, a new connection will be created.
 */
export const useWebSocketStore = defineStore('websocket', () => {
  /**
   * The WebSocket connection.
   *
   * Will be `null` until the connection is established.
   * Once the connection is closed, the value will be `null` again.
   */
  const socket = ref<WebSocket | null>(null)

  /**
   * Whether the WebSocket connection is currently open.
   * Computed from the socket's readyState.
   */
  const isConnected = computed(
    () => socket.value && socket.value.readyState === WebSocket.OPEN,
  )

  /**
   * Connect to the specified WebSocket URL.
   *
   * If the connection is already established to the same URL, this is a no-op.
   * Otherwise, a new connection will be established.
   *
   * @param url - The URL of the WebSocket to connect to.
   * @param token - The JWT token to authenticate with.
   */
  const connect = (url: string, token: string) => {
    if (socket.value?.url === url && isConnected.value) { // if the connection is already established
      return
    }
    socket.value = new WebSocket(url)

    socket.value.onopen = () => {
      send(JSON.stringify({ Authenticate: token }))
    }
    socket.value.onerror = (event) => {
      console.error(event)
    }
  }

  /**
   * Send a message to the WebSocket.
   *
   * If the connection is open, the message will be sent immediately.
   * Otherwise, the message will be discarded.
   *
   * @param data - The message to send.
   */
  const send = (data: string) => {
    if (isConnected.value) { // if the connection is open
      socket.value?.send(data)
    }
    else {
      console.warn('Cannot send message: WebSocket is not open')
    }
  }

  /**
   * Close the WebSocket connection.
   *
   * If the connection is open, this will initiate the closing handshake.
   * Otherwise, this is a no-op.
   */
  const close = () => {
    if (isConnected.value) { // if the connection is open
      socket.value?.close()
    }
  }

  /**
   * Set the onopen handler for the WebSocket connection.
   *
   * @param handler - The handler to set.
   */
  const setOnOpenHandler = (handler: (event: Event) => void) => {
    if (socket.value) {
      socket.value.onopen = handler
    }
  }

  /**
   * Set the onmessage handler for the WebSocket connection.
   *
   * @param handler - The handler to set.
   */
  const setOnMessageHandler = (handler: (event: MessageEvent) => void) => {
    if (socket.value) {
      socket.value.onmessage = handler
    }
    else {
      console.warn('Cannot set onMessageHandler: WebSocket is not open')
    }
  }

  return {
    socket,
    isConnected,
    connect,
    send,
    close,
    setOnOpenHandler,
    setOnMessageHandler,
  }
})
