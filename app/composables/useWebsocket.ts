/**
 * Create a reactive websocket connection.
 *
 * If the connection is already established, it will be reused.
 * Otherwise, a new connection will be created.
 */
export const useWebsocket = () => {
  /**
   * The websocket connection.
   *
   * Will be `null` until the connection is established.
   * Once the connection is closed, the value will be `null` again.
   */
  const socket = ref<WebSocket | null>(null)

  /**
   * Whether the websocket connection is currently open.
   * Computed from the socket's readyState.
   */
  const isConnected = computed(
    () => socket.value && socket.value.readyState === WebSocket.OPEN,
  )

  /**
   * Connect to the specified websocket URL.
   *
   * If the connection is already established to the same URL, this is a no-op.
   * Otherwise, a new connection will be established.
   *
   * @param url - The URL of the websocket to connect to.
   */
  const connect = (url: string) => {
    if (socket.value?.url === url && isConnected.value) { // if the connection is already established
      return
    }
    socket.value = new WebSocket(url)

    socket.value.onerror = (event) => {
      console.error(event)
    }
  }

  /**
   * Send a message to the websocket.
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
   * Close the websocket connection.
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
   * Set the onmessage handler for the websocket connection.
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

  onUnmounted(() => {
    if (isConnected.value) { // if the connection is open
      socket.value?.close()
    }
  })

  return {
    socket,
    isConnected,
    connect,
    send,
    close,
    setOnMessageHandler,
  }
}
