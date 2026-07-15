import { ref } from 'vue'

// Session duration (seconds) — set to very large to remove time limit
const SESSION_DURATION_SECONDS = 99999

// Control frame send interval (ms)
const CONTROL_INTERVAL_MS = 42
// TCP mode max frame buffer count
const FRAME_BUFFER_MAX = 36
// Heartbeat timeout threshold (ms)
const HEARTBEAT_TIMEOUT_MS = 5000
// Connection timeout threshold (ms)
const CONNECTION_TIMEOUT_MS = 30000

/**
 * Determine backend environment based on URL parameter 'level'
 */
function getLevel() {
  return (new URLSearchParams(location.search).get('level') || 'main').toLowerCase().trim()
}

/**
 * Get backend HTTP API base URL
 *
 * Priority:
 *   1. URL param ?api=<host:port>  — explicitly specify backend address
 *   2. Empty string (relative path)  — works with both Vite proxy and backend serving frontend
 *
 * Usage:
 *   Dev mode (Vite dev server): no ?api needed, Vite proxy auto-forwards /api/* → localhost:8889
 *   Prod mode (backend serves dist): no ?api needed, relative path hits backend directly
 *   Remote debug: ?api=192.168.1.100:8889
 */
export function getApiBaseUrl() {
  const params = new URLSearchParams(location.search)
  const apiOverride = (params.get('api') || '').trim()
  if (apiOverride) {
    return apiOverride.startsWith('http') ? apiOverride : `http://${apiOverride}`
  }
  // Default to relative path (works with Vite proxy and backend direct serve)
  return ''
}

/**
 * Get backend WebSocket base URL
 *
 * Same priority as getApiBaseUrl:
 *   1. URL param ?api=<host:port>
 *   2. Current page host (ws://current_host) — via Vite proxy or direct backend connection
 */
export function getWsBaseUrl() {
  const params = new URLSearchParams(location.search)
  const apiOverride = (params.get('api') || '').trim()
  if (apiOverride) {
    const clean = apiOverride.replace(/^https?:\/\//, '')
    const wsProto = location.protocol === 'https:' ? 'wss' : 'ws'
    return `${wsProto}://${clean}`
  }
  // Default to current page host (Vite proxy forwards /ws/* → localhost:8889)
  const wsProto = location.protocol === 'https:' ? 'wss' : 'ws'
  return `${wsProto}://${location.host}`
}

/**
 * Build ICE server configuration (supports TURN relay)
 */
function buildIceServers() {
  const params = new URLSearchParams(location.search)
  const enableTurn = ['1', 'true', 'yes', 'on'].includes(String(params.get('turn') || '').toLowerCase())
  const iceServers = [{ urls: 'stun:stun.l.google.com:19302' }]

  if (!enableTurn) return iceServers

  const turnHost = (params.get('turn_host') || '').trim()
  const turnPort = (params.get('turn_port') || '3478').trim()
  const turnUser = (params.get('turn_user') || '').trim()
  const turnPass = (params.get('turn_pass') || '').trim()

  if (turnHost && turnUser && turnPass) {
    iceServers.push({
      urls: [
        `turn:${turnHost}:${turnPort}?transport=udp`,
        `turn:${turnHost}:${turnPort}?transport=tcp`,
      ],
      username: turnUser,
      credential: turnPass,
    })
  }
  return iceServers
}

/**
 * Modify SDP to force H264 codec priority (payload type 96)
 */
function rewriteSdpForH264(sdp) {
  const lines = sdp.split('\r\n')
  let h264PayloadType = null

  for (const line of lines) {
    const match = line.match(/^a=rtpmap:(\d+)\s+H264\/90000$/)
    if (match) {
      h264PayloadType = match[1]
      break
    }
  }

  if (!h264PayloadType) return sdp

  // Simplify m=video payload list to keep only H264
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith('m=video ')) {
      const parts = lines[i].split(' ')
      lines[i] = `${parts[0]} ${parts[1]} ${parts[2]} 96`
    }
  }

  const remapLine = (line) => {
    line = line.replace(new RegExp(`^a=rtpmap:${h264PayloadType}\\b`), 'a=rtpmap:96')
    line = line.replace(new RegExp(`^a=fmtp:${h264PayloadType}\\b`), 'a=fmtp:96')
    line = line.replace(new RegExp(`^a=rtcp-fb:${h264PayloadType}\\b`), 'a=rtcp-fb:96')
    return line
  }

  const filteredLines = []
  for (const line of lines) {
    const payloadMatch = line.match(/^(?:a=rtpmap:|a=fmtp:|a=rtcp-fb:)(\d+)\b/)
    if (payloadMatch) {
      const payloadType = payloadMatch[1]
      if (payloadType !== h264PayloadType) continue
    }
    filteredLines.push(remapLine(line))
  }

  return filteredLines.join('\r\n')
}

/**
 * Main transport composable
 * Manages connection lifecycle for both WebRTC (UDP) and WebSocket (TCP) transport modes
 */
export function useTransport() {
  // Transport mode: 'udp' (WebRTC) or 'tcp' (WebSocket frame stream)
  const transportMode = ref('udp')
  // Debug status text
  const status = ref('Disconnected')
  // Connection state: 'disconnected' | 'connecting' | 'connected' | 'error'
  const connectionState = ref('disconnected')
  // Error type: '' | 'connection' | 'server' | 'high_demand' | 'queue_timeout'
  const errorType = ref('')
  // Whether to show loading overlay
  const loading = ref(true)
  // Loading text
  const loadingText = ref('Loading…')
  // Current session ID (from server)
  const sessionId = ref(null)
  // Current scene name
  const currentScene = ref(null)
  // Whether to show video/frame display
  const showVideo = ref(false)
  // WebRTC media stream (UDP mode)
  const mediaStream = ref(null)
  // Current frame image URL (TCP mode)
  const frameSrc = ref('')
  // Render frame image URL (TCP mode, when show_render enabled)
  const renderFrameSrc = ref('')
  // Whether server enabled render frame push (for UI display)
  const showRender = ref(false)
  // Whether server sends tagged frames (internal protocol, not affected by UI toggle)
  let serverSendsTaggedFrames = false
  // Whether first frame received
  const firstFrameReceived = ref(false)
  // Whether currently in queue
  const isInQueue = ref(false)
  // Time up callback (set externally)
  const onTimeUp = ref(null)
  // Explore mode: true = render-only, false = denoising active
  const exploreMode = ref(false)
  // Whether starting point was set by user
  const startingPointSet = ref(false)
  // Recording state
  const isRecording = ref(false)
  const recordingProcessing = ref(false)

  // Internal state
  let activeSessionCounter = 0
  let currentSessionId = 0
  let signalingWebSocket = null
  let peerConnection = null
  let pendingIceCandidates = []

  // TCP frame buffer
  let frameBuffer = []
  let currentFrameUrl = null
  let framePlaybackTimer = null

  // Render frame buffer
  let renderFrameBuffer = []
  let currentRenderFrameUrl = null

  // Heartbeat related
  let heartbeatCheckTimer = null
  let connectionTimeoutTimer = null
  let lastDataReceivedAt = null
  let lastPingSentAt = null

  // ─── Heartbeat ───────────────────────────────────────────────────────────────────

  function sendPing() {
    if (signalingWebSocket && signalingWebSocket.readyState === WebSocket.OPEN) {
      signalingWebSocket.send(JSON.stringify({ type: 'ping' }))
      lastPingSentAt = Date.now()
      console.log('[Heartbeat] Ping sent')
    }
  }

  function startHeartbeat() {
    stopHeartbeat()
    lastDataReceivedAt = Date.now()
    lastPingSentAt = Date.now()

    heartbeatCheckTimer = setInterval(() => {
      if (lastPingSentAt && Date.now() - lastPingSentAt >= HEARTBEAT_TIMEOUT_MS) {
        sendPing()
      }
    }, 500)

    connectionTimeoutTimer = setInterval(() => {
      if (!lastDataReceivedAt) return
      if (Date.now() - lastDataReceivedAt > CONNECTION_TIMEOUT_MS) {
        connectionState.value = 'error'
        errorType.value = 'connection'
        status.value = 'Connection timeout (no data)'
        loading.value = true
        loadingText.value = 'Connection timeout. Please check network or reconnect'
      }
    }, 500)
  }

  function stopHeartbeat() {
    if (heartbeatCheckTimer) { clearInterval(heartbeatCheckTimer); heartbeatCheckTimer = null }
    if (connectionTimeoutTimer) { clearInterval(connectionTimeoutTimer); connectionTimeoutTimer = null }
    lastDataReceivedAt = null
    lastPingSentAt = null
  }

  function onDataReceived() {
    lastDataReceivedAt = Date.now()
    if (lastPingSentAt && Date.now() - lastPingSentAt >= HEARTBEAT_TIMEOUT_MS) {
      sendPing()
    }
    if (connectionState.value === 'error') {
      connectionState.value = 'connected'
    }
  }

  // ─── TCP frame playback ──────────────────────────────────────────────────────────────

  function startFramePlayback() {
    if (framePlaybackTimer) return
    framePlaybackTimer = setInterval(() => {
      // Generated frame playback
      if (frameBuffer.length > 0) {
        const nextFrameUrl = frameBuffer.shift()
        frameSrc.value = nextFrameUrl
        if (currentFrameUrl) URL.revokeObjectURL(currentFrameUrl)
        currentFrameUrl = nextFrameUrl
      }
      // Render frame playback
      if (renderFrameBuffer.length > 0) {
        const nextRenderUrl = renderFrameBuffer.shift()
        renderFrameSrc.value = nextRenderUrl
        if (currentRenderFrameUrl) URL.revokeObjectURL(currentRenderFrameUrl)
        currentRenderFrameUrl = nextRenderUrl
      }
    }, CONTROL_INTERVAL_MS)
  }

  function stopFramePlayback() {
    if (framePlaybackTimer) { clearInterval(framePlaybackTimer); framePlaybackTimer = null }
    for (const url of frameBuffer) URL.revokeObjectURL(url)
    frameBuffer = []
    for (const url of renderFrameBuffer) URL.revokeObjectURL(url)
    renderFrameBuffer = []
  }

  // ─── Disconnect ────────────────────────────────────────────────────────────────

  async function disconnect(keepStream = false) {
    activeSessionCounter++
    currentSessionId = activeSessionCounter
    console.log('[Transport] disconnect() called, session counter:', activeSessionCounter)

    connectionState.value = 'disconnected'
    errorType.value = ''
    firstFrameReceived.value = false
    exploreMode.value = false
    startingPointSet.value = false
    isRecording.value = false
    recordingProcessing.value = false

    stopFramePlayback()
    stopHeartbeat()

    const oldWs = signalingWebSocket
    const oldPc = peerConnection
    signalingWebSocket = null
    peerConnection = null

    if (oldWs) {
      console.log('[Transport] Closing old WebSocket, readyState:', oldWs.readyState)
      try { oldWs.close() } catch {}
    }
    if (oldPc) {
      console.log('[Transport] Closing old PeerConnection')
      try { oldPc.close() } catch {}
    }

    if (mediaStream.value) {
      mediaStream.value.getTracks().forEach(track => track.stop())
      if (!keepStream) mediaStream.value = null
    }

    pendingIceCandidates = []
    sessionId.value = null

    if (!keepStream) {
      if (currentFrameUrl) { URL.revokeObjectURL(currentFrameUrl); currentFrameUrl = null }
      frameSrc.value = ''
      if (currentRenderFrameUrl) { URL.revokeObjectURL(currentRenderFrameUrl); currentRenderFrameUrl = null }
      renderFrameSrc.value = ''
    }
  }

  // ─── Send control commands ────────────────────────────────────────────────────────────

  function sendControl(payload) {
    if (signalingWebSocket && signalingWebSocket.readyState === WebSocket.OPEN) {
      signalingWebSocket.send(JSON.stringify({ type: 'control', ...payload }))
    }
  }

  // ─── Explore / Denoise / Recording controls ────────────────────────────────────

  function sendStartDenoising() {
    if (signalingWebSocket && signalingWebSocket.readyState === WebSocket.OPEN) {
      signalingWebSocket.send(JSON.stringify({ type: 'start_denoising' }))
    }
  }

  function sendSetStartingPoint() {
    if (signalingWebSocket && signalingWebSocket.readyState === WebSocket.OPEN) {
      signalingWebSocket.send(JSON.stringify({ type: 'set_starting_point' }))
    }
  }

  function startRecording() {
    if (signalingWebSocket && signalingWebSocket.readyState === WebSocket.OPEN) {
      signalingWebSocket.send(JSON.stringify({ type: 'start_recording' }))
    }
  }

  function stopRecording() {
    if (signalingWebSocket && signalingWebSocket.readyState === WebSocket.OPEN) {
      signalingWebSocket.send(JSON.stringify({ type: 'stop_recording' }))
    }
  }

  function isConnected() {
    return signalingWebSocket !== null && signalingWebSocket.readyState === WebSocket.OPEN
  }

  function handleVideoReady() {
    showVideo.value = true
    loading.value = false
  }

  // ─── TCP mode connection ────────────────────────────────────────────────────────────

  async function connectTcp(sceneName, thisSessionId) {
    console.log('[TCP] Starting TCP connection for session', thisSessionId)
    status.value = 'WS(frames) connecting…'
    loadingText.value = 'Loading world…'

    const wsUrl = `${await getWsBaseUrl()}/ws/frames`
    console.log('[TCP] Creating WebSocket to', wsUrl)

    const ws = new WebSocket(wsUrl)
    ws.binaryType = 'blob'
    ws._sessionId = thisSessionId
    signalingWebSocket = ws

    const isActiveSession = () => ws._sessionId === currentSessionId

    let isFirstFrame = true

    ws.onopen = () => {
      console.log('[TCP] WebSocket opened, sessionId:', ws._sessionId, 'activeSessionId:', currentSessionId)
      if (!isActiveSession()) {
        console.log('[TCP] Session no longer active, ignoring onopen')
        return
      }
      connectionState.value = 'connecting'
      status.value = 'WS(frames) connected'
      console.log('[TCP] Sending start message for scene:', sceneName)
      ws.send(JSON.stringify({ type: 'start', scene: sceneName }))
      startHeartbeat()
      startFramePlayback()
    }

    ws.onmessage = async (event) => {
      if (!isActiveSession()) return
      onDataReceived()

      if (typeof event.data === 'string') {
        try {
          const message = JSON.parse(event.data)
          if (message.type === 'pong') return

          if (message.type === 'timeup') {
            console.log('[TCP] Received timeup message from server')
            if (onTimeUp.value && typeof onTimeUp.value === 'function') onTimeUp.value()
            if (ws && ws.readyState === WebSocket.OPEN) ws.close()
            return
          }

          if (message.type === 'error') {
            connectionState.value = 'error'
            if (message.error === 'server_busy') {
              errorType.value = 'high_demand'
              loading.value = true
              loadingText.value = `Server busy: ${message.message || 'All resources occupied, please try again later'}`
              status.value = 'Server busy'
            } else if (message.error === 'queue_timeout') {
              errorType.value = 'queue_timeout'
              loading.value = true
              loadingText.value = `Queue timeout: ${message.message || 'Waited too long, please reconnect'}`
              status.value = 'Queue timeout'
            } else {
              errorType.value = 'server'
              status.value = `Backend error: ${message.error || 'unknown'}`
              loadingText.value = `Error occurred: ${message.message || message.error || 'Unknown error'}`
            }
            return
          }

          if (message.type === 'queue_status') {
            if (message.status === 'waiting') {
              connectionState.value = 'connecting'
              loading.value = true
              isInQueue.value = true
              loadingText.value = `In queue… (Position ${message.position})`
            } else if (message.status === 'active') {
              connectionState.value = 'connecting'
              isInQueue.value = false
              loadingText.value = 'Resources allocated, connecting…'
            }
          }

          // ── Explore / Denoise mode changes ────────────────────────────
          if (message.type === 'mode_change') {
            exploreMode.value = (message.mode === 'explore')
            if (exploreMode.value) startingPointSet.value = false
            console.log('[TCP] Mode changed to:', message.mode)
          }

          if (message.type === 'starting_point_set') {
            startingPointSet.value = true
            console.log('[TCP] Starting point set')
          }

          // ── Recording messages ────────────────────────────────────────
          if (message.type === 'recording_started') {
            isRecording.value = true
            recordingProcessing.value = false
            console.log('[TCP] Recording started')
          }

          if (message.type === 'recording_processing') {
            recordingProcessing.value = true
            console.log('[TCP] Recording processing...')
          }

          if (message.type === 'recording_complete') {
            isRecording.value = false
            recordingProcessing.value = false
            console.log('[TCP] Recording complete:', message.downloads)
            // Trigger browser downloads for each video
            if (message.downloads) {
              const apiBase = getApiBaseUrl()
              for (const [name, url] of Object.entries(message.downloads)) {
                const fullUrl = apiBase ? apiBase + url : url
                const a = document.createElement('a')
                a.href = fullUrl
                a.download = `${name}.mp4`
                document.body.appendChild(a)
                a.click()
                document.body.removeChild(a)
              }
            }
          }
        } catch {}
        return
      }

      // Binary frame data
      const blob = event.data
      if (!(blob instanceof Blob)) return

      if (serverSendsTaggedFrames && blob.size > 1) {
        // Tagged protocol: first byte 'G'(0x47)=generated frame, 'R'(0x52)=render frame
        const headerByte = await blob.slice(0, 1).arrayBuffer()
        const tag = new Uint8Array(headerByte)[0]
        const payload = blob.slice(1)

        if (tag === 0x52) {
          // Render frame 'R'
          const renderUrl = URL.createObjectURL(payload)
          renderFrameBuffer.push(renderUrl)
          while (renderFrameBuffer.length > FRAME_BUFFER_MAX) {
            URL.revokeObjectURL(renderFrameBuffer.shift())
          }
        } else {
          // Generated frame 'G' or unknown tag
          const frameUrl = URL.createObjectURL(payload)
          frameBuffer.push(frameUrl)
          while (frameBuffer.length > FRAME_BUFFER_MAX) {
            URL.revokeObjectURL(frameBuffer.shift())
          }
        }
      } else {
        // No prefix (backward compatible)
        const frameUrl = URL.createObjectURL(blob)
        frameBuffer.push(frameUrl)
        while (frameBuffer.length > FRAME_BUFFER_MAX) {
          URL.revokeObjectURL(frameBuffer.shift())
        }
      }

      if (isFirstFrame) {
        isFirstFrame = false
        showVideo.value = true
        connectionState.value = 'connected'
        loading.value = false
        firstFrameReceived.value = true
      }
    }

    ws.onclose = (event) => {
      console.log('[TCP] WebSocket closed, code:', event.code, 'wasClean:', event.wasClean, 'sessionId:', ws._sessionId, 'activeSessionId:', currentSessionId)
      if (!isActiveSession()) {
        console.log('[TCP] Session no longer active, ignoring onclose (intentional disconnect)')
        return
      }
      console.log('[TCP] Active session closed by remote, showing error')
      stopFramePlayback()
      stopHeartbeat()
      if (connectionState.value === 'error') {
        console.log('[TCP] Already in error state, not overriding')
        return
      }
      connectionState.value = 'error'
      errorType.value = 'connection'
      loading.value = true
      loadingText.value = event.wasClean
        ? 'Connection closed by server. Please reconnect'
        : 'Connection closed unexpectedly. Please reconnect'
    }

    ws.onerror = () => {
      console.log('[TCP] WebSocket error, sessionId:', ws._sessionId, 'activeSessionId:', currentSessionId)
      if (!isActiveSession()) {
        console.log('[TCP] Session no longer active, ignoring onerror')
        return
      }
      console.log('[TCP] Active session error')
      connectionState.value = 'error'
      errorType.value = 'connection'
      status.value = 'WS(frames) connection error'
      loading.value = true
      loadingText.value = 'Network connection failed. Please check network or backend service'
    }
  }

  // ─── WebRTC mode connection ─────────────────────────────────────────────────────────

  async function connectWebRtc(sceneName, thisSessionId) {
    console.log('[WebRTC] Starting WebRTC connection for session', thisSessionId)
    status.value = 'Loading world…'
    loadingText.value = 'Loading world…'

    const iceServers = buildIceServers()
    const pc = new RTCPeerConnection({ iceServers })
    peerConnection = pc
    pc._sessionId = thisSessionId

    const isActiveSession = () => pc._sessionId === currentSessionId

    pc.onconnectionstatechange = () => {
      const state = pc.connectionState
      console.log('[WebRTC] Connection state changed to:', state, 'sessionId:', pc._sessionId, 'activeSessionId:', currentSessionId)
      if (!isActiveSession()) {
        console.log('[WebRTC] Session no longer active, ignoring state change')
        return
      }
      status.value = `RTC: ${state}`
      if (state === 'failed') {
        console.log('[WebRTC] Connection failed')
        connectionState.value = 'error'
        errorType.value = 'connection'
        loading.value = true
        loadingText.value = 'WebRTC connection failed. Please check network or firewall settings'
      } else if (state === 'disconnected') {
        console.log('[WebRTC] Connection disconnected')
        connectionState.value = 'error'
        errorType.value = 'connection'
        loading.value = true
        loadingText.value = 'WebRTC connection lost. Please reconnect'
      } else if (state === 'connected') {
        console.log('[WebRTC] Connection established')
        connectionState.value = 'connected'
        errorType.value = ''
      } else if (state === 'connecting') {
        connectionState.value = 'connecting'
      }
    }

    pc.ontrack = (event) => {
      if (isActiveSession()) {
        mediaStream.value = event.streams[0]
      }
    }

    // Receive video only, prefer H264
    const videoTransceiver = pc.addTransceiver('video', { direction: 'recvonly' })
    try {
      const capabilities = RTCRtpReceiver.getCapabilities('video')
      if (capabilities?.codecs) {
        const h264Codecs = capabilities.codecs.filter(
          codec => codec.mimeType === 'video/H264' &&
            (!codec.sdpFmtpLine || codec.sdpFmtpLine.includes('packetization-mode=1'))
        )
        if (h264Codecs.length > 0 && videoTransceiver.setCodecPreferences) {
          videoTransceiver.setCodecPreferences(h264Codecs)
        }
      }
    } catch {}

    const wsUrl = `${await getWsBaseUrl()}/ws/session`
    console.log('[WebRTC] Creating signaling WebSocket to', wsUrl)

    const ws = new WebSocket(wsUrl)
    signalingWebSocket = ws
    ws._sessionId = thisSessionId

    const isWsActiveSession = () => ws._sessionId === currentSessionId

    ws.onopen = async () => {
      console.log('[WebRTC] Signaling WebSocket opened, sessionId:', ws._sessionId, 'activeSessionId:', currentSessionId)
      if (!isWsActiveSession()) {
        console.log('[WebRTC] Session no longer active, ignoring onopen')
        return
      }
      connectionState.value = 'connecting'
      status.value = 'WS connected, creating offer…'
      loading.value = true
      loadingText.value = 'Loading world…'
      startHeartbeat()

      console.log('[WebRTC] Creating and sending offer for scene:', sceneName)
      const offer = await pc.createOffer()
      const rewrittenOffer = {
        type: offer.type,
        sdp: rewriteSdpForH264(offer.sdp),
      }
      await pc.setLocalDescription(rewrittenOffer)
      ws.send(JSON.stringify({
        type: 'offer',
        scene: sceneName,
        sdp: pc.localDescription.sdp,
        sdpType: pc.localDescription.type,
      }))
    }

    ws.onmessage = async (event) => {
      if (!isWsActiveSession()) return
      onDataReceived()

      const message = JSON.parse(event.data)
      if (message.type === 'pong') return

      if (message.type === 'timeup') {
        console.log('[WebRTC] Received timeup message from server')
        if (onTimeUp.value && typeof onTimeUp.value === 'function') onTimeUp.value()
        if (ws && ws.readyState === WebSocket.OPEN) ws.close()
        if (pc && pc.connectionState !== 'closed') pc.close()
        return
      }

      if (message.type === 'answer') {
        sessionId.value = message.id || null
        await pc.setRemoteDescription({ type: 'answer', sdp: message.sdp })
        // Process pending ICE candidates
        for (const candidate of pendingIceCandidates) {
          try {
            await pc.addIceCandidate({
              candidate: candidate.candidate,
              sdpMLineIndex: candidate.sdpMLineIndex,
            })
          } catch {}
        }
        pendingIceCandidates = []
        status.value = 'Connected (RTC+WS)'
        loading.value = true
        loadingText.value = 'Waiting for first frame…'
        return
      }

      if (message.type === 'candidate') {
        if (message.candidate) {
          if (!pc.remoteDescription) {
            pendingIceCandidates.push(message)
            return
          }
          try {
            await pc.addIceCandidate({
              candidate: message.candidate,
              sdpMLineIndex: message.sdpMLineIndex,
            })
          } catch {}
        }
        return
      }

      if (message.type === 'queue_status') {
        if (message.status === 'waiting') {
          connectionState.value = 'connecting'
          loading.value = true
          isInQueue.value = true
          loadingText.value = `In queue… (Position ${message.position}, Est. ${message.estimated_wait_sec}s)`
        } else if (message.status === 'active') {
          connectionState.value = 'connecting'
          isInQueue.value = false
          loadingText.value = 'Resources allocated, connecting…'
        }
        return
      }

      if (message.type === 'error') {
        connectionState.value = 'error'
        if (message.error === 'server_busy') {
          errorType.value = 'high_demand'
          loading.value = true
          loadingText.value = `Server busy: ${message.message || 'All resources occupied, please try again later'}`
          status.value = 'Server busy'
        } else if (message.error === 'queue_timeout') {
          errorType.value = 'queue_timeout'
          loading.value = true
          loadingText.value = `Queue timeout: ${message.message || 'Waited too long, please reconnect'}`
          status.value = 'Queue timeout'
        } else {
          errorType.value = 'server'
          status.value = `Backend error: ${message.error || 'unknown'}`
          loadingText.value = `Error occurred: ${message.message || message.error || 'Unknown error'}`
        }
      }
    }

    ws.onclose = (event) => {
      console.log('[WebRTC] Signaling WebSocket closed, code:', event.code, 'wasClean:', event.wasClean, 'sessionId:', ws._sessionId, 'activeSessionId:', currentSessionId)
      if (!isWsActiveSession()) {
        console.log('[WebRTC] Session no longer active, ignoring onclose (intentional disconnect)')
        return
      }
      console.log('[WebRTC] Active session closed by remote, showing error')
      stopHeartbeat()
      if (connectionState.value === 'error') {
        console.log('[WebRTC] Already in error state, not overriding')
        return
      }
      connectionState.value = 'error'
      errorType.value = 'connection'
      loading.value = true
      loadingText.value = event.wasClean
        ? 'Signaling connection closed by server. Please reconnect'
        : 'Signaling connection closed unexpectedly. Please reconnect'
    }

    ws.onerror = () => {
      console.log('[WebRTC] Signaling WebSocket error, sessionId:', ws._sessionId, 'activeSessionId:', currentSessionId)
      if (!isWsActiveSession()) {
        console.log('[WebRTC] Session no longer active, ignoring onerror')
        return
      }
      console.log('[WebRTC] Active session error')
      connectionState.value = 'error'
      errorType.value = 'connection'
      status.value = 'WS connection error'
      loading.value = true
      loadingText.value = 'Signaling connection failed. Please check network or backend service'
    }

    pc.onicecandidate = (event) => {
      if (!isActiveSession()) return
      if (!event.candidate) return
      if (!ws || ws.readyState !== WebSocket.OPEN) return
      ws.send(JSON.stringify({
        type: 'candidate',
        candidate: event.candidate.candidate,
        sdpMLineIndex: event.candidate.sdpMLineIndex,
        sdpMid: event.candidate.sdpMid,
      }))
    }
  }

  // ─── Initialize (read transport mode config) ────────────────────────────────────────────

  async function init() {
    const urlTransport = (new URLSearchParams(location.search).get('transport') || '').toLowerCase().trim()
    if (urlTransport === 'tcp' || urlTransport === 'udp') {
      transportMode.value = urlTransport
      return
    }
    try {
      const apiBase = await getApiBaseUrl()
      const response = await fetch(`${apiBase}/api/config`)
      if (response.ok) {
        const config = await response.json()
        const serverTransport = String(config.transport || '').toLowerCase().trim()
        if (serverTransport === 'tcp' || serverTransport === 'udp') {
          transportMode.value = serverTransport
        }
        // Read render frame display config
        if (config.show_render === true) {
          showRender.value = true
          serverSendsTaggedFrames = true
        }
        return
      }
    } catch {}
    transportMode.value = 'tcp'
  }

  // ─── Connect to specified scene ──────────────────────────────────────────────────────────

  async function connect(sceneName) {
    const isSceneSwitch = currentScene.value !== null && currentScene.value !== sceneName
    console.log('[Transport] connect() called for scene:', sceneName, 'isSceneSwitch:', isSceneSwitch)

    connectionState.value = 'connecting'
    errorType.value = ''
    firstFrameReceived.value = false
    isInQueue.value = false
    loading.value = true
    loadingText.value = isSceneSwitch ? 'Switching world…' : 'Loading world…'

    await disconnect(false)

    const thisSessionId = activeSessionCounter
    currentSessionId = thisSessionId
    console.log('[Transport] New session started, thisSessionId:', thisSessionId, 'activeSessionId:', currentSessionId)

    currentScene.value = sceneName
    showVideo.value = transportMode.value !== 'tcp'

    if (transportMode.value === 'tcp') {
      console.log('[Transport] Starting TCP connection')
      await connectTcp(sceneName, thisSessionId)
    } else {
      console.log('[Transport] Starting WebRTC connection')
      await connectWebRtc(sceneName, thisSessionId)
    }
  }

  return {
    transportMode,
    status,
    connectionState,
    errorType,
    loading,
    loadingText,
    sessionId,
    currentScene,
    showVideo,
    mediaStream,
    frameSrc,
    renderFrameSrc,
    showRender,
    firstFrameReceived,
    isInQueue,
    onTimeUp,
    exploreMode,
    startingPointSet,
    isRecording,
    recordingProcessing,
    init,
    connect,
    disconnect,
    sendControl,
    sendStartDenoising,
    sendSetStartingPoint,
    startRecording,
    stopRecording,
    isConnected,
    handleVideoReady,
    SESSION_DURATION_SECONDS,
  }
}
