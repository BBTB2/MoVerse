import { ref, reactive, onMounted, onUnmounted } from 'vue'

/**
 * Camera control composable
 * Handles keyboard, mouse drag, and mobile virtual joystick input, converting them to control commands sent to backend
 *
 * @param {Function} sendControl - Function to send control commands
 * @param {Function} isConnected - Function returning whether currently connected
 */
export function useCameraControl(sendControl, isConnected) {
  const urlParams = new URLSearchParams(location.search)
  const moveSpeed = ref(parseFloat(urlParams.get('move_speed') || '0.15'))
  const lookSpeed = ref(parseFloat(urlParams.get('look_speed') || '0.02'))
  const BOOST_FACTOR = parseFloat(urlParams.get('boost') || '1.5')
  const MOUSE_LOOK_GAIN = parseFloat(urlParams.get('mouse_look_gain') || '0.004')
  // Control frame batch size per send (1 = send every frame)
  const BATCH_SIZE = 1

  console.log('[Camera] Control Parameters:')
  console.log('  moveSpeed:', moveSpeed.value)
  console.log('  lookSpeed:', lookSpeed.value)
  console.log('  BOOST_FACTOR:', BOOST_FACTOR)

  // Movement joystick input (from virtual joystick)
  const joy = reactive({ x: 0, y: 0 })
  // Look joystick input (from virtual joystick)
  const lookJoy = reactive({ x: 0, y: 0 })
  // Look visualization (for UI display of joystick direction)
  const lookViz = reactive({ x: 0, y: 0 })
  // Movement knob position (for UI display of joystick position)
  const moveKnob = reactive({ x: 0, y: 0 })
  // Accumulated mouse offset
  const mouseDelta = reactive({ x: 0, y: 0 })

  // Joystick active state
  const joyActive = ref(false)
  const lookJoyActive = ref(false)
  // Mouse drag state
  const isDragging = ref(false)

  // Currently pressed keys set
  const pressedKeys = new Set()

  let lastFrameTime = 0
  let controlLoopTimer = null
  let pendingBatch = []
  let hasPendingMovement = false

  // ─── Keyboard events ────────────────────────────────────────────────────────────────

  function onKeyDown(event) {
    // Skip if user is typing in an input/textarea/contenteditable
    const tag = event.target?.tagName?.toLowerCase()
    if (tag === 'input' || tag === 'textarea' || event.target?.isContentEditable) return

    pressedKeys.add(event.code)

    if (!isConnected()) return

    // Special function keys
    if (event.code === 'KeyZ') sendControl({ render_focal_delta: -10 })
    else if (event.code === 'KeyX') sendControl({ render_focal_delta: 10 })
    else if (event.code === 'KeyQ') sendControl({ local_map_build: true })
    else if (event.code === 'KeyE') sendControl({ local_map_clear: true })
    else if (event.code === 'KeyR') sendControl({ local_map_enable: true })
  }

  function onKeyUp(event) {
    const tag = event.target?.tagName?.toLowerCase()
    if (tag === 'input' || tag === 'textarea' || event.target?.isContentEditable) return

    pressedKeys.delete(event.code)
  }

  // ─── Mouse drag ────────────────────────────────────────────────────────────────

  function handleDragStart() {
    isDragging.value = true
  }

  function handleDragEnd() {
    isDragging.value = false
  }

  function handleDragLook(dx, dy) {
    mouseDelta.x += dx * MOUSE_LOOK_GAIN
    mouseDelta.y -= dy * MOUSE_LOOK_GAIN
  }

  // ─── Compute current movement vector ─────────────────────────────────────────────────

  function computeMoveVector() {
    let moveX = 0
    let moveZ = 0

    if (pressedKeys.has('KeyW')) moveZ += 1
    if (pressedKeys.has('KeyS')) moveZ -= 1
    if (pressedKeys.has('KeyA')) moveX -= 1
    if (pressedKeys.has('KeyD')) moveX += 1

    // Add joystick input
    moveX += joy.x
    moveZ += -joy.y

    // Normalize (prevent diagonal speed boost)
    const magnitude = Math.hypot(moveX, moveZ)
    if (magnitude > 1) {
      moveX /= magnitude
      moveZ /= magnitude
    }

    return [moveX, 0, moveZ]
  }

  // ─── Control loop (42ms interval) ────────────────────────────────────────────

  function controlLoop() {
    const now = performance.now()
    const deltaTime = (now - lastFrameTime) / 1000
    lastFrameTime = now

    const moveVector = computeMoveVector()

    // Update movement knob UI position
    if (!joyActive.value) {
      moveKnob.x = moveVector[0]
      moveKnob.y = -moveVector[2]
    }

    // Compute look delta (keyboard arrows + joystick + mouse)
    let arrowX = 0
    let arrowY = 0
    if (pressedKeys.has('ArrowLeft')) arrowX -= 1
    if (pressedKeys.has('ArrowRight')) arrowX += 1
    if (pressedKeys.has('ArrowUp')) arrowY += 1
    if (pressedKeys.has('ArrowDown')) arrowY -= 1

    const lookDeltaX = mouseDelta.x + lookJoy.x * lookSpeed.value + arrowX * lookSpeed.value
    const lookDeltaY = mouseDelta.y - lookJoy.y * lookSpeed.value + arrowY * lookSpeed.value

    // Consume accumulated mouse offset
    mouseDelta.x = 0
    mouseDelta.y = 0

    // Update look visualization (update when arrow keys active, otherwise decay)
    if (!lookJoyActive.value && !isDragging.value) {
      if (arrowX !== 0 || arrowY !== 0) {
        lookViz.x = arrowX
        lookViz.y = -arrowY
      } else {
        lookViz.x *= 0.82
        lookViz.y *= 0.82
        if (Math.abs(lookViz.x) < 0.01) lookViz.x = 0
        if (Math.abs(lookViz.y) < 0.01) lookViz.y = 0
      }
    }

    if (!isConnected()) return

    const hasMove = moveVector[0] !== 0 || moveVector[2] !== 0
    const hasLook = lookDeltaX !== 0 || lookDeltaY !== 0

    if (!hasMove && !hasLook) {
      if (!hasPendingMovement) return
      hasPendingMovement = false
    } else {
      hasPendingMovement = true
    }

    const isBoosting = pressedKeys.has('ShiftLeft') || pressedKeys.has('ShiftRight')
    const boostMultiplier = isBoosting ? BOOST_FACTOR : 1
    const moveLocal = [
      moveVector[0] * moveSpeed.value * boostMultiplier,
      0,
      moveVector[2] * moveSpeed.value * boostMultiplier,
    ]

    pendingBatch.push({
      dt: deltaTime,
      move_local: moveLocal,
      look_delta: [lookDeltaX, lookDeltaY],
    })

    if (pendingBatch.length >= BATCH_SIZE) {
      sendControl(pendingBatch[0])
      pendingBatch = []
      hasPendingMovement = false
    }
  }

  // ─── Joystick update ────────────────────────────────────────────────────────────────

  function updateJoy(x, y) {
    joy.x = Math.max(-1, Math.min(1, x))
    joy.y = Math.max(-1, Math.min(1, y))
  }

  function updateLookJoy(x, y) {
    lookJoy.x = Math.max(-1, Math.min(1, x))
    lookJoy.y = Math.max(-1, Math.min(1, y))
  }

  // ─── Reset view ────────────────────────────────────────────────────────────────

  function recenter() {
    joy.x = 0; joy.y = 0
    lookJoy.x = 0; lookJoy.y = 0
    mouseDelta.x = 0; mouseDelta.y = 0
    lookViz.x = 0; lookViz.y = 0
    moveKnob.x = 0; moveKnob.y = 0
    pendingBatch = []
    hasPendingMovement = false
    if (isConnected()) sendControl({ recenter: true })
  }

  // ─── Lifecycle ────────────────────────────────────────────────────────────────

  onMounted(() => {
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    lastFrameTime = performance.now()
    controlLoopTimer = setInterval(controlLoop, 42)
  })

  onUnmounted(() => {
    window.removeEventListener('keydown', onKeyDown)
    window.removeEventListener('keyup', onKeyUp)
    pressedKeys.clear()
    if (controlLoopTimer) { clearInterval(controlLoopTimer); controlLoopTimer = null }
  })

  return {
    joy,
    lookJoy,
    lookViz,
    moveKnob,
    joyActive,
    lookJoyActive,
    isDragging,
    moveSpeed,
    lookSpeed,
    handleDragStart,
    handleDragEnd,
    handleDragLook,
    updateJoy,
    updateLookJoy,
    recenter,
  }
}
