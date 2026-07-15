<template>
  <div
    class="relative select-none"
    :style="{ width: SIZE + 'px', height: SIZE + 'px' }"
    @touchstart.prevent="onTouchStart"
    @touchmove.prevent="onTouchMove"
    @touchend.prevent="onTouchEnd"
    @touchcancel.prevent="onTouchEnd"
    @mousedown.prevent="onMouseDown"
  >
    <!-- Base ring -->
    <div
      class="absolute inset-0 rounded-full border-2 transition-colors duration-200"
      :style="{
        background: isDarkMode ? 'rgba(74, 143, 231, 0.15)' : 'rgba(74, 143, 231, 0.1)',
        borderColor: isDarkMode ? 'rgba(74, 143, 231, 0.4)' : 'rgba(58, 123, 213, 0.3)',
      }"
    />

    <!-- Joystick knob -->
    <div
      class="absolute rounded-full shadow-md transition-colors duration-200 pointer-events-none"
      :style="{
        width: KNOB_SIZE + 'px',
        height: KNOB_SIZE + 'px',
        left: knobLeft + 'px',
        top: knobTop + 'px',
        background: isActive ? '#2563EB' : '#4A8FE7',
        border: '2px solid #3A7BD5',
        boxShadow: isActive ? '0 4px 12px rgba(37, 99, 235, 0.4)' : '0 2px 8px rgba(74, 143, 231, 0.3)',
      }"
    />
  </div>
</template>

<script setup>
import { ref, computed, onUnmounted } from 'vue'

const props = defineProps({
  /** Base diameter (px) */
  size: { type: Number, default: 96 },
  /** Knob diameter (px) */
  knobSize: { type: Number, default: 40 },
  /** External X value (-1 to 1), for keyboard mode sync display */
  externalX: { type: Number, default: 0 },
  /** External Y value (-1 to 1), for keyboard mode sync display */
  externalY: { type: Number, default: 0 },
  /** Dark mode flag */
  isDarkMode: { type: Boolean, default: true },
})

const emit = defineEmits([
  /** Joystick value change: { x, y }, range -1 to 1 */
  'update:value',
  /** Active state change: true/false */
  'activeChange',
])

const SIZE = computed(() => props.size)
const KNOB_SIZE = computed(() => props.knobSize)
const MAX_RADIUS = computed(() => (SIZE.value - KNOB_SIZE.value) / 2)
const CENTER = computed(() => SIZE.value / 2)

// Current knob offset (pixels)
const offsetX = ref(0)
const offsetY = ref(0)
// Whether being touched/dragged
const isActive = ref(false)
// Touch identifier
let activeTouchId = null
// Global event listeners during mouse drag
let mouseMoveHandler = null
let mouseUpHandler = null

// Knob top-left position (px)
const knobLeft = computed(() => {
  const activeX = isActive.value ? offsetX.value : props.externalX * MAX_RADIUS.value
  return CENTER.value - KNOB_SIZE.value / 2 + activeX
})
const knobTop = computed(() => {
  const activeY = isActive.value ? offsetY.value : props.externalY * MAX_RADIUS.value
  return CENTER.value - KNOB_SIZE.value / 2 + activeY
})

function clampToRadius(dx, dy) {
  const distance = Math.hypot(dx, dy)
  const maxR = MAX_RADIUS.value
  if (distance > maxR) {
    const scale = maxR / distance
    return { x: dx * scale, y: dy * scale }
  }
  return { x: dx, y: dy }
}

function applyOffset(dx, dy) {
  const clamped = clampToRadius(dx, dy)
  offsetX.value = clamped.x
  offsetY.value = clamped.y
  const maxR = MAX_RADIUS.value
  emit('update:value', {
    x: maxR > 0 ? clamped.x / maxR : 0,
    y: maxR > 0 ? clamped.y / maxR : 0,
  })
}

function getElementCenter(element) {
  const rect = element.getBoundingClientRect()
  return {
    cx: rect.left + rect.width / 2,
    cy: rect.top + rect.height / 2,
  }
}

// ─── Touch events ────────────────────────────────────────────────────────────────

function onTouchStart(event) {
  if (activeTouchId !== null) return
  const touch = event.changedTouches[0]
  activeTouchId = touch.identifier
  isActive.value = true
  emit('activeChange', true)
  const { cx, cy } = getElementCenter(event.currentTarget)
  applyOffset(touch.clientX - cx, touch.clientY - cy)
}

function onTouchMove(event) {
  for (const touch of event.changedTouches) {
    if (touch.identifier !== activeTouchId) continue
    const { cx, cy } = getElementCenter(event.currentTarget)
    applyOffset(touch.clientX - cx, touch.clientY - cy)
  }
}

function onTouchEnd(event) {
  for (const touch of event.changedTouches) {
    if (touch.identifier !== activeTouchId) continue
    activeTouchId = null
    isActive.value = false
    offsetX.value = 0
    offsetY.value = 0
    emit('update:value', { x: 0, y: 0 })
    emit('activeChange', false)
  }
}

// ─── Mouse events (desktop debugging) ────────────────────────────────────────────

function onMouseDown(event) {
  isActive.value = true
  emit('activeChange', true)
  const { cx, cy } = getElementCenter(event.currentTarget)
  applyOffset(event.clientX - cx, event.clientY - cy)

  mouseMoveHandler = (moveEvent) => {
    applyOffset(moveEvent.clientX - cx, moveEvent.clientY - cy)
  }
  mouseUpHandler = () => {
    isActive.value = false
    offsetX.value = 0
    offsetY.value = 0
    emit('update:value', { x: 0, y: 0 })
    emit('activeChange', false)
    window.removeEventListener('mousemove', mouseMoveHandler)
    window.removeEventListener('mouseup', mouseUpHandler)
    mouseMoveHandler = null
    mouseUpHandler = null
  }
  window.addEventListener('mousemove', mouseMoveHandler)
  window.addEventListener('mouseup', mouseUpHandler)
}

onUnmounted(() => {
  if (mouseMoveHandler) window.removeEventListener('mousemove', mouseMoveHandler)
  if (mouseUpHandler) window.removeEventListener('mouseup', mouseUpHandler)
})
</script>
