<template>
  <div
    class="relative w-full h-full flex items-center justify-center overflow-hidden"
    @mousedown="onMouseDown"
    @mousemove="onMouseMove"
    @mouseup="onMouseUp"
    @mouseleave="onMouseUp"
    @touchstart.prevent="onTouchStart"
    @touchmove.prevent="onTouchMove"
    @touchend.prevent="onTouchEnd"
    @touchcancel.prevent="onTouchEnd"
  >
    <!-- Dual view mode (render frame | generated frame) -->
    <div
      v-if="showRender && showVideo"
      class="w-full h-full flex gap-1"
    >
      <!-- Left: render frame -->
      <div class="relative flex-1 flex items-center justify-center overflow-hidden">
        <img
          v-if="renderFrameSrc"
          :src="renderFrameSrc"
          class="w-full h-full object-contain"
          draggable="false"
        />
        <div
          v-else
          class="flex items-center justify-center w-full h-full"
          :class="isDarkMode ? 'bg-zinc-700/30' : 'bg-zinc-100/50'"
        >
          <span class="text-xs font-mono opacity-40">Render</span>
        </div>
        <div class="absolute top-1 left-1 px-1.5 py-0.5 rounded text-[9px] font-mono uppercase tracking-wider opacity-60"
             :class="isDarkMode ? 'bg-zinc-800/70 text-zinc-400' : 'bg-white/70 text-zinc-500'">
          Render
        </div>
      </div>
      <!-- Right: generated frame -->
      <div class="relative flex-1 flex items-center justify-center overflow-hidden">
        <img
          v-if="frameSrc"
          :src="frameSrc"
          class="w-full h-full object-contain"
          draggable="false"
          @load="onFrameLoad"
        />
        <div
          v-else
          class="flex items-center justify-center w-full h-full"
          :class="isDarkMode ? 'bg-zinc-700/30' : 'bg-zinc-100/50'"
        >
          <span class="text-xs font-mono opacity-40">Generated</span>
        </div>
        <div class="absolute top-1 left-1 px-1.5 py-0.5 rounded text-[9px] font-mono uppercase tracking-wider opacity-60"
             :class="isDarkMode ? 'bg-zinc-800/70 text-zinc-400' : 'bg-white/70 text-zinc-500'">
          Generated
        </div>
      </div>
    </div>

    <!-- Single view mode (generated frame only) -->
    <template v-else>
      <!-- WebRTC video stream (UDP mode) -->
      <video
        v-if="showVideo && mediaStream"
        ref="videoElement"
        class="w-full h-full object-contain"
        autoplay
        playsinline
        muted
        @canplay="onVideoCanPlay"
      />

      <!-- TCP frame image stream -->
      <img
        v-else-if="showVideo && frameSrc"
        :src="frameSrc"
        class="w-full h-full object-contain"
        draggable="false"
        @load="onFrameLoad"
      />
    </template>

    <!-- Keyboard mode key hint HUD -->
    <div
      v-if="isKeyboardMode && showVideo"
      class="absolute bottom-4 left-0 right-0 flex items-end justify-between px-6 sm:px-10 pointer-events-none"
    >
      <!-- Leftmost: WASD movement controls -->
      <div class="flex flex-col items-center gap-1">
        <div class="text-[9px] font-mono uppercase tracking-wider opacity-40 mb-0.5"
             :class="isDarkMode ? 'text-zinc-400' : 'text-zinc-500'">Move</div>
        <div class="flex gap-1">
          <div class="kbd-key" :class="{ pressed: pressedKeys.has('w') }">W</div>
        </div>
        <div class="flex gap-1">
          <div class="kbd-key" :class="{ pressed: pressedKeys.has('a') }">A</div>
          <div class="kbd-key" :class="{ pressed: pressedKeys.has('s') }">S</div>
          <div class="kbd-key" :class="{ pressed: pressedKeys.has('d') }">D</div>
        </div>
      </div>

      <!-- Rightmost: Arrow key rotation controls -->
      <div class="flex flex-col items-center gap-1">
        <div class="text-[9px] font-mono uppercase tracking-wider opacity-40 mb-0.5"
             :class="isDarkMode ? 'text-zinc-400' : 'text-zinc-500'">Look</div>
        <div class="flex gap-1">
          <div class="kbd-arrow" :class="{ pressed: pressedKeys.has('arrowup') }">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 4l8 16H4z"/>
            </svg>
          </div>
        </div>
        <div class="flex gap-1">
          <div class="kbd-arrow" :class="{ pressed: pressedKeys.has('arrowleft') }">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
              <path d="M20 12H4M4 12l8-8M4 12l8 8" stroke="currentColor" stroke-width="2" fill="none"/>
            </svg>
          </div>
          <div class="kbd-arrow" :class="{ pressed: pressedKeys.has('arrowdown') }">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 20l8-16H4z"/>
            </svg>
          </div>
          <div class="kbd-arrow" :class="{ pressed: pressedKeys.has('arrowright') }">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
              <path d="M4 12h16M16 12l-8-8M16 12l-8 8" stroke="currentColor" stroke-width="2" fill="none"/>
            </svg>
          </div>
        </div>
      </div>
    </div>

    <!-- Error / loading overlay -->
    <div
      v-if="loading"
      class="absolute inset-0 flex flex-col items-center justify-center gap-4 z-20"
      :class="isDarkMode ? 'bg-zinc-800/90' : 'bg-white/90'"
    >
      <!-- Queue status -->
      <div v-if="isInQueue" class="flex flex-col items-center gap-3">
        <div class="w-8 h-8 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
        <p class="text-sm font-mono text-center px-6" :class="isDarkMode ? 'text-zinc-300' : 'text-zinc-600'">
          {{ loadingText }}
        </p>
      </div>

      <!-- Error state -->
      <div v-else-if="connectionState === 'error'" class="flex flex-col items-center gap-3 px-6">
        <div class="w-10 h-10 rounded-full flex items-center justify-center bg-red-500/20">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
        </div>
        <p class="text-sm font-mono text-center" :class="isDarkMode ? 'text-zinc-300' : 'text-zinc-600'">
          {{ loadingText }}
        </p>
        <p v-if="errorType === 'high_demand'" class="text-xs font-mono text-center text-zinc-500">
          High demand — please try again shortly
        </p>
      </div>

      <!-- Normal loading state -->
      <div v-else class="flex flex-col items-center gap-3">
        <div class="w-8 h-8 border-2 border-zinc-400 border-t-transparent rounded-full animate-spin" />
        <p class="text-sm font-mono text-center px-6" :class="isDarkMode ? 'text-zinc-300' : 'text-zinc-600'">
          {{ loadingText }}
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'

const props = defineProps({
  /** Whether to show video/frame display */
  showVideo: { type: Boolean, default: false },
  /** WebRTC media stream (UDP mode) */
  mediaStream: { type: Object, default: null },
  /** Current frame image URL (TCP mode) */
  frameSrc: { type: String, default: '' },
  /** Render frame image URL (TCP mode, when show_render enabled) */
  renderFrameSrc: { type: String, default: '' },
  /** Whether to show render frame side-by-side */
  showRender: { type: Boolean, default: false },
  /** Whether to show loading overlay */
  loading: { type: Boolean, default: true },
  /** Loading text */
  loadingText: { type: String, default: 'Loading…' },
  /** Whether keyboard mode (show key HUD) */
  isKeyboardMode: { type: Boolean, default: false },
  /** Currently pressed keys set (for HUD highlight) */
  pressedKeys: { type: Object, default: () => new Set() },
  /** Connection state */
  connectionState: { type: String, default: 'disconnected' },
  /** Error type */
  errorType: { type: String, default: '' },
  /** Whether game over */
  isGameOver: { type: Boolean, default: false },
  /** Whether in queue */
  isInQueue: { type: Boolean, default: false },
  /** Dark mode flag */
  isDarkMode: { type: Boolean, default: true },
})

const emit = defineEmits([
  /** Video element ready to play */
  'videoReady',
  /** Drag start */
  'dragStart',
  /** Drag end */
  'dragEnd',
  /** Drag look offset: { dx, dy } */
  'dragLook',
])

const videoElement = ref(null)

// Bind media stream to video element
watch(() => props.mediaStream, (stream) => {
  if (videoElement.value && stream) {
    videoElement.value.srcObject = stream
  }
}, { immediate: true })

watch(videoElement, (el) => {
  if (el && props.mediaStream) {
    el.srcObject = props.mediaStream
  }
})

function onVideoCanPlay() {
  emit('videoReady')
}

function onFrameLoad() {
  // TCP mode first frame loaded, notify parent
  emit('videoReady')
}

// ─── Mouse drag (view control) ────────────────────────────────────────────

let isDraggingMouse = false
let lastMouseX = 0
let lastMouseY = 0

function onMouseDown(event) {
  isDraggingMouse = true
  lastMouseX = event.clientX
  lastMouseY = event.clientY
  emit('dragStart')
}

function onMouseMove(event) {
  if (!isDraggingMouse) return
  const dx = event.clientX - lastMouseX
  const dy = event.clientY - lastMouseY
  lastMouseX = event.clientX
  lastMouseY = event.clientY
  emit('dragLook', { dx, dy })
}

function onMouseUp() {
  if (!isDraggingMouse) return
  isDraggingMouse = false
  emit('dragEnd')
}

// ─── Touch drag (view control, single finger) ──────────────────────────────────────────

let activeDragTouchId = null
let lastTouchX = 0
let lastTouchY = 0

function onTouchStart(event) {
  // Only handle the first touch point (multi-touch handled by joystick component)
  if (activeDragTouchId !== null) return
  const touch = event.changedTouches[0]
  activeDragTouchId = touch.identifier
  lastTouchX = touch.clientX
  lastTouchY = touch.clientY
  emit('dragStart')
}

function onTouchMove(event) {
  for (const touch of event.changedTouches) {
    if (touch.identifier !== activeDragTouchId) continue
    const dx = touch.clientX - lastTouchX
    const dy = touch.clientY - lastTouchY
    lastTouchX = touch.clientX
    lastTouchY = touch.clientY
    emit('dragLook', { dx, dy })
  }
}

function onTouchEnd(event) {
  for (const touch of event.changedTouches) {
    if (touch.identifier !== activeDragTouchId) continue
    activeDragTouchId = null
    emit('dragEnd')
  }
}
</script>
