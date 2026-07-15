<template>
  <div class="relative flex items-center justify-between shrink-0 z-10 px-4 py-1 min-h-[36px]">
    <!-- Left: connection status indicator -->
    <div
      class="flex items-center gap-2 text-[10px] font-mono tracking-wider uppercase"
      :style="{ color: statusConfig.color }"
    >
      <div
        class="w-1.5 h-1.5 rounded-full animate-pulse"
        :style="{ backgroundColor: statusConfig.dotColor }"
      />
      {{ statusConfig.text }}
    </div>

    <!-- Center: brand name -->
    <div class="absolute left-1/2 -translate-x-1/2 flex items-center gap-1.5 text-zinc-400 text-[10px] font-mono italic tracking-tighter whitespace-nowrap">
      SPATIAL BOY
    </div>

    <!-- Right: countdown timer -->
    <div
      class="text-[10px] font-mono tabular-nums"
      :class="[
        isGameOver ? 'gameover-blink' : '',
        isDarkMode ? 'text-zinc-400' : 'text-zinc-500',
      ]"
    >
      {{ timerDisplay }}
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'

const props = defineProps({
  /** Connection state: 'disconnected' | 'connecting' | 'connected' | 'error' */
  connectionState: { type: String, default: 'disconnected' },
  /** Loading text (shown while connecting) */
  loadingText: { type: String, default: '' },
  /** Whether game over (time up) */
  isGameOver: { type: Boolean, default: false },
  /** Whether first frame received (starts countdown) */
  firstFrameReceived: { type: Boolean, default: false },
  /** Dark mode flag */
  isDarkMode: { type: Boolean, default: true },
  /** Session duration (seconds) */
  sessionDuration: { type: Number, default: 300 },
})

const emit = defineEmits([
  /** Triggered when countdown reaches zero */
  'timeout',
])

// ─── Countdown logic ──────────────────────────────────────────────────────────────

const remainingSeconds = ref(props.sessionDuration)
let countdownTimer = null
let sessionStartTime = null
let isCountingDown = false

function formatTime(totalSeconds) {
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}

const timerDisplay = computed(() => formatTime(remainingSeconds.value))

function startCountdown() {
  if (isCountingDown) return
  isCountingDown = true
  sessionStartTime = Date.now()

  countdownTimer = setInterval(() => {
    const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000)
    const remaining = Math.max(0, props.sessionDuration - elapsed)
    remainingSeconds.value = remaining

    if (remaining <= 0) {
      clearInterval(countdownTimer)
      countdownTimer = null
      emit('timeout')
    }
  }, 100)
}

function stopCountdown() {
  if (countdownTimer) { clearInterval(countdownTimer); countdownTimer = null }
  isCountingDown = false
  sessionStartTime = null
}

function resetCountdown() {
  stopCountdown()
  remainingSeconds.value = props.sessionDuration
}

// Start countdown when first frame received
watch(() => props.firstFrameReceived, (received) => {
  if (received && !isCountingDown) startCountdown()
})

onMounted(() => {
  resetCountdown()
  if (props.firstFrameReceived) startCountdown()
})

onUnmounted(() => {
  stopCountdown()
})

// ─── Status indicator config ──────────────────────────────────────────────────────────

const STATUS_CONFIG = {
  connected: { text: 'Power', color: '#34C759', dotColor: '#34C759' },
  connecting: { text: 'Loading', color: '#FCD34D', dotColor: '#FCD34D' },
  disconnected: { text: 'Loading', color: '#FCD34D', dotColor: '#FCD34D' },
  error: { text: 'Power', color: '#EF4444', dotColor: '#EF4444' },
  gameover: { text: 'Power', color: '#EF4444', dotColor: '#EF4444' },
}

const statusConfig = computed(() => {
  if (props.isGameOver) return STATUS_CONFIG.gameover
  return STATUS_CONFIG[props.connectionState] || STATUS_CONFIG.connecting
})
</script>
