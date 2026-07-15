<template>
  <div class="dpad-container">
    <!-- D-Pad cross shape -->
    <div class="dpad-body">
      <!-- Up -->
      <button
        class="dpad-btn dpad-up"
        :class="{ active: isUpActive }"
        @mousedown.prevent="startDir('up')"
        @mouseup.prevent="stopDir('up')"
        @mouseleave.prevent="stopDir('up')"
        @touchstart.prevent="startDir('up')"
        @touchend.prevent="stopDir('up')"
        @touchcancel.prevent="stopDir('up')"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 5l7 10H5z"/></svg>
      </button>
      <!-- Down -->
      <button
        class="dpad-btn dpad-down"
        :class="{ active: isDownActive }"
        @mousedown.prevent="startDir('down')"
        @mouseup.prevent="stopDir('down')"
        @mouseleave.prevent="stopDir('down')"
        @touchstart.prevent="startDir('down')"
        @touchend.prevent="stopDir('down')"
        @touchcancel.prevent="stopDir('down')"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 19l7-10H5z"/></svg>
      </button>
      <!-- Left -->
      <button
        class="dpad-btn dpad-left"
        :class="{ active: isLeftActive }"
        @mousedown.prevent="startDir('left')"
        @mouseup.prevent="stopDir('left')"
        @mouseleave.prevent="stopDir('left')"
        @touchstart.prevent="startDir('left')"
        @touchend.prevent="stopDir('left')"
        @touchcancel.prevent="stopDir('left')"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M5 12l10 7V5z"/></svg>
      </button>
      <!-- Right -->
      <button
        class="dpad-btn dpad-right"
        :class="{ active: isRightActive }"
        @mousedown.prevent="startDir('right')"
        @mouseup.prevent="stopDir('right')"
        @mouseleave.prevent="stopDir('right')"
        @touchstart.prevent="startDir('right')"
        @touchend.prevent="stopDir('right')"
        @touchcancel.prevent="stopDir('right')"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M19 12L9 5v14z"/></svg>
      </button>
      <!-- Center circle -->
      <div class="dpad-center"></div>
    </div>
  </div>
</template>

<script setup>
import { computed, reactive } from 'vue'

const props = defineProps({
  /** Set of currently pressed keyboard keys (lowercase) */
  pressedKeys: { type: Object, default: () => new Set() },
})

const emit = defineEmits([
  /** Emits { x, y } normalized -1..1 when direction changes */
  'update:value',
  'activeChange',
])

// Track which directions are active via touch/mouse
const touchDirs = reactive({ up: false, down: false, left: false, right: false })

// Combine touch + keyboard state (WASD only - arrows are for look/orientation)
const isUpActive = computed(() => touchDirs.up || props.pressedKeys.has('w'))
const isDownActive = computed(() => touchDirs.down || props.pressedKeys.has('s'))
const isLeftActive = computed(() => touchDirs.left || props.pressedKeys.has('a'))
const isRightActive = computed(() => touchDirs.right || props.pressedKeys.has('d'))

function emitValue() {
  let x = 0, y = 0
  if (isUpActive.value) y -= 1
  if (isDownActive.value) y += 1
  if (isLeftActive.value) x -= 1
  if (isRightActive.value) x += 1
  emit('update:value', { x, y })
  emit('activeChange', x !== 0 || y !== 0)
}

function startDir(dir) {
  touchDirs[dir] = true
  emitValue()
}

function stopDir(dir) {
  touchDirs[dir] = false
  emitValue()
}
</script>

<style scoped>
.dpad-container {
  width: 100px;
  height: 100px;
  position: relative;
}

.dpad-body {
  position: absolute;
  inset: 0;
}

.dpad-btn {
  position: absolute;
  width: 34px;
  height: 34px;
  background: linear-gradient(180deg, #5BA0F0, #3A7BD5);
  border: 2px solid #2F6BC0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  cursor: pointer;
  transition: background 0.1s, transform 0.1s, box-shadow 0.1s;
  -webkit-tap-highlight-color: transparent;
  box-shadow: 0 3px 6px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.3);
}

.dpad-btn:hover {
  background: linear-gradient(180deg, #4A90E2, #2F6BC0);
}

.dpad-btn.active {
  background: linear-gradient(180deg, #2563EB, #1D4ED8);
  transform: scale(0.92);
  box-shadow: inset 0 2px 4px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.1);
}

.dpad-up {
  top: 0;
  left: 50%;
  transform: translateX(-50%);
  border-radius: 6px 6px 0 0;
}
.dpad-up.active { transform: translateX(-50%) scale(0.92); }

.dpad-down {
  bottom: 0;
  left: 50%;
  transform: translateX(-50%);
  border-radius: 0 0 6px 6px;
}
.dpad-down.active { transform: translateX(-50%) scale(0.92); }

.dpad-left {
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  border-radius: 6px 0 0 6px;
}
.dpad-left.active { transform: translateY(-50%) scale(0.92); }

.dpad-right {
  right: 0;
  top: 50%;
  transform: translateY(-50%);
  border-radius: 0 6px 6px 0;
}
.dpad-right.active { transform: translateY(-50%) scale(0.92); }

.dpad-center {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 22px;
  height: 22px;
  background: linear-gradient(135deg, #3A7BD5, #2F6BC0);
  border-radius: 5px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.2);
}
</style>
