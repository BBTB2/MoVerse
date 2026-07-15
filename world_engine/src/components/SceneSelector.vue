<template>
  <div class="w-full overflow-x-auto scrollbar-hide">
    <div class="flex gap-2 pb-1" style="min-width: max-content">
      <button
        v-for="scene in scenes"
        :key="scene.name"
        class="flex items-center gap-2 px-3 py-1.5 rounded-full border text-[10px] font-mono tracking-wider uppercase transition-all duration-200 shrink-0 active:scale-95"
        :class="[
          scene.name === currentScene
            ? isDarkMode
              ? 'bg-zinc-100 text-zinc-900 border-zinc-100'
              : 'bg-zinc-900 text-zinc-50 border-zinc-900'
            : isDarkMode
              ? 'bg-transparent text-zinc-400 border-zinc-700/60 hover:text-zinc-200 hover:border-zinc-500'
              : 'bg-transparent text-zinc-500 border-zinc-200 hover:text-zinc-800 hover:border-zinc-300',
        ]"
        @click="emit('select', scene.name)"
        @touchend.stop.prevent="emit('select', scene.name)"
      >
        <!-- Thumbnail -->
        <img
          v-if="scene.thumbnail"
          :src="scene.thumbnail"
          class="w-4 h-4 rounded-full object-cover shrink-0"
          draggable="false"
        />
        <!-- Dot shown when no thumbnail -->
        <div
          v-else
          class="w-2 h-2 rounded-full shrink-0"
          :class="scene.name === currentScene
            ? isDarkMode ? 'bg-zinc-700' : 'bg-zinc-300'
            : 'bg-current opacity-40'"
        />
        {{ scene.name }}
      </button>

      <!-- Placeholder when no scenes -->
      <div
        v-if="scenes.length === 0"
        class="px-3 py-1.5 text-[10px] font-mono"
        :class="isDarkMode ? 'text-zinc-600' : 'text-zinc-400'"
      >
        No scenes available
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  /** Scene list: [{ name: string, thumbnail?: string }] */
  scenes: { type: Array, default: () => [] },
  /** Currently selected scene name */
  currentScene: { type: String, default: null },
  /** Dark mode flag */
  isDarkMode: { type: Boolean, default: true },
})

const emit = defineEmits([
  /** User selects a scene: scene.name */
  'select',
])
</script>

<style scoped>
.scrollbar-hide {
  -ms-overflow-style: none;
  scrollbar-width: none;
}
.scrollbar-hide::-webkit-scrollbar {
  display: none;
}
</style>
