import { ref } from 'vue'
import { getApiBaseUrl } from './useTransport.js'

/**
 * Scene list composable
 * Fetches available scene list from backend /api/scenes
 */
export function useScenes() {
  const scenes = ref([])

  async function fetchScenes() {
    try {
      const apiBase = getApiBaseUrl()
      const response = await fetch(`${apiBase}/api/scenes`)
      const data = await response.json()
      scenes.value = (data.scenes || []).map(scene => ({
        ...scene,
        // Convert relative thumbnail path to absolute path
        thumbnail: scene.thumbnail ? `${apiBase}${scene.thumbnail}` : scene.thumbnail,
      }))
    } catch (error) {
      console.error('[useScenes] fetch failed:', error)
      scenes.value = []
    }
  }

  return {
    scenes,
    fetchScenes,
  }
}
