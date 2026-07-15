import { createApp } from 'vue'
import './style.css'
import App from './App.vue'

// Optional: debug mode loads Eruda mobile debug tool (via CDN, no npm install needed)
const urlParams = new URLSearchParams(location.search)
if (urlParams.get('debug') === '1') {
  const erudaScript = document.createElement('script')
  erudaScript.src = 'https://cdn.jsdelivr.net/npm/eruda'
  erudaScript.onload = () => {
    window.eruda.init()
    console.log('[Debug] Eruda initialized - mobile debugging enabled')
    console.log('[Debug] Current URL:', location.href)
    console.log('[Debug] User Agent:', navigator.userAgent)
  }
  erudaScript.onerror = (err) => {
    console.error('[Debug] Failed to load Eruda:', err)
  }
  document.head.appendChild(erudaScript)
}

document.addEventListener('contextmenu', e => e.preventDefault())

createApp(App).mount('#app')
