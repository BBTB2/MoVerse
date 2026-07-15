<template>
  <div
    class="fixed inset-0 flex flex-col items-center justify-center overflow-hidden select-none"
    style="touch-action: none; background: var(--console-bg); font-family: 'Inter', sans-serif;"
    @touchstart.prevent
  >
    <!-- Page-level logo (fixed top-left corner) -->
    <div class="page-logo">
      <img :src="originalLogoUrl" alt="Logo" />
    </div>

    <!-- ── Console Shell ─────────────────────────────────────────────── -->
    <div class="console-shell" :class="{ 'gen-only-mode': !showRender }">
      <!-- Left Wing (D-Pad for movement) -->
      <div class="console-wing console-wing-left">
        <!-- Logo at top-left corner -->
        <div class="wing-top">
          <img :src="logoUrl" alt="Logo" class="console-logo" />
        </div>

        <!-- D-Pad -->
        <div class="wing-center">
          <DPad
            :pressed-keys="pressedKeys"
            @update:value="({ x, y }) => updateJoy(x, y)"
            @active-change="(active) => joyActive = active"
          />
        </div>

        <!-- Orange minus button -->
        <div class="wing-bottom">
          <button class="pill-btn pill-btn-orange" @click="recenter" @touchend.stop.prevent="recenter">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14"/></svg>
          </button>
        </div>
      </div>

      <!-- Center Screen -->
      <div class="console-screen-wrap">
        <!-- Status bar inside screen -->
        <div class="screen-status-bar">
          <div class="status-indicator" :style="{ color: statusConfig.color }">
            <div class="status-dot" :style="{ backgroundColor: statusConfig.dotColor }"></div>
            <span>{{ statusConfig.text }}</span>
          </div>
          <div class="status-brand">MoVerse</div>
          <div v-if="isRecording" class="rec-indicator">
            <div class="rec-dot"></div>
            <span>REC</span>
          </div>
          <div class="status-timer" :class="{ 'gameover-blink': isGameOver }">{{ timerDisplay }}</div>
        </div>

        <!-- Video display area -->
        <div class="screen-display"
          @mousedown="onMouseDown"
          @mousemove="onMouseMove"
          @mouseup="onMouseUp"
          @mouseleave="onMouseUp"
          @touchstart.prevent="onTouchStart"
          @touchmove.prevent="onTouchMove"
          @touchend.prevent="onTouchEnd"
          @touchcancel.prevent="onTouchEnd"
        >
          <!-- Side by side mode -->
          <div v-show="showRender && showVideo" class="screen-split">
            <div class="screen-panel">
              <img v-if="renderFrameSrc" :src="renderFrameSrc" class="screen-img" draggable="false" />
              <div v-else class="screen-placeholder">Render</div>
              <span class="panel-label">Render</span>
            </div>
            <div class="screen-panel">
              <img v-if="frameSrc" :src="frameSrc" class="screen-img" draggable="false" />
              <div v-else class="screen-placeholder">Generated</div>
              <span class="panel-label">{{ exploreMode ? 'Render Preview' : 'Generated' }}</span>
              <!-- Start Denoising button overlay (explore mode) -->
              <div v-if="exploreMode" class="explore-overlay">
                <button class="start-denoising-btn" @click="onStartDenoising">
                  Click to start real time denoising
                </button>
              </div>
            </div>
          </div>

          <!-- Single view mode (Gen Only) -->
          <div v-show="!showRender && showVideo" class="screen-single">
            <video
              v-if="mediaStream"
              ref="videoElement"
              class="screen-img"
              autoplay playsinline muted
              @canplay="handleVideoReady"
            />
            <img
              v-else-if="frameSrc"
              :src="frameSrc"
              class="screen-img"
              draggable="false"
            />
            <!-- Start Denoising button overlay (explore mode) -->
            <div v-if="exploreMode" class="explore-overlay">
              <button class="start-denoising-btn" @click="onStartDenoising">
                Click to start real time denoising
              </button>
            </div>
          </div>

          <!-- Loading overlay -->
          <div v-if="loading" class="screen-overlay">
            <div v-if="isInQueue" class="overlay-content">
              <div class="spinner"></div>
              <p>{{ loadingText }}</p>
            </div>
            <div v-else-if="connectionState === 'error'" class="overlay-content overlay-error">
              <div class="error-icon">!</div>
              <p>{{ loadingText }}</p>
            </div>
            <div v-else class="overlay-content">
              <div class="spinner"></div>
              <p>{{ loadingText }}</p>
            </div>
          </div>
        </div>

        <!-- Help overlay -->
        <div v-if="showHelp" class="help-overlay" @click="showHelp = false">
          <div class="help-card" @click.stop>
            <h3>Controls</h3>
            <div class="help-grid">
              <div class="help-item"><span class="help-key">W A S D</span><span>Move forward/left/back/right</span></div>
              <div class="help-item"><span class="help-key">← ↑ → ↓</span><span>Look / rotate camera</span></div>
              <div class="help-item"><span class="help-key">Drag screen</span><span>Look around (mouse/touch)</span></div>
              <div class="help-item"><span class="help-key">Left D-Pad</span><span>Move (touch)</span></div>
              <div class="help-item"><span class="help-key">Right Joystick</span><span>Look / orientation (touch)</span></div>
              <div class="help-item"><span class="help-key">Shift</span><span>Boost movement speed</span></div>
              <div class="help-item"><span class="help-key">Z / X</span><span>Zoom out / in (focal)</span></div>
              <div class="help-item"><span class="help-key">− / + buttons</span><span>Reset / Reconnect</span></div>
            </div>
            <button class="help-close" @click="showHelp = false">Got it</button>
          </div>
        </div>
      </div>

      <!-- Right Wing (Joystick for orientation) -->
      <div class="console-wing console-wing-right">
        <!-- Face buttons placeholder (X/Y/A/B style) -->
        <div class="wing-top">
          <div class="face-buttons">
            <span class="face-btn face-x">X</span>
            <span class="face-btn face-y">Y</span>
            <span class="face-btn face-a">A</span>
            <span class="face-btn face-b">B</span>
          </div>
        </div>

        <!-- Right Joystick (Look/Orientation) -->
        <div class="wing-center">
          <Joystick
            :external-x="lookViz.x"
            :external-y="lookViz.y"
            :is-dark-mode="false"
            @update:value="({ x, y }) => updateLookJoy(x, y)"
            @active-change="(active) => lookJoyActive = active"
          />
        </div>

        <!-- Orange plus button -->
        <div class="wing-bottom">
          <button class="pill-btn pill-btn-orange" @click="onReconnect" @touchend.stop.prevent="onReconnect">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 5v14M5 12h14"/></svg>
          </button>
        </div>
      </div>
    </div>

    <!-- Control Panel (slim horizontal bar, below console) -->
    <div class="control-panel">
      <button
        class="ctrl-btn"
        :class="{ active: showRender }"
        @click="toggleRenderView"
        @touchend.stop.prevent="toggleRenderView"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <rect x="2" y="3" width="8" height="18" rx="1"/><rect x="14" y="3" width="8" height="18" rx="1"/>
        </svg>
        {{ showRender ? 'Split' : 'Gen Only' }}
      </button>

      <div class="ctrl-divider"></div>

      <div class="ctrl-slider-group">
        <span class="ctrl-label">Gauss</span>
        <input type="range" min="0.1" max="5.0" step="0.1" v-model.number="gaussianScale" class="ctrl-slider" @change="onGaussianScaleChange" @pointerup="$event.target.blur()" @keyup.enter="$event.target.blur()" @keyup.escape="$event.target.blur()" />
        <span class="ctrl-value">{{ gaussianScale.toFixed(1) }}</span>
      </div>

      <div class="ctrl-slider-group">
        <span class="ctrl-label">Move</span>
        <input type="range" min="0.01" max="0.5" step="0.01" v-model.number="moveSpeed" class="ctrl-slider" @pointerup="$event.target.blur()" @keyup.enter="$event.target.blur()" @keyup.escape="$event.target.blur()" />
        <span class="ctrl-value">{{ moveSpeed.toFixed(2) }}</span>
      </div>

      <div class="ctrl-slider-group">
        <span class="ctrl-label">Look</span>
        <input type="range" min="0.005" max="0.1" step="0.005" v-model.number="lookSpeed" class="ctrl-slider" @pointerup="$event.target.blur()" @keyup.enter="$event.target.blur()" @keyup.escape="$event.target.blur()" />
        <span class="ctrl-value">{{ lookSpeed.toFixed(3) }}</span>
      </div>

      <div class="ctrl-divider"></div>

      <button
        class="ctrl-btn"
        @click="showHelp = !showHelp"
        @touchend.stop.prevent="showHelp = !showHelp"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
        Help
      </button>

      <!-- Set Start Point button (explore mode only) -->
      <button
        v-if="exploreMode"
        class="ctrl-btn"
        :class="{ active: startingPointSet }"
        @click="onSetStartingPoint"
        @touchend.stop.prevent="onSetStartingPoint"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
        {{ startingPointSet ? 'Start Set' : 'Set Start' }}
      </button>

      <!-- Record button -->
      <button
        v-if="!exploreMode && showVideo"
        class="ctrl-btn"
        :class="{ 'recording-active': isRecording, 'recording-processing': recordingProcessing }"
        :disabled="recordingProcessing"
        @click="toggleRecording"
        @touchend.stop.prevent="toggleRecording"
      >
        <svg v-if="!isRecording && !recordingProcessing" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="6"/></svg>
        <svg v-else-if="recordingProcessing" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
        <svg v-else width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
        {{ recordingProcessing ? 'Processing...' : isRecording ? 'Stop Rec' : 'Record' }}
      </button>

      <button
        v-if="showStartButton"
        class="ctrl-btn ctrl-btn-accent"
        @click="onReconnect"
        @touchend.stop.prevent="onReconnect"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="6 3 20 12 6 21 6 3"/></svg>
        Start
      </button>
      <button
        v-else
        class="ctrl-btn"
        @click="recenter"
        @touchend.stop.prevent="recenter"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>
        </svg>
        Reset
      </button>
    </div>

    <!-- Scene Selector (below control panel) -->
    <div class="scene-bar">
      <div class="scene-bar-inner scrollbar-hide">
        <button
          v-for="scene in scenes"
          :key="scene.name"
          class="scene-card"
          :class="{ active: scene.name === currentScene }"
          @click="onSelectScene(scene.name)"
          @touchend.stop.prevent="onSelectScene(scene.name)"
        >
          <div class="scene-card-thumb">
            <img
              v-if="scene.thumbnail && !scene.thumbError"
              :src="scene.thumbnail"
              class="scene-card-img"
              draggable="false"
              @error="scene.thumbError = true"
            />
            <div v-else class="scene-card-fallback" :style="{ background: sceneGradient(scene.name) }">
              {{ scene.name.charAt(0).toUpperCase() }}
            </div>
          </div>
          <span class="scene-card-name">{{ scene.name }}</span>
        </button>

        <!-- Add scene from folder button -->
        <button
          class="scene-card scene-card-add"
          @click="showAddSceneModal = true"
          @touchend.stop.prevent="showAddSceneModal = true"
        >
          <div class="scene-card-thumb">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 5v14M5 12h14"/>
            </svg>
          </div>
          <span class="scene-card-name">Add</span>
        </button>

        <!-- Create new scene button -->
        <button
          class="scene-card scene-card-add"
          @click="showCreateSceneModal = true"
          @touchend.stop.prevent="showCreateSceneModal = true"
        >
          <div class="scene-card-thumb">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
            </svg>
          </div>
          <span class="scene-card-name">Create</span>
        </button>
      </div>
    </div>

    <!-- Add Scene Modal -->
    <div v-if="showAddSceneModal" class="add-scene-overlay" @click.self="closeAddSceneModal">
      <div class="add-scene-modal" @click.stop>
        <h3>Add Scene</h3>

        <!-- ── Idle: show form ── -->
        <template v-if="addSceneStatus === 'idle'">
          <label class="add-scene-filedrop">
            <input type="file" accept=".ply,.pt,.pth" class="add-scene-file-input" @change="onFileSelect" />
            <div v-if="!selectedFile" class="add-scene-filedrop-empty">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
              <span class="add-scene-filedrop-text">Choose .ply / .pt / .pth file</span>
            </div>
            <div v-else class="add-scene-filedrop-filled">
              <span class="add-scene-file-icon">📄</span>
              <div>
                <div class="add-scene-file-name">{{ selectedFile.name }}</div>
                <div class="add-scene-file-size">{{ formatFileSize(selectedFile.size) }}</div>
              </div>
            </div>
          </label>

          <div class="add-scene-field">
            <label class="add-scene-field-label">Thumbnail (optional)</label>
            <label class="add-scene-thumb-pick">
              <input type="file" accept="image/*" class="add-scene-file-input" @change="onThumbnailSelect" />
              <span v-if="!selectedThumbnail">Choose image…</span>
              <span v-else class="add-scene-thumb-chosen">✓ {{ selectedThumbnail.name }}</span>
            </label>
          </div>

          <div class="add-scene-field">
            <label class="add-scene-field-label">Prompt (optional)</label>
            <textarea v-model="promptText" class="add-scene-textarea" rows="2" placeholder="Text prompt for video generation"></textarea>
          </div>

          <div class="add-scene-divider"><span>or enter server folder path</span></div>
          <input v-model="addSceneFolderPath" type="text" class="add-scene-input" placeholder="e.g. C:\Users\...\DancingHall" @keyup.enter="onLoadScene" />
          <div v-if="addSceneError" class="add-scene-error">{{ addSceneError }}</div>
          <div class="add-scene-actions">
            <button class="add-scene-btn add-scene-btn-cancel" @click="closeAddSceneModal">Cancel</button>
            <button class="add-scene-btn add-scene-btn-load" @click="onLoadScene" :disabled="!canLoadScene">Load</button>
          </div>
        </template>

        <!-- ── Uploading: show progress ── -->
        <div v-else-if="addSceneStatus === 'uploading'" class="add-scene-progress-view">
          <div class="add-scene-steps">
            <div class="add-scene-step add-scene-step-active"><div class="add-scene-step-dot"></div><span>Uploading</span></div>
            <div class="add-scene-step"><div class="add-scene-step-dot"></div><span>Preparing</span></div>
            <div class="add-scene-step"><div class="add-scene-step-dot"></div><span>Ready</span></div>
          </div>
          <div class="add-scene-progress-label">{{ selectedFile ? selectedFile.name : 'file' }} · {{ uploadProgress }}%</div>
          <div class="add-scene-progress-bar-wrap"><div class="add-scene-progress-bar" :style="{ width: uploadProgress + '%' }"></div></div>
          <button class="add-scene-btn add-scene-btn-cancel add-scene-mt" @click="cancelAddScene">Cancel</button>
        </div>

        <!-- ── Processing: show spinner ── -->
        <div v-else-if="addSceneStatus === 'processing'" class="add-scene-progress-view">
          <div class="add-scene-steps">
            <div class="add-scene-step add-scene-step-done"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg><span>Upload</span></div>
            <div class="add-scene-step add-scene-step-active"><div class="add-scene-step-dot add-scene-step-dot-spin"></div><span>Preparing</span></div>
            <div class="add-scene-step"><div class="add-scene-step-dot"></div><span>Ready</span></div>
          </div>
          <div class="add-scene-processing-label">{{ addSceneStatusText }}</div>
          <div class="add-scene-processing-hint">This may take a few seconds…</div>
          <button class="add-scene-btn add-scene-btn-cancel add-scene-mt" @click="cancelAddScene">Cancel</button>
        </div>

        <!-- ── Error ── -->
        <div v-else-if="addSceneStatus === 'error'" class="add-scene-progress-view">
          <div class="add-scene-error-icon-big">!</div>
          <div class="add-scene-processing-label">{{ addSceneError }}</div>
          <button class="add-scene-btn add-scene-btn-load add-scene-mt" @click="resetAddSceneForm">Try Again</button>
        </div>
      </div>
    </div>

    <!-- Create Scene Modal (Pipeline Wizard) -->
    <div v-if="showCreateSceneModal" class="add-scene-overlay">
      <div class="add-scene-modal" @click.stop>
        <div class="add-scene-modal-header">
          <h3>Create New Scene</h3>
          <button class="add-scene-close-x" @click="onCreateSceneCloseX">✕</button>
        </div>

        <!-- ── Step 1: Upload Image ── -->
        <template v-if="createStep === 1">
          <label class="add-scene-filedrop">
            <input type="file" accept="image/*" class="add-scene-file-input" @change="onCreateImageSelect" />
            <div v-if="!createImagePreview" class="add-scene-filedrop-empty">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="M21 15l-5-5L5 21"/>
              </svg>
              <span class="add-scene-filedrop-text">Choose perspective image</span>
            </div>
            <div v-else class="create-image-preview">
              <img :src="createImagePreview" alt="Input preview" />
            </div>
          </label>
          <div class="add-scene-actions">
            <button class="add-scene-btn add-scene-btn-cancel" @click="closeCreateSceneModal">Cancel</button>
            <button class="add-scene-btn add-scene-btn-load" @click="createStep = 2" :disabled="!createImagePreview">Next</button>
          </div>
        </template>

        <!-- ── Step 2: Prompt Input ── -->
        <template v-if="createStep === 2">
          <div class="create-prompt-info">
            <div class="create-prompt-info-row">
              <span class="create-prompt-info-label">Prepended:</span>
              <code>Equirectangular 360 panorama.</code>
            </div>
            <div class="create-prompt-info-row">
              <span class="create-prompt-info-label">Appended:</span>
              <code>, high-quality, high resolution, sharp and detailed, clear, 8k, HDRI, Ultra HD</code>
            </div>
            <div class="create-prompt-info-hint">Back prompt will be inserted between your prompt and the suffix.</div>
          </div>

          <div class="add-scene-field">
            <label class="add-scene-field-label">Prompt (required)</label>
            <textarea v-model="createPrompt" class="add-scene-textarea" rows="3"
              placeholder="a bright breakfast alcove, curved booth seating, small round table, colorful patterned wall tiles, pendant lamp, one window with soft daylight, simple fruit bowl, cheerful creative design, clean geometry, realistic lighting"></textarea>
          </div>

          <div class="add-scene-field">
            <label class="add-scene-field-label">Back Prompt (optional)</label>
            <textarea v-model="createBackPrompt" class="add-scene-textarea" rows="2"
              placeholder="with additional alcove seating and a small kitchen behind"></textarea>
          </div>

          <div class="create-prompt-preview">
            <span class="create-prompt-preview-label">Full FluxFill Prompt:</span>
            <p class="create-prompt-preview-text">Equirectangular 360 panorama. {{ createPrompt }}{{ createBackPrompt.trim() ? ', ' + createBackPrompt : '' }}, high-quality, high resolution, sharp and detailed, clear, 8k, HDRI, Ultra HD</p>
          </div>

          <div class="add-scene-actions">
            <button class="add-scene-btn add-scene-btn-cancel" @click="createStep = 1">Back</button>
            <button class="add-scene-btn add-scene-btn-load" @click="onCreateSceneSubmit" :disabled="!createPrompt.trim()">Generate</button>
          </div>
        </template>

        <!-- ── Step 3: Generating ── -->
        <template v-if="createStep === 3">
          <div class="create-gen-steps">
            <div class="create-gen-step" :class="{ done: createGenStepIndex > 1, active: createGenStepIndex === 1 }">
              <div class="create-gen-step-icon" v-html="createGenStepIcon(0)"></div>
              <span>Panorama</span>
            </div>
            <div class="create-gen-step" :class="{ done: createGenStepIndex > 2, active: createGenStepIndex === 2 }">
              <div class="create-gen-step-icon" v-html="createGenStepIcon(1)"></div>
              <span>Depth</span>
            </div>
            <div class="create-gen-step" :class="{ done: createGenStepIndex > 3, active: createGenStepIndex === 3 }">
              <div class="create-gen-step-icon" v-html="createGenStepIcon(2)"></div>
              <span>Gaussians</span>
            </div>
            <div class="create-gen-step" :class="{ done: createGenStepIndex > 4, active: createGenStepIndex === 4 }">
              <div class="create-gen-step-icon" v-html="createGenStepIcon(3)"></div>
              <span>Register</span>
            </div>
          </div>

          <div class="create-gen-message" v-if="createGenMessage">{{ createGenMessage }}</div>

          <div class="create-gen-paths" v-if="createGenPaths.length">
            <div v-for="p in createGenPaths" :key="p.label" class="create-gen-path">
              <span class="create-gen-path-label">{{ p.label }}:</span>
              <code>{{ p.path }}</code>
            </div>
          </div>

          <div v-if="createGenError" class="add-scene-error">{{ createGenError }}</div>

          <div v-if="createGenDone" class="create-gen-done">
            <p>Scene "{{ createGenSceneName }}" is ready!</p>
            <button class="add-scene-btn add-scene-btn-load" @click="onCreateSceneDone">Enter Scene</button>
          </div>

          <button v-if="!createGenDone && !createGenError" class="add-scene-btn add-scene-btn-cancel add-scene-mt" @click="onCreateSceneCancel">Cancel</button>
        </template>
      </div>
    </div>

    <!-- Gaussian scale toast (inside root div, v-show + inline styles) -->
    <div
      v-show="gaussianScaleToast"
      style="position: fixed; bottom: 200px; left: 50%; transform: translateX(-50%); background: rgba(30,30,40,0.95); color: #FCD34D; padding: 12px 24px; border-radius: 12px; font-size: 13px; font-weight: 600; z-index: 9999; white-space: nowrap; box-shadow: 0 8px 24px rgba(0,0,0,0.4); border: 1px solid rgba(252,211,77,0.3); pointer-events: none; transition: opacity 0.3s;"
    >
      Gaussian scale will take effect on next scene change or reset.
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, onMounted, onUnmounted } from 'vue'
import { useTransport, getApiBaseUrl } from './composables/useTransport.js'
import { useCameraControl } from './composables/useCameraControl.js'
import { useScenes } from './composables/useScenes.js'
import Joystick from './components/Joystick.vue'
import DPad from './components/DPad.vue'

const logoUrl = `${import.meta.env.BASE_URL}moku-logo.png`
const originalLogoUrl = `${import.meta.env.BASE_URL}logo.jpg`

// Scene gradient colors for thumbnail fallback
const SCENE_GRADIENTS = [
  'linear-gradient(135deg, #667eea, #764ba2)',
  'linear-gradient(135deg, #f093fb, #f5576c)',
  'linear-gradient(135deg, #4facfe, #00f2fe)',
  'linear-gradient(135deg, #43e97b, #38f9d7)',
  'linear-gradient(135deg, #fa709a, #fee140)',
  'linear-gradient(135deg, #30cfd0, #330867)',
]
function sceneGradient(name) {
  let hash = 0
  for (let i = 0; i < name.length; i++) hash = name.charCodeAt(i) + ((hash << 5) - hash)
  return SCENE_GRADIENTS[Math.abs(hash) % SCENE_GRADIENTS.length]
}

// ─── Transport ──────────────────────────────────────────────────────────────────

const {
  connectionState,
  errorType,
  loading,
  loadingText,
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
  init: initTransport,
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
} = useTransport()

// ─── Camera Control ─────────────────────────────────────────────────────────────

const {
  moveKnob,
  lookViz,
  joyActive,
  lookJoyActive,
  handleDragStart,
  handleDragEnd,
  handleDragLook,
  updateJoy,
  updateLookJoy,
  recenter,
  moveSpeed,
  lookSpeed,
} = useCameraControl(sendControl, isConnected)

// ─── Scenes ─────────────────────────────────────────────────────────────────────

const { scenes, fetchScenes } = useScenes()

// ─── Gaussian Scale ─────────────────────────────────────────────────────────────

const gaussianScale = ref(1.0)
const gaussianScaleToast = ref(false)
let gaussianScaleToastTimer = null

async function onGaussianScaleChange() {
  // Show toast notification FIRST (before any async calls)
  console.log('[App] onGaussianScaleChange fired, scale=', gaussianScale.value)
  gaussianScaleToast.value = true
  if (gaussianScaleToastTimer) clearTimeout(gaussianScaleToastTimer)
  gaussianScaleToastTimer = setTimeout(() => {
    gaussianScaleToast.value = false
  }, 4000)

  try {
    const apiBase = await getApiBaseUrl()
    await fetch(`${apiBase}/api/gaussian_scale`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scale: gaussianScale.value }),
    })
  } catch (e) {
    console.warn('[App] Failed to set gaussian_scale:', e)
  }
}

// ─── Render Toggle ──────────────────────────────────────────────────────────────

function toggleRenderView() {
  showRender.value = !showRender.value
}

// ─── Explore Mode Actions ────────────────────────────────────────────────────────

function onStartDenoising() {
  console.log('[App] Start denoising clicked')
  sendStartDenoising()
}

function onSetStartingPoint() {
  console.log('[App] Set starting point clicked')
  sendSetStartingPoint()
}

// ─── Recording ───────────────────────────────────────────────────────────────────

function toggleRecording() {
  if (recordingProcessing.value) return
  if (isRecording.value) {
    console.log('[App] Stop recording clicked')
    stopRecording()
  } else {
    console.log('[App] Start recording clicked')
    startRecording()
  }
}

// ─── UI State ───────────────────────────────────────────────────────────────────

const isGameOver = ref(false)
const sessionKey = ref(0)
const showHelp = ref(false)
const pressedKeys = reactive(new Set())

function onKeyDown(event) {
  // Skip if user is typing in an input/textarea/contenteditable
  const tag = event.target?.tagName?.toLowerCase()
  if (tag === 'input' || tag === 'textarea' || event.target?.isContentEditable) return
  pressedKeys.add(event.key.toLowerCase())
}
function onKeyUp(event) {
  const tag = event.target?.tagName?.toLowerCase()
  if (tag === 'input' || tag === 'textarea' || event.target?.isContentEditable) return
  pressedKeys.delete(event.key.toLowerCase())
}

const showStartButton = computed(() => isGameOver.value || connectionState.value === 'error')

// ─── Timer ──────────────────────────────────────────────────────────────────────

const remainingSeconds = ref(SESSION_DURATION_SECONDS)
let countdownTimer = null
let sessionStartTime = null
let isCountingDown = false

function formatTime(totalSeconds) {
  const m = Math.floor(totalSeconds / 60)
  const s = totalSeconds % 60
  return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`
}
const timerDisplay = computed(() => formatTime(remainingSeconds.value))

function startCountdown() {
  if (isCountingDown) return
  isCountingDown = true
  sessionStartTime = Date.now()
  countdownTimer = setInterval(() => {
    const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000)
    const remaining = Math.max(0, SESSION_DURATION_SECONDS - elapsed)
    remainingSeconds.value = remaining
    if (remaining <= 0) { clearInterval(countdownTimer); countdownTimer = null; onSessionTimeout() }
  }, 100)
}

function resetCountdown() {
  if (countdownTimer) { clearInterval(countdownTimer); countdownTimer = null }
  isCountingDown = false
  sessionStartTime = null
  remainingSeconds.value = SESSION_DURATION_SECONDS
}

watch(firstFrameReceived, (received) => {
  if (received && !isCountingDown) startCountdown()
})

// ─── Status Config ──────────────────────────────────────────────────────────────

const STATUS_CONFIG = {
  connected: { text: 'Live', color: '#34C759', dotColor: '#34C759' },
  connecting: { text: 'Loading', color: '#FCD34D', dotColor: '#FCD34D' },
  disconnected: { text: 'Offline', color: '#9CA3AF', dotColor: '#9CA3AF' },
  error: { text: 'Error', color: '#EF4444', dotColor: '#EF4444' },
  gameover: { text: 'Done', color: '#EF4444', dotColor: '#EF4444' },
}

const statusConfig = computed(() => {
  if (isGameOver.value) return STATUS_CONFIG.gameover
  return STATUS_CONFIG[connectionState.value] || STATUS_CONFIG.connecting
})

// ─── Scene Selection ────────────────────────────────────────────────────────────

async function onSelectScene(sceneName) {
  if (isGameOver.value) isGameOver.value = false
  sessionKey.value++
  resetCountdown()
  const url = new URL(window.location.href)
  url.searchParams.set('world', sceneName)
  window.history.replaceState({}, '', url.toString())
  try { await connect(sceneName) } catch (e) { console.error('[App] Connection failed:', e) }
}

// ─── Add Scene (File Upload + Folder Path) ─────────────────────────────────────────

const showAddSceneModal = ref(false)
const addSceneFolderPath = ref('')
const addSceneError = ref('')
const addSceneStatus = ref('idle')  // 'idle' | 'uploading' | 'processing' | 'error'
const addSceneStatusText = ref('')
const selectedFile = ref(null)
const selectedThumbnail = ref(null)
const promptText = ref('')
const uploadProgress = ref(0)
let uploadXhr = null

const canLoadScene = computed(() => !!(selectedFile.value || addSceneFolderPath.value.trim()))

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB'
  if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + ' MB'
  return (bytes / 1073741824).toFixed(2) + ' GB'
}

function onFileSelect(event) {
  const file = event.target.files[0]
  if (file) { selectedFile.value = file; addSceneError.value = '' }
}

function onThumbnailSelect(event) {
  const file = event.target.files[0]
  if (file) selectedThumbnail.value = file
}

function closeAddSceneModal() { showAddSceneModal.value = false; resetAddSceneForm() }

function resetAddSceneForm() {
  addSceneStatus.value = 'idle'
  addSceneError.value = ''
  addSceneFolderPath.value = ''
  selectedFile.value = null
  selectedThumbnail.value = null
  promptText.value = ''
  uploadProgress.value = 0
  addSceneStatusText.value = ''
  uploadXhr = null
}

function cancelAddScene() {
  if (uploadXhr) { uploadXhr.abort(); uploadXhr = null }
  resetAddSceneForm()
}

async function onLoadScene() {
  addSceneError.value = ''
  if (selectedFile.value) { await onUploadScene() }
  else if (addSceneFolderPath.value.trim()) { await onAddSceneFolder() }
}

async function onAddSceneFolder() {
  addSceneStatus.value = 'processing'
  addSceneStatusText.value = 'Scanning folder and preparing scene…'
  try {
    const apiBase = await getApiBaseUrl()
    const response = await fetch(`${apiBase}/api/add_scene_folder`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ folder_path: addSceneFolderPath.value }),
    })
    const data = await response.json()
    if (!response.ok) { addSceneError.value = data.error || 'Failed to add scene'; addSceneStatus.value = 'error'; return }
    console.log('[App] Scene added from folder:', data)
    await fetchScenes()
    closeAddSceneModal()
    await onSelectScene(data.name)
  } catch (e) {
    console.error('[App] Failed to add scene from folder:', e)
    addSceneError.value = `Network error: ${e.message || e}`
    addSceneStatus.value = 'error'
  }
}

async function onUploadScene() {
  addSceneStatus.value = 'uploading'
  uploadProgress.value = 0
  const formData = new FormData()
  formData.append('file', selectedFile.value)
  if (selectedThumbnail.value) formData.append('thumbnail', selectedThumbnail.value)
  if (promptText.value.trim()) formData.append('prompt', promptText.value.trim())
  try {
    const apiBase = await getApiBaseUrl()
    const result = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      uploadXhr = xhr
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) uploadProgress.value = Math.round((e.loaded / e.total) * 100)
      })
      xhr.upload.addEventListener('load', () => {
        addSceneStatus.value = 'processing'
        addSceneStatusText.value = promptText.value.trim() ? 'Encoding prompt for video generation…' : 'Preparing scene…'
      })
      xhr.addEventListener('load', () => {
        try {
          const data = JSON.parse(xhr.responseText)
          if (xhr.status >= 200 && xhr.status < 300) resolve(data)
          else reject(new Error(data.error || 'Upload failed'))
        } catch (e) { reject(new Error('Invalid server response')) }
      })
      xhr.addEventListener('error', () => reject(new Error('Network error during upload')))
      xhr.addEventListener('abort', () => reject(new Error('Upload cancelled')))
      xhr.open('POST', `${apiBase}/api/upload_scene`)
      xhr.send(formData)
    })
    console.log('[App] Scene uploaded:', result)
    await fetchScenes()
    closeAddSceneModal()
    await onSelectScene(result.name)
  } catch (e) {
    console.error('[App] Upload failed:', e)
    if (e.message === 'Upload cancelled') resetAddSceneForm()
    else { addSceneError.value = e.message || 'Upload failed'; addSceneStatus.value = 'error' }
  }
}

// ─── Create Scene (Pipeline: Pano → Depth → 3DGS) ─────────────────────────────

const showCreateSceneModal = ref(false)
const createStep = ref(1)  // 1: image upload, 2: prompt, 3: generating
const createImageFile = ref(null)
const createImagePreview = ref('')
const createPrompt = ref('')
const createBackPrompt = ref('')
const createGenStepIndex = ref(0)  // 0: idle, 1: panorama, 2: depth, 3: gaussians, 4: register, 5+: done
const createGenMessage = ref('')
const createGenError = ref('')
const createGenDone = ref(false)
const createGenSceneName = ref('')
const createGenPaths = ref([])
let createAbortController = null

const CREATE_GEN_STEPS = ['panorama', 'depth', 'gaussian', 'register']

function closeCreateSceneModal() {
  if (createAbortController) { createAbortController.abort(); createAbortController = null }
  showCreateSceneModal.value = false
  createStep.value = 1
  createImageFile.value = null
  createImagePreview.value = ''
  createPrompt.value = ''
  createBackPrompt.value = ''
  createGenStepIndex.value = 0
  createGenMessage.value = ''
  createGenError.value = ''
  createGenDone.value = false
  createGenSceneName.value = ''
  createGenPaths.value = []
}

function onCreateSceneCancel() {
  if (createAbortController) { createAbortController.abort(); createAbortController = null }
  createGenError.value = 'Cancelled by user'
}

function onCreateSceneCloseX() {
  // If generating and not done, ask for confirmation
  if (createStep.value === 3 && !createGenDone.value && !createGenError.value) {
    if (!confirm('A generation task is still running. Cancel and close?')) return
    if (createAbortController) { createAbortController.abort(); createAbortController = null }
  }
  closeCreateSceneModal()
}

function onCreateImageSelect(event) {
  const file = event.target.files[0]
  if (file) {
    createImageFile.value = file
    const reader = new FileReader()
    reader.onload = (e) => { createImagePreview.value = e.target.result }
    reader.readAsDataURL(file)
  }
}

function createGenStepIcon(index) {
  if (createGenStepIndex.value > index + 1) {
    return '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>'
  } else if (createGenStepIndex.value === index + 1) {
    return '<div class="add-scene-step-dot add-scene-step-dot-spin"></div>'
  }
  return '<div class="add-scene-step-dot"></div>'
}

function handleCreateGenEvent(data) {
  const stepIndex = CREATE_GEN_STEPS.indexOf(data.step)
  if (data.status === 'processing') {
    createGenStepIndex.value = stepIndex + 1
    createGenMessage.value = data.message || ''
  } else if (data.status === 'done') {
    // 'done' is not in CREATE_GEN_STEPS, so stepIndex=-1 → set to 99 so all stages show as completed
    createGenStepIndex.value = data.step === 'done' ? 99 : stepIndex + 2
    createGenMessage.value = ''
    if (data.path) {
      const labels = { panorama: 'Panorama', depth: 'Depth', gaussian: '3DGS PLY' }
      createGenPaths.value.push({ label: labels[data.step] || data.step, path: data.path })
    }
    if (data.step === 'init' && data.scene_name) {
      createGenSceneName.value = data.scene_name
    }
    if (data.step === 'done') {
      createGenDone.value = true
      createGenSceneName.value = data.scene_name
      createGenMessage.value = ''
    }
  } else if (data.status === 'error') {
    createGenError.value = data.message || 'Unknown error'
  }
}

async function onCreateSceneSubmit() {
  createStep.value = 3
  createGenStepIndex.value = 0
  createGenError.value = ''
  createGenDone.value = false
  createGenPaths.value = []

  const formData = new FormData()
  formData.append('image', createImageFile.value)
  formData.append('prompt', createPrompt.value)
  formData.append('back_prompt', createBackPrompt.value)

  try {
    const apiBase = await getApiBaseUrl()
    createAbortController = new AbortController()
    const response = await fetch(`${apiBase}/api/generate_pipeline`, {
      method: 'POST',
      body: formData,
      signal: createAbortController.signal,
    })

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop()
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try { handleCreateGenEvent(JSON.parse(line.slice(6))) } catch (e) { /* skip */ }
        }
      }
    }
  } catch (e) {
    if (e.name === 'AbortError') return
    createGenError.value = e.message || 'Generation failed'
  }
}

async function onCreateSceneDone() {
  const sceneName = createGenSceneName.value
  closeCreateSceneModal()
  await fetchScenes()
  if (sceneName) await onSelectScene(sceneName)
}

function onSessionTimeout() {
  isGameOver.value = true
  disconnect()
}

async function onReconnect() {
  isGameOver.value = false
  sessionKey.value++
  resetCountdown()
  if (currentScene.value) await onSelectScene(currentScene.value)
}

// ─── Drag for look (on screen) ──────────────────────────────────────────────────

let isDragging = false
let lastX = 0, lastY = 0
const videoElement = ref(null)

function onMouseDown(e) { isDragging = true; lastX = e.clientX; lastY = e.clientY; handleDragStart() }
function onMouseMove(e) { if (!isDragging) return; handleDragLook(e.clientX - lastX, e.clientY - lastY); lastX = e.clientX; lastY = e.clientY }
function onMouseUp() { if (!isDragging) return; isDragging = false; handleDragEnd() }

let activeTouchId = null
function onTouchStart(e) { if (activeTouchId !== null) return; const t = e.changedTouches[0]; activeTouchId = t.identifier; lastX = t.clientX; lastY = t.clientY; handleDragStart() }
function onTouchMove(e) { for (const t of e.changedTouches) { if (t.identifier !== activeTouchId) continue; handleDragLook(t.clientX - lastX, t.clientY - lastY); lastX = t.clientX; lastY = t.clientY } }
function onTouchEnd(e) { for (const t of e.changedTouches) { if (t.identifier !== activeTouchId) continue; activeTouchId = null; handleDragEnd() } }

watch(() => mediaStream, (stream) => { if (videoElement.value && stream) videoElement.value.srcObject = stream }, { immediate: true })
watch(videoElement, (el) => { if (el && mediaStream.value) el.srcObject = mediaStream.value })

// ─── Lifecycle ──────────────────────────────────────────────────────────────────

onMounted(async () => {
  window.addEventListener('keydown', onKeyDown)
  window.addEventListener('keyup', onKeyUp)
  onTimeUp.value = onSessionTimeout
  window.addEventListener('beforeunload', () => disconnect())
  document.body.style.overscrollBehavior = 'none'
  document.documentElement.style.overscrollBehavior = 'none'

  await Promise.all([initTransport(), fetchScenes()])

  const urlScene = new URLSearchParams(window.location.search).get('world')
  if (urlScene && scenes.value.some(s => s.name === urlScene)) {
    await onSelectScene(urlScene)
  } else if (scenes.value.length > 0) {
    await onSelectScene(scenes.value[0].name)
  }
})

onUnmounted(() => {
  window.removeEventListener('keydown', onKeyDown)
  window.removeEventListener('keyup', onKeyUp)
  disconnect()
})
</script>

<style scoped>
/* ─── Console Shell Layout (3D depth) ─── */
.console-shell {
  display: flex;
  align-items: stretch;
  max-width: 1200px;
  width: 95vw;
  max-height: 85vh;
  background: linear-gradient(145deg, #F8F4EF, #EDE8E3);
  border-radius: 40px;
  box-shadow:
    0 30px 80px -20px rgba(0, 0, 0, 0.2),
    0 10px 30px -5px rgba(0, 0, 0, 0.08),
    inset 0 2px 0 rgba(255, 255, 255, 0.6),
    inset 0 -2px 4px rgba(0, 0, 0, 0.05),
    0 0 0 1px rgba(0, 0, 0, 0.04);
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.4);
}

/* Wings */
.console-wing {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
  padding: 24px 20px;
  min-width: 140px;
  flex-shrink: 0;
  position: relative;
}

.console-wing-left {
  background: linear-gradient(135deg, rgba(255,255,255,0.3), rgba(0,0,0,0.02));
}

.console-wing-right {
  background: linear-gradient(225deg, rgba(255,255,255,0.3), rgba(0,0,0,0.02));
}

@media (max-width: 768px) {
  .console-wing { min-width: 95px; padding: 14px 10px; }
  .console-shell { border-radius: 28px; }
}

.wing-top, .wing-bottom { flex-shrink: 0; }
.wing-center { flex: 1; display: flex; align-items: center; justify-content: center; }

/* Page-level logo (fixed top-left corner) */
.page-logo {
  position: fixed;
  top: 16px;
  left: 16px;
  z-index: 100;
  pointer-events: none;
}
.page-logo img {
  width: 56px;
  height: 56px;
  object-fit: contain;
  opacity: 0.9;
}

/* Logo (inside console wing) */
.console-logo {
  width: 52px;
  height: 52px;
  object-fit: contain;
  border-radius: 14px;
  opacity: 0.9;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* Screen */
.console-screen-wrap {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #0a0a0f;
  border-radius: 16px;
  margin: 14px 0;
  overflow: hidden;
  min-height: 300px;
  border: 4px solid #1a1a24;
  box-shadow:
    inset 0 4px 12px rgba(0,0,0,0.6),
    inset 0 0 30px rgba(0,0,0,0.3),
    0 2px 0 rgba(255,255,255,0.15),
    0 -1px 3px rgba(0,0,0,0.2);
}

.screen-status-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 12px;
  background: rgba(0,0,0,0.7);
  font-size: 10px;
  font-weight: 500;
  color: #9CA3AF;
  border-bottom: 1px solid rgba(255,255,255,0.05);
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 6px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.status-brand {
  font-weight: 600;
  letter-spacing: 0.1em;
  color: #6B7280;
}

.status-timer {
  font-variant-numeric: tabular-nums;
  color: #9CA3AF;
}

.screen-display {
  flex: 1;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  cursor: grab;
  background: radial-gradient(ellipse at center, #111118, #080810);
}
.screen-display:active { cursor: grabbing; }

.screen-split {
  width: 100%;
  height: 100%;
  display: flex;
  gap: 2px;
}

.screen-single {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.screen-panel {
  flex: 1;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.panel-label {
  position: absolute;
  top: 4px;
  left: 6px;
  font-size: 9px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: rgba(255,255,255,0.5);
  background: rgba(0,0,0,0.5);
  padding: 2px 6px;
  border-radius: 4px;
}

.screen-img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.screen-placeholder {
  color: #4B5563;
  font-size: 12px;
  font-weight: 500;
}

.screen-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(10, 10, 15, 0.92);
  z-index: 20;
}

.overlay-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  color: #D1D5DB;
  font-size: 13px;
  text-align: center;
  padding: 0 24px;
}

.overlay-error { color: #FCA5A5; }

.error-icon {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: rgba(239, 68, 68, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 18px;
  color: #EF4444;
}

.spinner {
  width: 28px;
  height: 28px;
  border: 2px solid #4B5563;
  border-top-color: var(--blue-primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }

/* Control Panel (slim horizontal bar below console) */
.control-panel {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 8px 20px;
  width: 95vw;
  max-width: 1200px;
  background: linear-gradient(145deg, #F8F4EF, #EDE8E3);
  border-radius: 18px;
  box-shadow:
    0 8px 24px -8px rgba(0,0,0,0.12),
    0 2px 8px -2px rgba(0,0,0,0.06),
    inset 0 1px 0 rgba(255,255,255,0.6),
    0 0 0 1px rgba(0,0,0,0.04);
  margin-top: 10px;
  flex-wrap: wrap;
  border: 1px solid rgba(255,255,255,0.4);
}

.ctrl-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 10px;
  background: rgba(0,0,0,0.05);
  border: 1px solid rgba(0,0,0,0.06);
  font-size: 11px;
  font-weight: 500;
  color: #6B7280;
  cursor: pointer;
  transition: all 0.15s;
  white-space: nowrap;
}
.ctrl-btn:hover { background: rgba(0,0,0,0.08); color: #4B5563; }
.ctrl-btn:active { transform: scale(0.96); }
.ctrl-btn.active {
  background: var(--blue-primary);
  color: white;
  border-color: var(--blue-primary);
  box-shadow: 0 2px 8px rgba(74,143,231,0.3);
}
.ctrl-btn-accent {
  background: linear-gradient(180deg, #F7B84C, #E89E1A);
  color: white;
  border-color: #E89E1A;
  box-shadow: 0 2px 8px rgba(232,158,26,0.3);
}
.ctrl-btn-accent:hover { background: linear-gradient(180deg, #F5A623, #D08C0A); color: white; }

.ctrl-divider {
  width: 1px;
  height: 20px;
  background: rgba(0,0,0,0.08);
  flex-shrink: 0;
}

.ctrl-slider-group {
  display: flex;
  align-items: center;
  gap: 6px;
}

.ctrl-label {
  font-size: 10px;
  font-weight: 500;
  color: #6B7280;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  white-space: nowrap;
}

.ctrl-value {
  font-size: 10px;
  font-variant-numeric: tabular-nums;
  color: #9CA3AF;
  min-width: 30px;
  text-align: right;
}

.ctrl-slider {
  width: 64px;
  height: 4px;
  -webkit-appearance: none;
  appearance: none;
  background: #D1D5DB;
  border-radius: 2px;
  outline: none;
  cursor: pointer;
}

.ctrl-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--blue-primary);
  border: 2px solid white;
  box-shadow: 0 1px 4px rgba(74,143,231,0.4);
  cursor: pointer;
}

@media (max-width: 768px) {
  .control-panel { gap: 8px; padding: 6px 12px; border-radius: 14px; }
  .ctrl-btn { padding: 5px 8px; font-size: 10px; }
  .ctrl-slider { width: 48px; }
}

/* Orange pill buttons */
.pill-btn {
  width: 36px;
  height: 20px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: transform 0.1s, box-shadow 0.1s;
  border: none;
  box-shadow: 0 3px 6px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.3);
}
.pill-btn:active { transform: scale(0.9); box-shadow: 0 1px 3px rgba(0,0,0,0.2); }

.pill-btn-orange {
  background: linear-gradient(180deg, #F7B84C, #E89E1A);
  color: white;
}
.pill-btn-orange:hover { background: linear-gradient(180deg, #F5A623, #D08C0A); }

/* Face buttons (3D raised) */
.face-buttons {
  position: relative;
  width: 76px;
  height: 76px;
}

.face-btn {
  position: absolute;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  font-weight: 700;
  color: white;
  box-shadow: 0 3px 6px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.3);
  border: 1px solid rgba(0,0,0,0.1);
}

.face-x { top: 0; left: 50%; transform: translateX(-50%); background: linear-gradient(180deg, #5BA0F0, #3D80D0); }
.face-y { top: 50%; left: 0; transform: translateY(-50%); background: linear-gradient(180deg, #70CCF0, #4AABE8); }
.face-a { top: 50%; right: 0; transform: translateY(-50%); background: linear-gradient(180deg, #FBBD48, #E89E1A); }
.face-b { bottom: 0; left: 50%; transform: translateX(-50%); background: linear-gradient(180deg, #5BA0F0, #3D80D0); }

/* Gen-only mode: wider wings for balance */
.console-shell.gen-only-mode .console-wing {
  min-width: 160px;
  padding: 24px 24px;
}

.console-shell.gen-only-mode .screen-single {
  justify-content: center;
  padding: 8px;
}

.console-shell.gen-only-mode .screen-single .screen-img {
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
  object-fit: contain;
}

@media (max-width: 768px) {
  .console-shell.gen-only-mode .console-wing { min-width: 100px; padding: 14px 10px; }
}

/* Scene cards */
.scene-bar {
  margin-top: 10px;
  width: 95vw;
  max-width: 1200px;
  overflow-x: auto;
}

.scene-bar-inner {
  display: flex;
  gap: 12px;
  padding: 4px 0;
  min-width: max-content;
}

.scene-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 8px;
  border-radius: 14px;
  background: rgba(255,255,255,0.08);
  border: 2px solid transparent;
  cursor: pointer;
  transition: all 0.15s;
  flex-shrink: 0;
}
.scene-card:hover { background: rgba(255,255,255,0.15); }
.scene-card:active { transform: scale(0.95); }
.scene-card.active {
  border-color: var(--blue-primary);
  background: rgba(74,143,231,0.1);
  box-shadow: 0 4px 12px rgba(74,143,231,0.2);
}

.scene-card-thumb {
  width: 56px;
  height: 56px;
  border-radius: 12px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0,0,0,0.1);
}

.scene-card-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.scene-card-fallback {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
  font-weight: 700;
  color: white;
  text-transform: uppercase;
}

.scene-card-name {
  font-size: 11px;
  font-weight: 500;
  color: #9CA3AF;
  text-transform: capitalize;
}

@media (max-width: 768px) {
  .scene-card-thumb { width: 44px; height: 44px; }
  .scene-card-name { font-size: 10px; }
}

/* Add scene card ("+" button) */
.scene-card-add { border: 2px dashed rgba(255,255,255,0.2); background: rgba(255,255,255,0.03); }
.scene-card-add .scene-card-thumb { color: rgba(255,255,255,0.4); }
.scene-card-add:hover { border-color: var(--blue-primary); background: rgba(74,143,231,0.05); }
.scene-card-add:hover .scene-card-thumb { color: var(--blue-primary); }

/* Add scene modal */
.add-scene-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); display: flex; align-items: center; justify-content: center; z-index: 200; backdrop-filter: blur(4px); }
.add-scene-modal { background: #1F2937; border-radius: 20px; padding: 28px; max-width: 480px; width: 90%; box-shadow: 0 20px 60px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.1); }
.add-scene-modal h3 { color: white; font-size: 18px; font-weight: 600; margin-bottom: 20px; text-align: center; }
.add-scene-modal-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; }
.add-scene-modal-header h3 { margin-bottom: 0; }
.add-scene-close-x { background: none; border: none; color: rgba(255,255,255,0.5); font-size: 18px; cursor: pointer; padding: 4px 8px; border-radius: 6px; line-height: 1; transition: all 0.15s; }
.add-scene-close-x:hover { color: white; background: rgba(255,255,255,0.1); }

/* File drop zone */
.add-scene-filedrop { display: block; border: 2px dashed rgba(255,255,255,0.15); border-radius: 14px; padding: 24px; text-align: center; cursor: pointer; transition: all 0.15s; margin-bottom: 16px; }
.add-scene-filedrop:hover { border-color: var(--blue-primary); background: rgba(74,143,231,0.05); }
.add-scene-file-input { display: none; }
.add-scene-filedrop-empty { display: flex; flex-direction: column; align-items: center; gap: 8px; color: #6B7280; }
.add-scene-filedrop-text { font-size: 13px; }
.add-scene-filedrop-filled { display: flex; align-items: center; gap: 12px; justify-content: center; }
.add-scene-file-icon { font-size: 28px; }
.add-scene-file-name { color: white; font-size: 13px; font-weight: 500; text-align: left; }
.add-scene-file-size { color: #6B7280; font-size: 11px; text-align: left; }

/* Form fields */
.add-scene-field { margin-bottom: 14px; }
.add-scene-field-label { display: block; color: #9CA3AF; font-size: 11px; font-weight: 500; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em; }
.add-scene-thumb-pick { display: inline-block; padding: 6px 14px; border-radius: 8px; background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1); color: #9CA3AF; font-size: 12px; cursor: pointer; transition: all 0.15s; }
.add-scene-thumb-pick:hover { border-color: var(--blue-primary); color: white; }
.add-scene-thumb-chosen { color: #34D399; }
.add-scene-textarea { width: 100%; padding: 8px 12px; border-radius: 10px; background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1); color: white; font-size: 13px; font-family: 'Inter', sans-serif; outline: none; resize: vertical; box-sizing: border-box; transition: border-color 0.15s; }
.add-scene-textarea:focus { border-color: var(--blue-primary); }
.add-scene-textarea::placeholder { color: #6B7280; }

/* Divider */
.add-scene-divider { display: flex; align-items: center; gap: 12px; margin: 16px 0; }
.add-scene-divider::before, .add-scene-divider::after { content: ''; flex: 1; height: 1px; background: rgba(255,255,255,0.1); }
.add-scene-divider span { color: #6B7280; font-size: 11px; white-space: nowrap; }

/* Input (folder path) */
.add-scene-input { width: 100%; padding: 10px 14px; border-radius: 10px; background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1); color: white; font-size: 13px; font-family: 'Inter', sans-serif; outline: none; transition: border-color 0.15s; box-sizing: border-box; }
.add-scene-input:focus { border-color: var(--blue-primary); }
.add-scene-input::placeholder { color: #6B7280; }

/* Error message */
.add-scene-error { margin-top: 10px; padding: 8px 12px; border-radius: 8px; background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.3); color: #FCA5A5; font-size: 12px; }

/* Action buttons */
.add-scene-actions { display: flex; gap: 10px; margin-top: 18px; justify-content: flex-end; }
.add-scene-btn { padding: 8px 20px; border-radius: 10px; font-size: 13px; font-weight: 500; border: none; cursor: pointer; transition: all 0.15s; }
.add-scene-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.add-scene-btn-cancel { background: rgba(255,255,255,0.08); color: #D1D5DB; }
.add-scene-btn-cancel:hover:not(:disabled) { background: rgba(255,255,255,0.15); }
.add-scene-btn-load { background: var(--blue-primary); color: white; }
.add-scene-btn-load:hover:not(:disabled) { background: var(--blue-hover); }
.add-scene-mt { margin-top: 16px; }

/* Progress view */
.add-scene-progress-view { display: flex; flex-direction: column; align-items: center; padding: 8px 0; }

/* Step indicator */
.add-scene-steps { display: flex; align-items: center; gap: 8px; margin-bottom: 24px; }
.add-scene-step { display: flex; align-items: center; gap: 6px; color: #4B5563; font-size: 12px; }
.add-scene-step-done { color: #34D399; }
.add-scene-step-active { color: var(--blue-primary); }
.add-scene-step-dot { width: 10px; height: 10px; border-radius: 50%; background: #374151; }
.add-scene-step-done .add-scene-step-dot, .add-scene-step-active .add-scene-step-dot { background: currentColor; }
.add-scene-step-dot-spin { animation: add-scene-pulse 1s ease-in-out infinite; }
@keyframes add-scene-pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.7); } }

/* Progress bar */
.add-scene-progress-label { color: #D1D5DB; font-size: 13px; margin-bottom: 10px; text-align: center; }
.add-scene-progress-bar-wrap { width: 100%; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden; }
.add-scene-progress-bar { height: 100%; background: var(--blue-primary); border-radius: 3px; transition: width 0.2s ease; }

/* Processing */
.add-scene-processing-label { color: white; font-size: 14px; font-weight: 500; text-align: center; margin-bottom: 6px; }
.add-scene-processing-hint { color: #6B7280; font-size: 12px; text-align: center; }

/* Error icon */
.add-scene-error-icon-big { width: 48px; height: 48px; border-radius: 50%; background: rgba(239,68,68,0.2); display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 24px; color: #EF4444; margin-bottom: 16px; }

/* Help overlay */
.help-overlay {
  position: absolute;
  inset: 0;
  background: rgba(0,0,0,0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 50;
  border-radius: 12px;
}

.help-card {
  background: #1F2937;
  border-radius: 16px;
  padding: 24px;
  max-width: 380px;
  width: 90%;
  box-shadow: 0 20px 40px rgba(0,0,0,0.4);
  border: 1px solid rgba(255,255,255,0.1);
}

.help-card h3 {
  color: white;
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 16px;
  text-align: center;
}

.help-grid {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.help-item {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 12px;
  color: #D1D5DB;
}

.help-key {
  font-weight: 600;
  font-size: 11px;
  color: var(--blue-primary);
  min-width: 100px;
  flex-shrink: 0;
}

.help-close {
  margin-top: 16px;
  width: 100%;
  padding: 8px;
  border-radius: 8px;
  background: var(--blue-primary);
  color: white;
  font-weight: 500;
  font-size: 13px;
  border: none;
  cursor: pointer;
}
.help-close:hover { background: var(--blue-hover); }

/* ─── Explore mode overlay ─── */
.explore-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 15;
  pointer-events: none;
}

.start-denoising-btn {
  pointer-events: auto;
  padding: 14px 28px;
  border-radius: 16px;
  background: linear-gradient(180deg, #4A8FE7, #2D6FD4);
  color: white;
  font-size: 14px;
  font-weight: 600;
  border: 2px solid rgba(255, 255, 255, 0.3);
  cursor: pointer;
  box-shadow: 0 8px 32px rgba(74, 143, 231, 0.5), 0 0 0 0 rgba(74, 143, 231, 0.4);
  animation: denoise-pulse 2s ease-in-out infinite;
  transition: transform 0.15s, box-shadow 0.15s;
  text-align: center;
  line-height: 1.3;
  max-width: 80%;
}
.start-denoising-btn:hover {
  transform: scale(1.05);
  box-shadow: 0 12px 40px rgba(74, 143, 231, 0.6);
}
.start-denoising-btn:active {
  transform: scale(0.95);
}

@keyframes denoise-pulse {
  0%, 100% {
    box-shadow: 0 8px 32px rgba(74, 143, 231, 0.5), 0 0 0 0 rgba(74, 143, 231, 0.4);
  }
  50% {
    box-shadow: 0 8px 32px rgba(74, 143, 231, 0.5), 0 0 0 12px rgba(74, 143, 231, 0);
  }
}

/* ─── Recording button states ─── */
.ctrl-btn.recording-active {
  background: rgba(239, 68, 68, 0.15);
  border-color: rgba(239, 68, 68, 0.4);
  color: #EF4444;
  animation: rec-blink 1s ease-in-out infinite;
}
@keyframes rec-blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}
.ctrl-btn.recording-processing {
  opacity: 0.6;
  cursor: not-allowed;
}
.ctrl-btn:disabled {
  cursor: not-allowed;
}

/* ─── Recording indicator in status bar ─── */
.rec-indicator {
  display: flex;
  align-items: center;
  gap: 4px;
  color: #EF4444;
  font-size: 9px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.rec-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #EF4444;
  animation: rec-blink 1s ease-in-out infinite;
}

/* ─── Create Scene Wizard ─── */
.create-image-preview { display: flex; justify-content: center; }
.create-image-preview img { max-height: 200px; border-radius: 12px; object-fit: contain; }
.create-prompt-info {
  background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.3);
  border-radius: 12px; padding: 12px; margin-bottom: 16px; font-size: 11px;
}
.create-prompt-info-row { display: flex; gap: 8px; margin-bottom: 6px; align-items: baseline; }
.create-prompt-info-label { color: #818CF8; font-weight: 600; white-space: nowrap; min-width: 70px; }
.create-prompt-info-row code { color: #C4B5FD; font-size: 10px; word-break: break-all; }
.create-prompt-info-hint { color: #9CA3AF; font-size: 10px; margin-top: 4px; }
.create-prompt-preview {
  background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.2);
  border-radius: 12px; padding: 12px; margin-top: 12px; font-size: 11px;
}
.create-prompt-preview-label { color: #4ADE80; font-weight: 600; display: block; margin-bottom: 4px; font-size: 10px; }
.create-prompt-preview-text { color: #D1FAE5; font-size: 11px; line-height: 1.5; word-break: break-word; margin: 0; }
.create-gen-steps { display: flex; justify-content: space-between; margin-bottom: 20px; }
.create-gen-step { display: flex; flex-direction: column; align-items: center; gap: 6px; opacity: 0.4; transition: opacity 0.3s; }
.create-gen-step.active, .create-gen-step.done { opacity: 1; }
.create-gen-step-icon { width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 2px solid #374151; }
.create-gen-step.active .create-gen-step-icon { border-color: #FCD34D; color: #FCD34D; }
.create-gen-step.done .create-gen-step-icon { border-color: #34D399; color: #34D399; background: rgba(52,211,153,0.1); }
.create-gen-step span { color: #9CA3AF; font-size: 10px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
.create-gen-step.active span, .create-gen-step.done span { color: #F3F4F6; }
.create-gen-message { color: #FCD34D; font-size: 12px; text-align: center; margin-bottom: 12px; }
.create-gen-paths { margin-bottom: 12px; }
.create-gen-path { display: flex; gap: 6px; align-items: baseline; margin-bottom: 4px; font-size: 10px; }
.create-gen-path-label { color: #9CA3AF; white-space: nowrap; }
.create-gen-path code { color: #6EE7B7; font-size: 10px; word-break: break-all; }
.create-gen-done { text-align: center; }
.create-gen-done p { color: #34D399; font-size: 14px; margin-bottom: 12px; }
</style>
