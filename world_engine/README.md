# MoVerse World Engine — Frontend

Cloud-rendered 3D world streaming client, supporting both WebRTC (UDP) and WebSocket (TCP) transport modes.

## Tech Stack

- **Vue 3** + Composition API
- **Vite 6**
- **TailwindCSS v4**

## Project Structure

```
src/
├── composables/
│   ├── useTransport.js      # WebRTC/WS connection core (signaling, heartbeat, frame transport)
│   ├── useCameraControl.js  # Keyboard/mouse/joystick input → control commands
│   └── useScenes.js         # Fetch scene list from backend
├── components/
│   ├── VideoPanel.vue       # Video stream/frame display + drag look
│   ├── StatusBar.vue        # Connection status indicator + countdown
│   ├── Joystick.vue         # Mobile virtual joystick (touch + mouse)
│   └── SceneSelector.vue    # Bottom scene switcher bar
├── App.vue                  # Main component
├── main.js                  # Entry point
└── style.css                # Global styles (TailwindCSS + custom)
```

## Quick Start

### 1. Install dependencies

```bash
cd world_engine
npm install
```

### 2. Start dev server

```bash
npm run dev
```

Open `http://localhost:5173/moverse/` in browser.

### 3. Build for production

```bash
npm run build
```

Output in `dist/` directory, deploy to any static file server.

## Backend API Requirements

The frontend requires a backend to run. The backend must implement:

| Endpoint | Description |
|----------|-------------|
| `GET /api/config` | Returns `{ transport: "tcp" \| "udp" }` |
| `GET /api/scenes` | Returns `{ scenes: [{ name, thumbnail }] }` |
| `WS /ws/session` | WebRTC signaling channel (offer/answer/candidate/queue_status/error/timeup/pong) |
| `WS /ws/frames` | TCP mode Blob frame transport channel |

## URL Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `world` | — | Directly enter specified scene |
| `transport` | From server | Force `tcp` or `udp` |
| `level` | `main` | Backend environment: `base` or `main` |
| `turn` | `0` | Enable TURN relay: `1` |
| `turn_host` | — | TURN server address |
| `turn_port` | `3478` | TURN server port |
| `turn_user` | — | TURN username |
| `turn_pass` | — | TURN password |
| `move_speed` | `0.4` | Movement speed |
| `boost` | `1.5` | Shift boost multiplier |
| `mouse_look_gain` | `0.004` | Mouse look sensitivity |
| `look_joy_gain` | `0.04` | Joystick look sensitivity |
| `kb_look_gain` | `0.05` | Keyboard arrow key look sensitivity |
| `debug` | `0` | Enable Eruda mobile debug: `1` |

## Keyboard Controls

| Key | Function |
|-----|----------|
| `W A S D` | Move |
| `Shift` | Boost movement |
| `Arrow Keys` | Rotate view |
| `Mouse Drag` | Rotate view |
| `Z / X` | Adjust focal length |
| `Q` | Build local map |
| `E` | Clear local map |
| `R` | Enable local map |
