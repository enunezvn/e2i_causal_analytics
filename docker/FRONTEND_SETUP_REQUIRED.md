# Frontend Setup Required

## Status: Docker Configuration Complete ✅ | React App Not Created ⚠️

The Docker infrastructure for the frontend is now ready, but the actual React application needs to be created.

---

## What's Complete

✅ **docker/frontend/Dockerfile** - Multi-stage build for dev and production
✅ **docker/frontend/nginx.conf** - Production nginx configuration
✅ **docker-compose.yml** - Frontend service configured
✅ **docker-compose.dev.yml** - Development overrides with hot-reload

---

## What's Missing

The `frontend/` directory currently only contains HTML files:
- `E2I_Causal_Dashboard_V2.html`
- `E2I_Causal_Dashboard_V3.html`

**Required for Docker to work:**
```
frontend/
├── package.json           # ⚠️ MISSING
├── package-lock.json      # ⚠️ MISSING
├── index.html            # ⚠️ MISSING
├── vite.config.js        # ⚠️ MISSING
├── src/                  # ⚠️ MISSING
│   ├── main.jsx
│   ├── App.jsx
│   └── ...
└── public/               # ⚠️ MISSING
    └── ...
```

---

## Setup Options

### Option 1: Create New Vite + React App (Recommended)

```bash
# Navigate to project root
cd /mnt/c/Users/nunezes1/Downloads/Projects/e2i_causal_analytics

# Backup existing HTML files
mkdir -p frontend_backup
mv frontend/*.html frontend_backup/

# Create new Vite React app
npm create vite@latest frontend -- --template react

# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Add additional dependencies for E2I
npm install @tanstack/react-query axios recharts lucide-react
npm install -D @vitejs/plugin-react

# Test locally
npm run dev
```

### Option 2: Convert Existing HTML to React

If you want to convert your existing HTML dashboards to React components:

1. Create Vite app (as above)
2. Extract components from HTML files
3. Convert HTML to JSX
4. Add state management and API integration

### Option 3: Use Static HTML (Quick Test)

For testing Docker setup only (not recommended for production):

```bash
cd frontend

# Create minimal package.json
cat > package.json << 'EOF'
{
  "name": "e2i-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.2.1",
    "vite": "^5.0.0"
  }
}
EOF

# Create minimal vite.config.js
cat > vite.config.js << 'EOF'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
  },
})
EOF

# Create minimal src structure
mkdir -p src public
cat > src/main.jsx << 'EOF'
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
EOF

cat > src/App.jsx << 'EOF'
import React from 'react'

function App() {
  return (
    <div>
      <h1>E2I Causal Analytics</h1>
      <p>Frontend placeholder - replace with actual app</p>
    </div>
  )
}

export default App
EOF

cat > index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>E2I Causal Analytics</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
EOF

# Install dependencies
npm install

# Test
npm run dev
```

---

## Recommended Vite Configuration

Once you create the app, use this `vite.config.js`:

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '/api'),
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
        },
      },
    },
  },
})
```

---

## Testing the Docker Setup

Once frontend app is created:

```bash
# Test development mode
cd docker
make dev

# Frontend should be available at:
# http://localhost:3001 (Vite dev server)

# Test production build
make prod

# Frontend should be available at:
# http://localhost:3001 (Nginx serving static files)
```

---

## Environment Variables

The frontend Docker setup expects these environment variables (already configured in docker-compose files):

**Development:**
- `NODE_ENV=development`
- `VITE_API_URL=http://localhost:8000`
- `VITE_WS_URL=ws://localhost:8000`

**Production:**
- `VITE_API_URL=http://api:8000`
- `VITE_WS_URL=ws://api:8000`

---

## Next Steps

1. **Choose setup option** (Option 1 recommended)
2. **Create React app** in `frontend/` directory
3. **Test locally** with `npm run dev`
4. **Test Docker** with `make dev`
5. **Develop your dashboard** using the existing HTML as reference

---

## Converting HTML Dashboards to React

Your existing HTML files can serve as a blueprint. Key components to extract:

- Dashboard layout
- Chart components (convert to Recharts/D3)
- Data fetching logic (convert to React Query + axios)
- State management (React Context or Zustand)
- Styling (Tailwind CSS or styled-components)

---

## Questions?

If you need help with:
- React app architecture
- Component structure for 18-agent system
- State management strategy
- API integration patterns

Just ask!
