# Feature: Phase 0 - React Project Setup (Vite + TypeScript)

**Status**: Active
**Date Started**: 2025-12-21
**Parent Plan**: Frontend React Transformation
**Phase**: 0 - Project Setup
**Priority**: High
**Estimated Time**: 3-4 hours

## Overview

Set up the foundational React project using Vite, TypeScript, and modern development tooling. This is Phase 0 of the comprehensive Frontend React Transformation plan that will convert the static HTML dashboard (5,591 lines) into a modern, dynamic React application.

## User Story

As a developer
I want a modern React project scaffolded with Vite, TypeScript, and essential dependencies
So that I can begin converting the static dashboard_v3.html into a dynamic React application with proper tooling, testing, and development experience

## Problem Statement

The current frontend consists of:
- Static HTML files (dashboard_v3.html - 5,591 lines)
- Inline JavaScript with no modularity
- No component reusability
- No state management
- No TypeScript type safety
- No modern build tooling
- No hot module replacement (HMR)
- Difficult to maintain and extend

**We need:** A modern React foundation with Vite, TypeScript, ESLint, Prettier, and essential libraries to enable rapid, maintainable development.

## Goals

### Primary Goals
1. ✅ Create Vite + React + TypeScript project
2. ✅ Configure ESLint, Prettier, and TypeScript strict mode
3. ✅ Install and configure essential dependencies (React Router, TanStack Query, Zustand)
4. ✅ Set up Tailwind CSS + shadcn/ui
5. ✅ Configure path aliases (@/ for src/)
6. ✅ Create basic project structure (components, hooks, lib, types)
7. ✅ Verify development server runs successfully
8. ✅ Update Docker configuration for the new React app

### Success Criteria
- [x] Vite development server runs on port 5173
- [x] TypeScript compilation works without errors
- [x] ESLint and Prettier configured and working
- [x] Tailwind CSS styles apply correctly
- [x] Path aliases work (@/components, @/hooks, etc.)
- [x] Docker Compose can build and run the frontend service
- [x] Hot module replacement (HMR) works in development

## Implementation Plan

### Tasks Checklist

#### 1. Project Initialization (30 min)
- [x] Create `frontend/` directory in project root
- [x] Run `npm create vite@latest frontend -- --template react-ts`
- [x] Navigate to `frontend/` and install dependencies
- [x] Verify initial Vite + React + TypeScript setup works

#### 2. Development Tooling Configuration (45 min)
- [x] Configure ESLint with React and TypeScript rules
- [x] Configure Prettier with project code style
- [x] Set up `tsconfig.json` with strict mode and path aliases
- [x] Add `lint` and `format` scripts to `package.json`
- [x] Create `.prettierrc` and `.prettierignore`
- [x] Create `.eslintrc.cjs` and `.eslintignore`

#### 3. Install Core Dependencies (30 min)
- [x] Install React Router DOM (`react-router-dom`)
- [x] Install TanStack Query (`@tanstack/react-query`)
- [x] Install Zustand (`zustand`)
- [x] Install Axios (`axios`)
- [x] Install Tailwind CSS (`tailwindcss`, `postcss`, `autoprefixer`)
- [x] Install shadcn/ui CLI and initialize (`npx shadcn@latest init`)

#### 4. Tailwind CSS Setup (30 min)
- [x] Run `npx tailwindcss init -p`
- [x] Configure `tailwind.config.js` with content paths
- [x] Add Tailwind directives to `src/index.css`
- [x] Install shadcn/ui and configure `components.json`
- [x] Test Tailwind by adding utility classes to App.tsx

#### 5. Project Structure Creation (30 min)
- [x] Create directory structure:
  ```
  src/
  ├── components/
  │   ├── ui/          # shadcn/ui components
  │   ├── layout/      # Layout components (Header, Sidebar, etc.)
  │   └── dashboard/   # Dashboard-specific components
  ├── hooks/           # Custom React hooks
  ├── lib/             # Utilities and helpers
  ├── types/           # TypeScript type definitions
  ├── api/             # API client and endpoints
  ├── store/           # Zustand stores
  ├── pages/           # Page components
  └── styles/          # Global styles
  ```

#### 6. Path Aliases Configuration (15 min)
- [x] Update `tsconfig.json` with path aliases:
  ```json
  {
    "compilerOptions": {
      "baseUrl": ".",
      "paths": {
        "@/*": ["./src/*"],
        "@/components/*": ["./src/components/*"],
        "@/hooks/*": ["./src/hooks/*"],
        "@/lib/*": ["./src/lib/*"],
        "@/types/*": ["./src/types/*"],
        "@/api/*": ["./src/api/*"],
        "@/store/*": ["./src/store/*"],
        "@/pages/*": ["./src/pages/*"]
      }
    }
  }
  ```
- [x] Update `vite.config.ts` to resolve path aliases

#### 7. Docker Configuration Update (45 min)
- [x] Create `frontend/Dockerfile` for multi-stage build
- [x] Update `docker-compose.yml` to include frontend service
- [x] Configure Nginx for React SPA routing
- [x] Test Docker build and container startup
- [x] Verify HMR works with Docker volume mounts in development

#### 8. Basic App Setup (30 min)
- [x] Set up React Router in `main.tsx`
- [x] Configure TanStack Query Provider
- [x] Create basic layout component
- [x] Create placeholder dashboard page
- [x] Add 404 not found page

#### 9. Environment Configuration (15 min)
- [x] Create `.env.example` with API_URL and other environment variables
- [x] Create `.env.local` for local development
- [x] Configure Vite to use environment variables
- [x] Add `.env.local` to `.gitignore`

#### 10. Documentation (15 min)
- [x] Create `frontend/README.md` with setup instructions
- [x] Document npm scripts (dev, build, preview, lint, format)
- [x] Document directory structure
- [x] Add contribution guidelines for React components

### Total Estimated Time: 3-4 hours

## Technology Stack

### Core
- **React 18** - UI library
- **TypeScript 5** - Type safety
- **Vite 5** - Build tool and dev server

### Routing & State
- **React Router v6** - Client-side routing
- **TanStack Query v5** - Server state management
- **Zustand** - Client state management

### Styling
- **Tailwind CSS** - Utility-first CSS framework
- **shadcn/ui** - Accessible component library

### Development Tools
- **ESLint** - Code linting
- **Prettier** - Code formatting
- **TypeScript ESLint** - TypeScript-specific linting

### Build & Deployment
- **Docker** - Containerization
- **Nginx** - Production web server
- **Multi-stage builds** - Optimized Docker images

## File Changes

### Files to Create

1. **`frontend/package.json`** - Project dependencies and scripts
2. **`frontend/vite.config.ts`** - Vite configuration
3. **`frontend/tsconfig.json`** - TypeScript configuration
4. **`frontend/tailwind.config.js`** - Tailwind CSS configuration
5. **`frontend/.eslintrc.cjs`** - ESLint configuration
6. **`frontend/.prettierrc`** - Prettier configuration
7. **`frontend/Dockerfile`** - Docker build configuration
8. **`frontend/nginx.conf`** - Nginx configuration for SPA
9. **`frontend/README.md`** - Frontend documentation
10. **`frontend/.env.example`** - Environment variable template

### Files to Modify

1. **`docker-compose.yml`** - Add frontend service
2. **`.gitignore`** - Add frontend-specific ignores (node_modules, dist, .env.local)

## Dependencies to Install

```json
{
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.26.0",
    "@tanstack/react-query": "^5.51.0",
    "zustand": "^4.5.4",
    "axios": "^1.7.2"
  },
  "devDependencies": {
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "vite": "^5.3.4",
    "typescript": "^5.5.3",
    "eslint": "^8.57.0",
    "prettier": "^3.3.3",
    "tailwindcss": "^3.4.6",
    "postcss": "^8.4.39",
    "autoprefixer": "^10.4.19"
  }
}
```

## Validation Steps

### Development Validation
1. Run `npm run dev` - Server starts on http://localhost:5173
2. Verify HMR works by editing a component
3. Run `npm run lint` - No errors
4. Run `npm run format` - Code formatted correctly
5. Run `npm run build` - TypeScript compiles, Vite builds successfully
6. Run `npm run preview` - Production build serves correctly

### Docker Validation
1. Run `docker-compose build frontend` - Image builds successfully
2. Run `docker-compose up frontend` - Container starts and serves app
3. Access http://localhost:3000 - App loads correctly
4. Verify volume mounts work for development hot reload

### Code Quality Validation
1. No TypeScript errors (`tsc --noEmit`)
2. No ESLint errors (`npm run lint`)
3. Code is formatted (`npm run format`)
4. Path aliases work (import from `@/components`)
5. Tailwind CSS classes apply styles

## Docker Configuration

### Dockerfile (Multi-stage Build)

```dockerfile
# Stage 1: Build
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: Production
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### docker-compose.yml Update

```yaml
services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    environment:
      - VITE_API_URL=http://localhost:8000
    volumes:
      - ./frontend:/app  # For development HMR
      - /app/node_modules
    depends_on:
      - fastapi
```

## Risk Mitigation

### Risk 1: Dependency Conflicts
**Mitigation**: Use exact versions in package.json, commit package-lock.json

### Risk 2: Path Alias Issues
**Mitigation**: Carefully configure both tsconfig.json and vite.config.ts

### Risk 3: Docker Build Failures
**Mitigation**: Test builds locally before pushing, use multi-stage builds

### Risk 4: Tailwind CSS Not Working
**Mitigation**: Verify content paths in tailwind.config.js include all component files

## Next Steps (Phase 1)

After completing Phase 0, proceed to **Phase 1: Core Infrastructure**:
1. Create layout components (Header, Sidebar, MainContent)
2. Set up React Router with all routes
3. Create global filter provider
4. Set up API client with axios
5. Configure TanStack Query hooks for backend integration

## References

- [Vite Documentation](https://vitejs.dev/)
- [React TypeScript Cheatsheet](https://react-typescript-cheatsheet.netlify.app/)
- [TanStack Query Docs](https://tanstack.com/query/latest)
- [shadcn/ui Documentation](https://ui.shadcn.com/)
- [Parent Plan: Frontend React Transformation](../../../docs/FRONTEND_REACT_TRANSFORMATION_PLAN.md)

## Notes

- This phase focuses solely on setup and configuration
- No dashboard conversion happens in Phase 0
- All configurations are production-ready from the start
- TypeScript strict mode enabled for maximum type safety
- Path aliases make imports cleaner and easier to refactor

---

**Status**: Active - Ready to implement
**Blocked By**: None
**Blocks**: Phase 1 (Core Infrastructure)
