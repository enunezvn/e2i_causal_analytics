# =============================================================================
# E2I Causal Analytics - Frontend Multi-Stage Dockerfile
# =============================================================================
# Stages:
#   1. base         - Node.js base image with dependencies
#   2. development  - Vite dev server with hot-reloading
#
# Build targets:
#   docker build --target development -t e2i-frontend:dev .
#
# Production runs Vite-dev via bind mount on the single droplet
# (dev = prod). See MEMORY.md.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base Image
# -----------------------------------------------------------------------------
FROM node:20-alpine AS base

# Set working directory
WORKDIR /app

# Install dumb-init for proper signal handling
RUN apk add --no-cache dumb-init

# Set environment variables
ENV NODE_ENV=development \
    NPM_CONFIG_LOGLEVEL=warn

# Copy package files
COPY frontend/package*.json ./

# -----------------------------------------------------------------------------
# Stage 2: Development
# -----------------------------------------------------------------------------
FROM base AS development

# Install all dependencies (including dev dependencies)
RUN npm ci

# Copy application source
COPY frontend/ .

# Run as the built-in node user (UID 1000) instead of root.
# Vite dev server only needs read access to node_modules + bind-mount.
RUN chown -R node:node /app
USER node

# Expose Vite dev server port
EXPOSE 5173

# Health check for Vite dev server
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:5173 || exit 1

# Start Vite development server
ENTRYPOINT ["dumb-init", "--"]
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
