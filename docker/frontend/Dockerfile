# =============================================================================
# E2I Causal Analytics - Frontend Multi-Stage Dockerfile
# =============================================================================
# Stages:
#   1. base         - Node.js base image with dependencies
#   2. development  - Vite dev server with hot-reloading
#   3. builder      - Production build stage (tsc + vite build)
#   4. production   - Nginx serving the static build
#
# Build targets:
#   docker build --target development -t e2i-frontend:dev .
#   docker build --target production  -t e2i-frontend:prod .
#
# #528-B: the production (nginx) stage was dropped in aa58e71c when the frontend
# ran Vite-dev via bind mount in prod (dev = prod). #528-A flipped the app services
# to the prod target; #528-B restores this nginx image so the frontend serves a real
# production build behind nginx (base docker-compose `frontend` already targets
# `production` on 3002:80) instead of the Vite dev server.
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

# -----------------------------------------------------------------------------
# Stage 3: Builder (Production Build)
# -----------------------------------------------------------------------------
FROM base AS builder

# Install all dependencies including dev (typescript, vite, etc. live in
# devDependencies and are needed by `npm run build` -> `tsc -b && vite build`).
# NOTE: we deliberately do NOT set NODE_ENV=production here — npm 7+ honors
# NODE_ENV=production by skipping devDependencies even when --include=dev is
# absent, so combining the two would silently break tsc availability.
RUN npm ci --include=dev && \
    npm cache clean --force

# Copy application source
COPY frontend/ .

# Build the application (tsc emits to dist/, vite bundles)
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 4: Production
# -----------------------------------------------------------------------------
FROM nginx:alpine AS production

# Install curl for health checks
RUN apk add --no-cache curl

# Copy custom nginx configuration
COPY docker/frontend/nginx.conf /etc/nginx/nginx.conf

# Copy built files from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Create health check endpoint
RUN echo '<!DOCTYPE html><html><body>OK</body></html>' > /usr/share/nginx/html/health

# Create non-root user for nginx
RUN addgroup -g 1000 e2i && \
    adduser -D -u 1000 -G e2i e2i && \
    chown -R e2i:e2i /usr/share/nginx/html && \
    chown -R e2i:e2i /var/cache/nginx && \
    chown -R e2i:e2i /var/log/nginx && \
    touch /var/run/nginx.pid && \
    chown -R e2i:e2i /var/run/nginx.pid

# Switch to non-root user
USER e2i

# Expose HTTP port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
