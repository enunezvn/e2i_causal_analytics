# =============================================================================
# E2I Causal Analytics - Frontend Multi-Stage Dockerfile
# =============================================================================
# Stages:
#   1. base         - Node.js base image with dependencies
#   2. development  - Vite dev server with hot-reloading
#   3. builder      - Production build stage
#   4. production   - Nginx serving static files
#
# Build targets:
#   docker build --target development -t e2i-frontend:dev .
#   docker build --target production -t e2i-frontend:prod .
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

# Set production environment
ENV NODE_ENV=production

# Install only production dependencies first (better caching)
RUN npm ci --only=production && \
    npm cache clean --force

# Install dev dependencies needed for build
RUN npm ci

# Copy application source
COPY frontend/ .

# Build the application
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
