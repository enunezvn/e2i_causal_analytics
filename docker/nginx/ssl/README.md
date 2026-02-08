# SSL Certificates & Port 443 Configuration

## Production Architecture

The production droplet uses **sslh** to multiplex SSH and HTTPS on port 443,
allowing SSH access from networks that block port 22 (e.g., corporate firewalls).

```
                    ┌─────────────────────────────────────┐
                    │           Port 443 (sslh)           │
                    │         Protocol Multiplexer        │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
              SSH Traffic                   HTTPS Traffic
                    │                             │
                    ▼                             ▼
           ┌───────────────┐            ┌─────────────────┐
           │  sshd :22     │            │ nginx :4443     │
           │  (localhost)  │            │ (localhost)     │
           └───────────────┘            └─────────────────┘
```

**Droplet IP**: `138.197.4.36`
**Domain**: `eznomics.site`

## sslh Configuration

sslh listens on port 443 and routes traffic based on protocol detection:

```bash
# Config file: /etc/default/sslh
DAEMON_OPTS="--user sslh --listen 0.0.0.0:443 --ssh 127.0.0.1:22 --tls 127.0.0.1:4443"

# Service management:
sudo systemctl status sslh
sudo systemctl restart sslh

# View logs:
sudo journalctl -u sslh -f
```

## SSH Access

Connect via port 443 (bypasses corporate firewalls):

```bash
# Using the configured alias:
ssh droplet

# Or directly:
ssh -p 443 enunez@138.197.4.36

# Port 22 still works as fallback:
ssh -p 22 enunez@138.197.4.36
```

SSH config (`~/.ssh/config`):
```
Host droplet
    HostName 138.197.4.36
    User enunez
    Port 443
    IdentityFile ~/.ssh/replit
```

## SSL Certificates (Let's Encrypt)

SSL is managed by **certbot** with automatic nginx integration.

```bash
# Certificates are at:
/etc/letsencrypt/live/eznomics.site/fullchain.pem
/etc/letsencrypt/live/eznomics.site/privkey.pem

# Certbot auto-modifies /etc/nginx/sites-available/e2i-analytics
# Auto-renewal is handled by systemd timer (certbot.timer)

# Verify auto-renewal works:
sudo certbot renew --dry-run

# Manual renewal (if needed):
sudo certbot renew
```

## nginx Configuration

nginx listens on `127.0.0.1:4443` (internal only) since sslh handles port 443:

```bash
# Config file:
/etc/nginx/sites-enabled/e2i-analytics

# Key lines:
listen 127.0.0.1:4443 ssl;  # Internal port for sslh

# Test and reload:
sudo nginx -t && sudo systemctl reload nginx
```

> This directory (`docker/nginx/ssl/`) is for the **Docker-based** nginx config
> (`nginx.secure.conf`). The production droplet uses certbot-managed certs directly
> in `/etc/letsencrypt/`.

## Troubleshooting

### sslh not routing correctly

```bash
# Check what's listening on ports:
sudo ss -tlnp | grep -E ':443|:4443|:22'

# Expected output:
# LISTEN  127.0.0.1:4443  nginx    (internal HTTPS)
# LISTEN  0.0.0.0:443     sslh     (multiplexer)
# LISTEN  0.0.0.0:22      sshd     (SSH)

# Restart both services:
sudo systemctl restart sslh nginx
```

### SSH connection refused on port 443

```bash
# Verify sslh is running:
sudo systemctl status sslh

# Check sslh config:
cat /etc/default/sslh

# Reinstall if needed:
sudo apt-get install --reinstall sslh
```

### HTTPS not working

```bash
# Test nginx directly (from droplet):
curl -k https://127.0.0.1:4443/health

# Check nginx error logs:
sudo tail -f /var/log/nginx/e2i-app.error.log
```

---

## For Development (Self-Signed)

Generate self-signed certificates for local Docker testing:

```bash
cd docker/nginx/ssl

openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout key.pem \
  -out cert.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

chmod 600 key.pem
chmod 644 cert.pem
```

## Files

- `cert.pem` - SSL certificate (for Docker nginx only)
- `key.pem` - Private key (for Docker nginx only)

## Security Notes

- Never commit SSL certificates to version control
- Keep private keys secure (chmod 600)
- Let's Encrypt certificates auto-renew every 60-90 days
- Production certs live in `/etc/letsencrypt/`, not in this directory
