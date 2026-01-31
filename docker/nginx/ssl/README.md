# SSL Certificates

## Production (Let's Encrypt — Active)

SSL is managed by **certbot** with automatic nginx integration on the droplet.

**Domain**: `eznomics.site`

```bash
# Certificates are at:
/etc/letsencrypt/live/eznomics.site/fullchain.pem
/etc/letsencrypt/live/eznomics.site/privkey.pem

# Certbot auto-modifies /etc/nginx/sites-available/e2i-app
# Auto-renewal is handled by systemd timer (certbot.timer)

# Verify auto-renewal works:
sudo certbot renew --dry-run

# Manual renewal (if needed):
sudo certbot renew
```

> This directory (`docker/nginx/ssl/`) is for the **Docker-based** nginx config
> (`nginx.secure.conf`). The production droplet uses certbot-managed certs directly
> in `/etc/letsencrypt/`.

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
