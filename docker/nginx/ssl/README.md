# SSL Certificates Directory

This directory should contain SSL certificates for HTTPS in production.

## For Production (Let's Encrypt)

Use Certbot to generate free SSL certificates:

```bash
# On your Digital Ocean droplet
sudo apt-get update
sudo apt-get install certbot

# Generate certificate (replace with your domain)
sudo certbot certonly --standalone -d yourdomain.com -d www.yourdomain.com

# Certificates will be placed in /etc/letsencrypt/live/yourdomain.com/
# Create symlinks:
sudo ln -s /etc/letsencrypt/live/yourdomain.com/fullchain.pem /path/to/project/docker/nginx/ssl/cert.pem
sudo ln -s /etc/letsencrypt/live/yourdomain.com/privkey.pem /path/to/project/docker/nginx/ssl/key.pem

# Auto-renewal (certbot creates this automatically)
sudo certbot renew --dry-run
```

## For Development (Self-Signed)

Generate self-signed certificates for local testing:

```bash
# Navigate to this directory
cd docker/nginx/ssl

# Generate self-signed certificate (valid for 365 days)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout key.pem \
  -out cert.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# Set proper permissions
chmod 600 key.pem
chmod 644 cert.pem
```

## Files Required

- `cert.pem` - SSL certificate (or fullchain.pem from Let's Encrypt)
- `key.pem` - Private key (or privkey.pem from Let's Encrypt)

## Security Notes

⚠️ **IMPORTANT**:
- Never commit SSL certificates to version control
- Keep private keys secure (chmod 600)
- Rotate certificates before expiration
- Use strong encryption (RSA 2048+ or ECC)

## Nginx Configuration

The nginx.conf file references these certificates:

```nginx
ssl_certificate /etc/nginx/ssl/cert.pem;
ssl_certificate_key /etc/nginx/ssl/key.pem;
```

These paths are inside the Docker container, mapped from this directory via docker-compose.yml:

```yaml
volumes:
  - ./docker/nginx/ssl:/etc/nginx/ssl:ro
```
