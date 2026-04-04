# AIGUIBook with Podman

Podman is daemonless and rootless by default, which makes it more secure and
often easier to manage than Docker. Here's how to set it up.

## Prerequisites

```bash
# Install Podman
sudo apt install podman podman-compose  # Ubuntu/Debian
# OR
sudo dnf install podman podman-compose  # Fedora
# OR (Arch)
sudo pacman -S podman podman-compose
```

## NVIDIA GPU Support

Podman needs `nvidia-container-toolkit`:

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\
sed \"s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g\" | \\
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=podman
systemctl --user restart podman
```

## Build the Image

```bash
podman build -t aiguibook .
```

## Run the Container

### Option A: GUI Mode (default)

```bash
mkdir -p ./output ./voices ./config ./epub

podman run -d --name aiguibook \\
  --device nvidia.com/gpu=all \\
  -p 7860:7860 \\
  -v $(pwd)/config:/home/aiguibook/.aiguibook:Z \\
  -v $(pwd)/voices:/home/aiguibook/app/voices:Z \\
  -v $(pwd)/output:/home/aiguibook/app/output:Z \\
  -v $(pwd)/epub:/home/aiguibook/epubs:ro,Z \\
  -e LMSTUDIO_BASE_URL=http://10.0.2.2:1234 \\
  aiguibook:latest
```

> **Note**: `10.0.2.2` is Podman's default gateway to the host.
> For LM Studio on your host, this lets the container reach LM Studio at port 1234.

### Option B: CLI Mode (one-shot generation)

```bash
podman run --rm \\
  --device nvidia.com/gpu=all \\
  -v $(pwd)/config:/home/aiguibook/.aiguibook:Z \\
  -v $(pwd)/output:/home/aiguibook/app/output:Z \\
  -v $(pwd)/epub:/home/aiguibook/epubs:ro,Z \\
  aiguibook:latest \\
  python3.12 cli.py generate \\
    --input /home/aiguibook/epubs/my_book.epub \\
    --output /home/aiguibook/app/output/
```

## Useful Commands

```bash
# Check container status
podman ps

# View logs (GUI is running here)
podman logs -f aiguibook

# Stop the container
podman stop aiguibook

# Restart
podman start aiguibook

# Remove and rebuild
podman stop aiguibook && podman rm aiguibook
podman build -t aiguibook .
podman run -d --name aiguibook ...  # (see above)

# Run a shell inside the container for debugging
podman exec -it aiguibook bash

# Clean up unused images/cache
podman system prune
```

## Docker vs Podman: Which is Better?

| Feature | Docker | Podman |
|---------|--------|--------|
| Rootless | No (requires daemon) | Yes (rootless by default) |
| Daemon | Required | Not needed (fork/exec) |
| Docker Compose | docker-compose | podman-compose (compatible) |
| GPU Support | Mature | Works via nvidia-container-toolkit |
| Security | Good | Better (no daemon, no root) |
| Systemd integration | Manual | Native (podman generate systemd) |
| Resource usage | Higher (daemon) | Lower (no daemon) |

**Recommendation**: Podman is slightly better for a personal/home setup because:
- No always-running daemon
- Rootless = more secure
- Better systemd integration
- Drop-in compatible with docker-compose

But honestly, for your use case, both work fine. Pick whichever you already
have installed or are more comfortable with.

## LM Studio Integration from Container

Since LM Studio runs on your host machine, the container needs to reach it:

- **Docker**: Use `host.docker.internal` as the hostname
- **Podman**: Use `10.0.2.2` (default gateway) or `--network host`

If using `--network host`, simply add `--network host` to the run command
and change `LMSTUDIO_BASE_URL` to `http://localhost:1234`.
