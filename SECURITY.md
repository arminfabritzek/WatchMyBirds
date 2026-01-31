# Security Policy & Appliance Hardening

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.x     | :white_check_mark: |

Only the latest release receives security updates. We recommend always running the most recent version.

## Reporting a Vulnerability

We take the security of this project seriously. If you discover a security vulnerability, please report it privately.

**Do not open public issues for security vulnerabilities.**

**How to report:**
1. **GitHub Security Advisories (Preferred):** Use the "Report a vulnerability" button in the Security tab of this repository.
2. **Email:** Contact the repository maintainer directly via their GitHub profile.

**Response Commitment:** We aim to acknowledge reports within 48 hours and provide a fix or mitigation plan within 7 days for critical issues.

---

## Raspberry Pi Appliance Security
The WatchMyBirds Raspberry Pi image is designed as a secure-by-default appliance. It implements strict system hardening measures to ensure suitability for exposed environments.

### 1. User & Authentication
- **No Default User:** The standard `pi` user is completely removed.
- **Dedicated Service User:** The application runs as a dedicated system user `watchmybirds` with:
  - No login shell (`/usr/sbin/nologin`).
  - No home directory suitable for interactive use.
  - Minimal group privileges (`video`, `gpio`, `plugdev`).
- **Web UI Authentication:** 
  - The Web Interface is protected by a password (`EDIT_PASSWORD`) loaded from `/etc/app/app.env` or `settings.yaml`.
  - Default is empty unless configured; an empty password currently allows login and should be avoided on real networks.
  - Flask sessions use `FLASK_SECRET_KEY`; if unset, a static dev key is used and should be overridden in production.
- **Root Locked:** The `root` account is locked (`passwd -l root`) and has no password access.
- **No Interactive Access:** There are no interactive users enabled by default.
- **Dev Images (Non-Production):** `build-dev` images enable SSH and leave the `admin` user unlocked with passwordless sudo for convenience.

### 2. Environment Differences (Prod vs Dev)
| Feature | Production (`app.service`) | Development (`app-dev.service`) |
| :--- | :--- | :--- |
| **Filesystem** | Read-Only (`ProtectSystem=strict`) | Writes allowed to `/opt/app` |
| **Home Access** | Blocked (`ProtectHome=yes`) | Allowed (`ProtectHome=yes` with exceptions or disabled) |
| **Privileges** | `NoNewPrivileges=true`, SUID blocked | `NoNewPrivileges=false`, relaxed for debugging |
| **Power Management** | Via Polkit/logind (no sudo) | Via Polkit/logind (no sudo) |
| **Admin User** | Locked | Unlocked, NOPASSWD sudo |
| **SSH** | Disabled | Enabled by default |

> [!WARNING]
> **Dev Image Risk:** The development image is significantly less secure than the production image. It is designed for rapid iteration and debugging, not for exposure to untrusted networks. Never deploy a Dev image in a production environment.

### 2. Network Security
- **SSH Disabled by Default:** SSH is disabled on the image. It must be explicitly enabled by the user.
- **SSH Hardening (When Enabled):**
  - `PermitRootLogin no`
  - `PasswordAuthentication no` (Public Key Authentication ONLY)
  - `X11Forwarding` and `AgentForwarding` disabled.
- **Unique Host Keys:** SSH host keys are deleted during the build and regenerated when SSH is enabled (via `/boot/firmware/ssh` on first boot or later).
- **Firewall (UFW):**
  - **Default Policy:** Deny Incoming, Allow Outgoing.
  - **Web Interface:** Port 8050/tcp allowed.
  - **AP Services:** DNS/DHCP restricted strictly to the `wlan0` interface.
  - **Enforcement:** UFW is configured and enabled by the first-boot script.
- **Fail2Ban:** Not installed by default in the appliance image.

### 3. System Isolation & Hardening
- **Systemd Sandboxing:** The main application service (`app.service`) runs with maximum security directives:
    ```ini
    ProtectSystem=strict          # Read-only filesystem view (CRITICAL)
    ReadWritePaths=/var/lib/watchmybirds /var/log/app
    WorkingDirectory=/var/lib/watchmybirds
    PrivateTmp=true               # Isolated /tmp
    NoNewPrivileges=true          # SUID binaries blocked (sudo cannot escalate)
    ProtectHome=yes               # No access to /home
    ProtectKernelTunables=true    # No sysctl modifications
    ProtectKernelModules=true     # No module loading
    ProtectKernelLogs=true        # No kernel log access
    ProtectControlGroups=true
    ProtectClock=true
    RestrictNamespaces=true       # Container escape prevention
    MemoryDenyWriteExecute=true   # Prevent memory modification (W^X)
    LockPersonality=true          # Lock kernel execution domain
    RestrictRealtime=true
    RestrictSUIDSGID=true         # Block SUID/SGID bit execution
    ```
  - **Dev Mode:** `app-dev.service` relaxes sandboxing (`NoNewPrivileges=false`, `RestrictSUIDSGID=false`) to allow rsync-based updates and debugging.

- **Power Management (Polkit/logind):**
  - Reboot/Shutdown from the Web UI uses `systemctl` via DBus, not `sudo`.
  - A Polkit rule (`/etc/polkit-1/rules.d/10-watchmybirds-power.rules`) grants the `watchmybirds` user permission to call `org.freedesktop.login1.reboot` and `power-off`.
  - This design is compatible with `NoNewPrivileges=true` and avoids SUID-based privilege escalation.
- **Minimal Attack Surface:**
  - **Headless OpenCV:** Uses `opencv-python-headless` to eliminate dependencies on X11/GL libraries, significantly reducing the installed package footprint.
- **Filesystem Layout:**
  - **Code (`/opt/app`)**: Read-only for the application.
  - **Data (`/var/lib/watchmybirds`)**: Writable storage for database and images.
  - **Logs (`/var/log/app`)**: Writable app log directory.

### 4. Updates & Hygiene
- **Unattended Upgrades:** Package is installed; activation relies on OS defaults or user configuration.
- **Dependency Audit:** `pip-audit` is not executed in the current CI workflows.
- **Config Permissions:** If `/etc/app/app.env` is used, it should be `chmod 600` (not enforced by hardening scripts).
- **Log Hygiene:** 
  - Boot logs (`first-boot.log`) are rotated.
  - Diagnostic logs (`debuglogs/`) on the boot partition are automatically cleaned up after 48h to prevent Denial-of-Service via disk filling.
- **Build Hygiene:**
  - **Golden Image Pattern:** The OS base is built manually and rarely ("Golden Image"). Releases are injected into this trusted base, preventing accidental drift.
  - Bash history is wiped for all users.
  - Secrets and credentials are never baked into the image.

### 5. Wireless Security
- **WiFi Country (Regulatory):** WiFi country defaults to `DE` for regulatory compliance. Operators are responsible for adjusting the regulatory domain when deploying the device outside Germany.
- **AP Mode:** 
  - WPA2-protected Access Point for initial setup (SSID `WatchMyBirds-XXXX`).
  - The AP password is currently static (`watchmybirds`) via template configuration.
  - The WiFi watchdog may re-enable AP mode automatically on WiFi failure; this expands the management surface and should be considered in threat models.

---

## Developer Notes
- **Build System:** The image is built via GitHub Actions (`build-golden.yml`) using QEMU-managed Chroot execution.
- **Verification:** Security properties are audited against the `rpi/harden.sh` script and `rpi/first-boot/first-boot.sh` logic.
