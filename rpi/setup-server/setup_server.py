#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# rpi/setup-server/setup_server.py
# ------------------------------------------------------------------------------
# Minimal AP-only setup server (port 80) to collect WiFi credentials.
# Writes pending config for systemd path unit to process, then returns success.
# ------------------------------------------------------------------------------

import logging
import os

from flask import Flask, render_template, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wmb-setup")

PENDING_FILE = "/opt/app/data/pending_wifi.conf"
PENDING_PASSWORD_FILE = "/opt/app/data/pending_admin_password"
SSID_SCAN_FILE = "/opt/app/data/ssid_scan.txt"
MIN_PASSWORD_LENGTH = 8
DISALLOWED_PASSWORDS = {"watchmybirds", "SECRET_PASSWORD", "default_pass", ""}


def _write_pending_config(ssid: str, password: str, admin_password: str) -> None:
    safe_ssid = ssid.replace('"', '\\"')
    safe_pass = password.replace('"', '\\"')

    config_content = (
        "country=DE\n"
        "ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev\n"
        "update_config=1\n"
        "\n"
        "network={\n"
        f'    ssid="{safe_ssid}"\n'
        f'    psk="{safe_pass}"\n'
        "    key_mgmt=WPA-PSK\n"
        "}\n"
    )

    with open(PENDING_FILE, "w", encoding="utf-8") as handle:
        handle.write(config_content)

    os.chmod(PENDING_FILE, 0o600)

    with open(PENDING_PASSWORD_FILE, "w", encoding="utf-8") as handle:
        handle.write(admin_password)

    os.chmod(PENDING_PASSWORD_FILE, 0o600)
    os.sync()


def _load_ssids() -> list[str]:
    if not os.path.exists(SSID_SCAN_FILE):
        return []
    ssids: list[str] = []
    with open(SSID_SCAN_FILE, encoding="utf-8") as handle:
        for line in handle:
            ssid = line.strip()
            if ssid and ssid not in ssids:
                ssids.append(ssid)
    return ssids


def _create_app() -> Flask:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    template_folder = os.environ.get(
        "SETUP_TEMPLATE_DIR", os.path.join(base_dir, "templates")
    )
    app = Flask(__name__, template_folder=template_folder)

    @app.route("/", methods=["GET", "POST"])
    def setup_root():
        ssids = _load_ssids()
        if request.method == "POST":
            ssid = (request.form.get("ssid") or "").strip()
            password = request.form.get("password") or ""
            admin_password = request.form.get("admin_password") or ""
            admin_password_confirm = request.form.get("admin_password_confirm") or ""

            if not ssid or not password:
                return render_template(
                    "setup.html",
                    error="SSID and password are required.",
                    ssid=ssid,
                    ssids=ssids,
                )

            if len(admin_password.strip()) < MIN_PASSWORD_LENGTH:
                return render_template(
                    "setup.html",
                    error=f"Admin password must be at least {MIN_PASSWORD_LENGTH} characters long.",
                    ssid=ssid,
                    ssids=ssids,
                )

            if admin_password.strip() in DISALLOWED_PASSWORDS:
                return render_template(
                    "setup.html",
                    error="Please choose an admin password that is not a known default.",
                    ssid=ssid,
                    ssids=ssids,
                )

            if admin_password != admin_password_confirm:
                return render_template(
                    "setup.html",
                    error="Admin password confirmation does not match.",
                    ssid=ssid,
                    ssids=ssids,
                )

            try:
                _write_pending_config(ssid, password, admin_password.strip())
                logger.info("WiFi config and admin password saved to pending files.")
                return render_template("setup.html", success=True, ssids=ssids)
            except Exception as exc:
                logger.exception("Failed to write pending WiFi config.")
                return render_template(
                    "setup.html",
                    error=f"Error while saving: {exc}",
                    ssid=ssid,
                    ssids=ssids,
                )

        return render_template("setup.html", ssids=ssids)

    return app


def main() -> None:
    port = int(os.environ.get("SETUP_PORT", "80"))
    app = _create_app()
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
