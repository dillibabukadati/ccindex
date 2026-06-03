from __future__ import annotations
import sys
import os
import subprocess
from pathlib import Path


def _get_ccindex_exe() -> str:
    import shutil
    exe = shutil.which("ccindex")
    return exe or "ccindex"


def register_daemon() -> None:
    if sys.platform == "darwin":
        _register_launchd()
    elif sys.platform.startswith("linux"):
        _register_systemd()
    elif sys.platform == "win32":
        _register_windows_task()
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")


def unregister_daemon() -> None:
    if sys.platform == "darwin":
        _unregister_launchd()
    elif sys.platform.startswith("linux"):
        _unregister_systemd()
    elif sys.platform == "win32":
        _unregister_windows_task()


def get_daemon_status() -> str:
    if sys.platform == "darwin":
        result = subprocess.run(
            ["launchctl", "list", "com.ccindex.daemon"],
            capture_output=True, text=True
        )
        return "running" if result.returncode == 0 else "stopped"
    elif sys.platform.startswith("linux"):
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "ccindex"],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    return "unknown"


def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / "com.ccindex.daemon.plist"


def _register_launchd():
    exe = _get_ccindex_exe()
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ccindex.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>{exe}</string>
        <string>index</string>
    </array>
    <key>WatchPaths</key>
    <array>
        <string>{Path.cwd()}</string>
    </array>
    <key>ThrottleInterval</key>
    <integer>2</integer>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{Path.home()}/.ccindex/daemon.log</string>
    <key>StandardErrorPath</key>
    <string>{Path.home()}/.ccindex/daemon.log</string>
</dict>
</plist>"""
    p = _plist_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(plist)
    subprocess.run(["launchctl", "load", str(p)], check=True)


def _unregister_launchd():
    p = _plist_path()
    if p.exists():
        subprocess.run(["launchctl", "unload", str(p)], check=False)
        p.unlink()


def _systemd_unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / "ccindex.service"


def _register_systemd():
    exe = _get_ccindex_exe()
    unit = f"""[Unit]
Description=ccindex file watcher
After=default.target

[Service]
ExecStart={exe} index
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""
    p = _systemd_unit_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(unit)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", "ccindex"], check=True)


def _unregister_systemd():
    subprocess.run(["systemctl", "--user", "disable", "--now", "ccindex"], check=False)
    p = _systemd_unit_path()
    if p.exists():
        p.unlink()


def _register_windows_task():
    exe = _get_ccindex_exe()
    subprocess.run([
        "schtasks", "/create", "/tn", "ccindex_daemon",
        "/tr", f'"{exe}" index',
        "/sc", "onlogon", "/f",
    ], check=True)


def _unregister_windows_task():
    subprocess.run(
        ["schtasks", "/delete", "/tn", "ccindex_daemon", "/f"],
        check=False
    )
