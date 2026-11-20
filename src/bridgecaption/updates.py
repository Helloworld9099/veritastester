"""Signed update manifest and streaming package download.

The signing public key is baked into official builds as BRIDGECAPTION_UPDATE_PUBLIC_KEY.
No unsigned executable may be installed.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ._update_build import BUILD_VERSION, UPDATE_PUBLIC_KEY

MAX_MANIFEST = 64 * 1024
MAX_PACKAGE = 1024 * 1024 * 1024
DEFAULT_FEED = "https://github.com/Helloworld9099/veritastester/releases/latest/download/manifest.json"


@dataclass(frozen=True)
class Update:
    version: str
    url: str
    sha256: str
    size: int
    platform: str


def current_version() -> str:
    if BUILD_VERSION:
        return BUILD_VERSION
    try:
        return version("bridgecaption")
    except PackageNotFoundError:
        return "0.1.0"


def version_tuple(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"v?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", value)
    if not match:
        raise ValueError("Invalid semantic version")
    return tuple(map(int, match.groups()))


def _https(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Update URLs must be HTTPS without embedded credentials")
    return url


class HTTPSOnlyRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, code, msg, headers, _https(newurl))


def urlopen(request, *, timeout):
    # Do not allow urllib to follow an HTTPS -> HTTP redirect.
    return build_opener(HTTPSOnlyRedirect()).open(request, timeout=timeout)


def platform_key(system=None, machine=None):
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    return f"{system}-{machine}"


PACKAGE_SUFFIXES = {"windows": ".exe", "darwin": ".zip", "linux": ".AppImage"}


def _public_key():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    value = UPDATE_PUBLIC_KEY.strip()
    if not value:
        raise RuntimeError("This build has no trusted update signing key")
    return Ed25519PublicKey.from_public_bytes(base64.b64decode(value, validate=True))


def verify_manifest(document: dict) -> dict:
    """Verify signature over canonical payload, not arbitrary remote instructions."""
    if not isinstance(document, dict) or set(document) != {"payload", "signature"}:
        raise ValueError("Malformed signed manifest")
    payload = document["payload"]
    if not isinstance(payload, dict):
        raise ValueError("Malformed update payload")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    _public_key().verify(base64.b64decode(document["signature"], validate=True), canonical)
    return payload


def find_update(current: str | None = None) -> Update | None:
    feed = _https(os.getenv("BRIDGECAPTION_UPDATE_ENDPOINT", DEFAULT_FEED))
    req = Request(feed, headers={"User-Agent": "BridgeCaption-Updater/1"})
    with urlopen(req, timeout=10) as response:
        raw = response.read(MAX_MANIFEST + 1)
    if len(raw) > MAX_MANIFEST:
        raise ValueError("Update manifest too large")
    payload = verify_manifest(json.loads(raw))
    latest = payload["version"]
    version_tuple(latest)
    if version_tuple(latest) <= version_tuple(current or current_version()):
        return None
    key = platform_key()
    artifacts = payload["artifacts"]
    artifact = artifacts.get(key)
    # Version 1 manifests used the unnormalized platform identifiers.
    if artifact is None:
        artifact = artifacts.get(f"{platform.system().lower()}-{platform.machine().lower()}")
    if artifact is None:
        raise RuntimeError(f"No signed update for {key}")
    url = _https(artifact["url"])
    digest = artifact["sha256"]
    size = artifact["size"]
    if not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest):
        raise ValueError("Invalid package checksum")
    if type(size) is not int or not 0 < size <= MAX_PACKAGE:
        raise ValueError("Invalid package size")
    return Update(latest.removeprefix("v"), url, digest, size, key)


def download_update(update: Update, progress, cancel=None) -> Path:
    """Download a verified package without running it. Caller owns returned temp path."""
    req = Request(_https(update.url), headers={"User-Agent": "BridgeCaption-Updater/1"})
    if update.platform != platform_key():
        raise ValueError("Package is not for this platform")
    suffix = PACKAGE_SUFFIXES.get(update.platform.split("-", 1)[0])
    if suffix is None:
        raise RuntimeError("This operating system has no supported update package")
    # A private directory prevents another process from writing a lookalike package
    # into a shared temp filename between verification and installer startup.
    temp_dir = Path(tempfile.mkdtemp(prefix="bridgecaption-update-"))
    fd, name = tempfile.mkstemp(dir=temp_dir, prefix="verified-", suffix=suffix)
    destination = Path(name)
    received = 0
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as output, urlopen(req, timeout=30) as response:
            while True:
                if cancel and cancel():
                    raise RuntimeError("Update cancelled")
                chunk = response.read(128 * 1024)
                if not chunk:
                    break
                received += len(chunk)
                if received > update.size:
                    raise ValueError("Package exceeds declared size")
                digest.update(chunk)
                output.write(chunk)
                progress(received, update.size)
        if received != update.size or digest.hexdigest() != update.sha256:
            raise ValueError("Package integrity verification failed")
        if cancel and cancel():
            raise RuntimeError("Update cancelled")
        return destination
    except BaseException:
        discard_update(destination)
        raise

def discard_update(package: Path):
    package.unlink(missing_ok=True)
    if package.parent.name.startswith("bridgecaption-update-"):
        try:
            package.parent.rmdir()
        except OSError:
            pass


def installed_app():
    """Return the actual target of an official writable install, never a source app."""
    if not getattr(sys, "frozen", False):
        raise RuntimeError("Install a packaged BridgeCaption desktop build to enable updates.")
    system = platform.system()
    executable = Path(sys.executable).resolve()
    if system == "Linux" and os.getenv("APPIMAGE"):
        target = Path(os.environ["APPIMAGE"]).resolve()
        if target.is_file() and os.access(target.parent, os.W_OK):
            return target
    if system == "Darwin" and len(executable.parents) >= 3:
        target = executable.parents[2]
        if (executable.parts[-3:-1] == ("Contents", "MacOS")
                and target.name == "BridgeCaption.app" and os.access(target.parent, os.W_OK)
                and "AppTranslocation" not in target.parts and target.parts[:2] != ("/", "Volumes")):
            return target
        raise RuntimeError("Move BridgeCaption.app to a writable Applications folder, then reopen it.")
    if system == "Windows":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\BridgeCaption\Desktop") as key:
                location, kind = winreg.QueryValueEx(key, "InstallPath")
            if kind == winreg.REG_SZ and Path(location, "BridgeCaption.exe").resolve() == executable:
                return executable
        except (OSError, TypeError):
            pass
        raise RuntimeError("Run the official BridgeCaption Windows installer first.")
    raise RuntimeError("Install BridgeCaption in a writable location to enable updates.")


_UNIX_SWAP = r"""
pid="$1"; source="$2"; destination="$3"; format="$4"; stage="$5"
cleanup() { rm -rf -- "$stage"; }
trap cleanup EXIT
count=0
while kill -0 "$pid" 2>/dev/null; do
  count=$((count + 1))
  [ "$count" -lt 150 ] || exit 1
  sleep 0.2
done
backup="$stage/previous"
if [ "$format" = linux ]; then
  # Keep the original path in place through the atomic file replacement.
  ln -- "$destination" "$backup" || exit 1
else
  mv -- "$destination" "$backup" || exit 1
fi
if ! mv -f -- "$source" "$destination"; then
  if [ "$format" = darwin ] && ! mv -- "$backup" "$destination"; then
    trap - EXIT    # Preserve the previous bundle if the rollback cannot complete.
  fi
  exit 1
fi
if [ "$format" = darwin ]; then
  if ! /usr/bin/open -n "$destination"; then
    if ! mv -- "$destination" "$stage/failed" || ! mv -- "$backup" "$destination"; then
      trap - EXIT
    fi
    exit 1
  fi
else
  "$destination" </dev/null >/dev/null 2>&1 &
fi
"""


def _native_environment():
    """Do not load bundled Tcl/Python libraries into native OS helpers."""
    environment = os.environ.copy()
    if "LD_LIBRARY_PATH_ORIG" in environment:
        environment["LD_LIBRARY_PATH"] = environment.pop("LD_LIBRARY_PATH_ORIG")
    elif getattr(sys, "frozen", False):
        environment.pop("LD_LIBRARY_PATH", None)
    # AppImage/PyInstaller markers belong to the old process, not its replacement.
    for key in ("APPIMAGE", "APPDIR", "ARGV0"):
        environment.pop(key, None)
    environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    return environment


def _install_windows(package, target, diagnostics=None):
    import ctypes
    buf = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetSystemDirectoryW(buf, len(buf))
    if not 0 < length < len(buf):
        raise OSError("Could not find the Windows system directory")
    powershell = Path(buf.value, "WindowsPowerShell", "v1.0", "powershell.exe")

    def literal(text):
        return "'" + str(text).replace("'", "''") + "'"

    # Launch the signed installer directly. Lock it to the registered install
    # directory, and don't join the process tree containing the relaunched app.
    log_path = diagnostics / "installer.log" if diagnostics else ""
    helper_log = diagnostics / "helper.log" if diagnostics else ""
    command = f"""
$ErrorActionPreference = 'Stop'
$setupPath = {literal(package)}
$installDir = {literal(target.parent)}
$setupLog = {literal(log_path)}
$helperLog = {literal(helper_log)}
function Write-HelperLog([string]$Message) {{
    if ($helperLog) {{ Add-Content -LiteralPath $helperLog -Value $Message }}
}}
try {{
    Write-HelperLog 'Waiting for BridgeCaption to exit'
    $p = Get-Process -Id {os.getpid()} -ErrorAction SilentlyContinue
    if ($p -and -not $p.WaitForExit(30000)) {{ throw 'BridgeCaption did not exit in time' }}
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $setupPath
    $info.Arguments = '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /RELAUNCH /DIR="' + $installDir + '"'
    if ($setupLog) {{ $info.Arguments += ' /LOG="' + $setupLog + '"' }}
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    Write-HelperLog 'Starting the installer in the registered directory'
    $setup = [System.Diagnostics.Process]::Start($info)
    if (-not $setup.WaitForExit(180000)) {{ throw 'Setup did not exit in time' }}
    Write-HelperLog ('Setup exit code: ' + $setup.ExitCode)
    exit $setup.ExitCode
}} catch {{
    Write-HelperLog ($_.ToString())
    exit 1
}} finally {{
    Remove-Item -LiteralPath $setupPath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath {literal(package.parent)} -ErrorAction SilentlyContinue
}}
"""
    # PyInstaller's DLL search override is inherited by Windows child processes
    # even if the environment block is replaced.
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle and not ctypes.windll.kernel32.SetDllDirectoryW(None):
        raise ctypes.WinError()
    try:
        subprocess.Popen(
            [str(powershell), "-NoProfile", "-NoLogo", "-NonInteractive", "-EncodedCommand",
             base64.b64encode(command.encode("utf-16le")).decode("ascii")],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            env=_native_environment(),
        )
    finally:
        if bundle:
            ctypes.windll.kernel32.SetDllDirectoryW(ctypes.c_wchar_p(bundle))


def install_update_on_exit(package: Path, *, diagnostics: Path | None = None) -> None:
    """Stage an already-verified artifact and start a native detached installer."""
    target = installed_app()
    system = platform.system().lower()
    if package.suffix != PACKAGE_SUFFIXES[system] or not package.is_file():
        raise ValueError("Unexpected update package format")
    if system == "windows":
        _install_windows(package, target, diagnostics)
        return

    # Stage under the installation's parent so both the rename and rollback are
    # on the same volume even if your temp directory lives on a separate disk.
    stage = Path(tempfile.mkdtemp(prefix=".bridgecaption-upgrade-", dir=target.parent))
    try:
        environment = _native_environment()
        if system == "darwin":
            subprocess.run(["/usr/bin/ditto", "-x", "-k", str(package), str(stage)],
                           check=True, capture_output=True, timeout=120, env=environment)
            staged = stage / "BridgeCaption.app"
            if staged.is_symlink() or not (staged / "Contents/MacOS/BridgeCaption").is_file():
                raise ValueError("Update archive does not contain BridgeCaption.app")
            subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(staged)],
                           check=True, capture_output=True, timeout=120, env=environment)
        else:
            staged = stage / "replacement.AppImage"
            shutil.copyfile(package, staged)
            staged.chmod(0o755)
        subprocess.Popen(
            ["/bin/sh", "-c", _UNIX_SWAP, "bridgecaption-update", str(os.getpid()),
             str(staged), str(target), system, str(stage)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, env=environment,
        )
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    discard_update(package)
