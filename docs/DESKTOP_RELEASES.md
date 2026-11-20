# Desktop packages and signed updates

The desktop PR workflow builds Windows x64 and macOS Intel/Apple Silicon packages
and exercises the installer against two actual frozen builds. It is intentionally
read-only and never creates a public release. Linux CI uploads a portable build;
Linux in-app installs still require an AppImage.

## Candidate builds

On the target OS, with the repository checked out:

    python -m pip install -e ".[test]" "pyinstaller>=6.14,<7"
    python scripts/build_desktop.py --version 0.2.0 --output release

Use Inno Setup 6 on Windows. On macOS, move BridgeCaption.app out of the archive
to a writable Applications folder before opening. Installing from source or using
a portable Windows folder won't enable in-app replacement.

## Publisher checklist

1. Generate and retain an Ed25519 release-signing private key in approved
   credential storage. Never commit it or pass it to the build matrix. Supply
   only its base64 raw public key via `BRIDGECAPTION_UPDATE_PUBLIC_KEY` during
   official builds; the packaged client never reads a runtime public key.
2. Distribute appropriately platform-signed packages. For macOS, use
   `BRIDGECAPTION_MACOS_SIGNING_IDENTITY`, sign with Developer ID, notarize,
   staple the app, and recreate the final ZIP after stapling. Sign the Windows
   installer with Authenticode. No updater private key is needed for these steps.
3. Produce one signed manifest from the final, immutable package bytes:

       python scripts/sign_update.py 0.2.0 \
         --artifact windows-x86_64 release/BridgeCaption-windows-x86_64-setup.exe https://github.com/Helloworld9099/veritastester/releases/download/v0.2.0/BridgeCaption-windows-x86_64-setup.exe \
         --artifact darwin-arm64 release/BridgeCaption-darwin-arm64.zip https://github.com/Helloworld9099/veritastester/releases/download/v0.2.0/BridgeCaption-darwin-arm64.zip \
         --artifact darwin-x86_64 release/BridgeCaption-darwin-x86_64.zip https://github.com/Helloworld9099/veritastester/releases/download/v0.2.0/BridgeCaption-darwin-x86_64.zip > manifest.json

   The signing command reads `BRIDGECAPTION_UPDATE_PRIVATE_KEY`. Add a
   `linux-x86_64` entry only when a verified official AppImage is available.
4. Upload the matching assets and manifest to the versioned GitHub release.
   Check the anonymous `releases/latest/download/manifest.json` endpoint and
   run the old signed installed app through “Check for updates” on each OS.
   Verify the new version launches and the prior preferences remain intact.

The CI test serves fixture bytes in memory with a disposable signing key. It
doesn't replace the final release check of the public feed, OS trust prompts,
real device audio, and signed installers.
