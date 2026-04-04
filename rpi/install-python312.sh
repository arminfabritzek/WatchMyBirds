#!/bin/bash
set -euo pipefail

PYTHON312_VERSION="${PYTHON312_VERSION:-3.12.12}"
PYTHON312_PREFIX="${PYTHON312_PREFIX:-/usr/local}"
PYTHON312_ARCHIVE="Python-${PYTHON312_VERSION}.tar.xz"
PYTHON312_ARCHIVE_URL="https://www.python.org/ftp/python/${PYTHON312_VERSION}/${PYTHON312_ARCHIVE}"
PYTHON312_SIGNATURE_URL="${PYTHON312_ARCHIVE_URL}.asc"
PYTHON_RELEASE_SIGNING_KEY="${PYTHON_RELEASE_SIGNING_KEY:-A821E680E5FA6305}"
BUILD_DIR="$(mktemp -d /tmp/python312-build-XXXXXX)"
GNUPGHOME="${BUILD_DIR}/gnupg"

cleanup() {
    rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

if command -v python3.12 >/dev/null 2>&1; then
    python3.12 --version
    exit 0
fi

APT_OPTIONS="-y -o Dpkg::Options::=--force-confnew -o Dpkg::Options::=--force-confdef"
PYTHON_BUILD_PACKAGES=(
    build-essential
    dirmngr
    gnupg
    libbz2-dev
    libffi-dev
    libgdbm-dev
    liblzma-dev
    libncursesw5-dev
    libnss3-dev
    libreadline-dev
    libsqlite3-dev
    libssl-dev
    tk-dev
    uuid-dev
    xz-utils
    zlib1g-dev
)

apt-get update
apt-get $APT_OPTIONS install \
    curl \
    "${PYTHON_BUILD_PACKAGES[@]}"

cd "$BUILD_DIR"
mkdir -p "$GNUPGHOME"
chmod 700 "$GNUPGHOME"

curl -fsSL "$PYTHON312_ARCHIVE_URL" -o "$PYTHON312_ARCHIVE"
curl -fsSL "$PYTHON312_SIGNATURE_URL" -o "${PYTHON312_ARCHIVE}.asc"

gpg --batch --keyserver hkps://keys.openpgp.org --recv-keys "$PYTHON_RELEASE_SIGNING_KEY" \
    || gpg --batch --keyserver hkp://keyserver.ubuntu.com --recv-keys "$PYTHON_RELEASE_SIGNING_KEY"
gpg --batch --verify "${PYTHON312_ARCHIVE}.asc" "$PYTHON312_ARCHIVE"

tar -xJf "$PYTHON312_ARCHIVE"
cd "Python-${PYTHON312_VERSION}"

./configure \
    --prefix="$PYTHON312_PREFIX" \
    --enable-shared \
    --with-ensurepip=install
make -j"$(nproc)"
make altinstall

echo "${PYTHON312_PREFIX}/lib" >/etc/ld.so.conf.d/python312.conf
ldconfig

apt-get $APT_OPTIONS purge --auto-remove "${PYTHON_BUILD_PACKAGES[@]}"

"${PYTHON312_PREFIX}/bin/python3.12" --version
