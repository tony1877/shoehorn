#!/usr/bin/env bash
# .deb and .rpm from the portable tarball (docs/LINUX-PLAN.md, Packaging 3): the tarball's tree goes to /opt/shoehorn,
# /usr/bin/shoehorn links to it, the systemd user units go to /usr/lib/systemd/user. Built in debian / rockylinux
# containers with the distributions' own tools (dpkg-deb, rpmbuild).
#
#   scripts/package-distro.sh dist/shoehorn-0.1.0-linux-x86_64-cuda.tar.gz   # -> dist/*.deb, dist/*.rpm
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
docker=${DOCKER:-docker}
tgz=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
name=$(basename "$tgz" .tar.gz)              # shoehorn-0.1.0-linux-x86_64-cuda
ver=$(echo "$name" | cut -d- -f2)
flavor=${name##*-}                           # cuda
desc="shoehorn: local inference engine for Qwen3.x with a split-precision residual ($flavor build)"

$docker run --rm -v "$tgz:/in.tgz:ro" -v "$root/dist:/dist" debian:12 bash -c "
set -e
d=/tmp/pkg; mkdir -p \$d/opt \$d/usr/bin \$d/usr/lib/systemd/user \$d/DEBIAN
tar -xzf /in.tgz -C \$d/opt && mv \$d/opt/$name \$d/opt/shoehorn
ln -s /opt/shoehorn/bin/shoehorn \$d/usr/bin/shoehorn
cp \$d/opt/shoehorn/systemd/*.service \$d/usr/lib/systemd/user/
sed -i 's|%h/.local/bin/shoehorn|/usr/bin/shoehorn|' \$d/usr/lib/systemd/user/shoehorn.service
cat > \$d/DEBIAN/control <<EOF
Package: shoehorn-$flavor
Version: $ver
Architecture: amd64
Maintainer: tony1877 <noreply@github.com>
Depends: libc6 (>= 2.34)
Recommends: curl
Description: $desc
EOF
dpkg-deb --root-owner-group --build \$d /dist/shoehorn-${flavor}_${ver}_amd64.deb
"

$docker run --rm -v "$tgz:/in.tgz:ro" -v "$root/dist:/dist" rockylinux:9 bash -c "
set -e
dnf install -y -q rpm-build >/dev/null
mkdir -p ~/rpmbuild/{SOURCES,SPECS}
cp /in.tgz ~/rpmbuild/SOURCES/$name.tar.gz
cat > ~/rpmbuild/SPECS/shoehorn.spec <<'EOF'
Name: shoehorn-$flavor
Version: $ver
Release: 1
Summary: $desc
License: see /opt/shoehorn
Source0: $name.tar.gz
AutoReqProv: no
Requires: glibc >= 2.34
%global debug_package %{nil}
# the bundled libraries (libgomp, CUDA runtime) would register build-id links that clash with the system's own copies
%global _build_id_links none
%global __strip /bin/true
%description
$desc
%prep
%setup -q -n $name
%install
mkdir -p %{buildroot}/opt/shoehorn %{buildroot}/usr/bin %{buildroot}/usr/lib/systemd/user
cp -a . %{buildroot}/opt/shoehorn/
ln -s /opt/shoehorn/bin/shoehorn %{buildroot}/usr/bin/shoehorn
cp systemd/*.service %{buildroot}/usr/lib/systemd/user/
sed -i 's|%%h/.local/bin/shoehorn|/usr/bin/shoehorn|' %{buildroot}/usr/lib/systemd/user/shoehorn.service
%files
/opt/shoehorn
/usr/bin/shoehorn
/usr/lib/systemd/user/shoehorn.service
/usr/lib/systemd/user/dsh.service
EOF
rpmbuild -bb --quiet ~/rpmbuild/SPECS/shoehorn.spec
cp ~/rpmbuild/RPMS/x86_64/*.rpm /dist/
"
ls -la "$root"/dist/*.deb "$root"/dist/*.rpm
