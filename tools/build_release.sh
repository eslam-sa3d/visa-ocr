#!/bin/sh
# Build the production image and the complete release folder in one go, on a machine with Docker and internet access.
#   sh tools/build_release.sh <build id>          e.g.  sh tools/build_release.sh 1ea376e
# The build id is the source commit the folder was copied from; the image is tagged visa-ocr:<VERSION>-<build id>.
# Steps: pin the base image by digest, build with every package checked by hash, prove the image offline in test mode and in
# the production set-up, save it, and write release/ with the production pack, docs, tools, build record and checksums.
# It stops at the first failure and leaves release/ untouched until both proofs have passed.
set -eu
BUILD="${1:?build id (the source commit), e.g. 1ea376e}"
cd "$(dirname "$0")/.."
VERSION="$(cat VERSION)"; TAG="$VERSION-$BUILD"; IMAGE="visa-ocr:$TAG"
SAMPLE="scenarios/images/passport_deu__clean.png"
BASE="python:3.12-slim"; PLATFORM="linux/amd64"

echo "== 1/6 base image, pinned by digest"
docker pull --quiet --platform "$PLATFORM" "$BASE" >/dev/null
DIGEST="$(docker image inspect "$BASE" --format '{{index .RepoDigests 0}}' 2>/dev/null || true)"
case "$DIGEST" in *@sha256:*) ;; *)     # some Docker set-ups do not list a digest for an image of another platform
  DIGEST="python@$(docker buildx imagetools inspect "$BASE" --format '{{.Manifest.Digest}}' 2>/dev/null || true)";; esac
case "$DIGEST" in *@sha256:*) ;; *) echo "could not read the digest of $BASE" >&2; exit 1;; esac
echo "   $DIGEST"

echo "== 2/6 build $IMAGE (packages installed by hash)"
docker build --quiet --platform "$PLATFORM" --build-arg BASE_IMAGE="$DIGEST" --build-arg LOCKED=1 -t "$IMAGE" . >/dev/null

echo "== 3/6 offline proof (test mode, no network)"
OFFLINE="$(sh tools/offline_smoke.sh "$IMAGE" "$SAMPLE" 2>&1)" || { echo "$OFFLINE"; exit 1; }
echo "$OFFLINE" | tail -2

echo "== 4/6 production proof (production mode, no network, read-only image)"
PROD="$(sh tools/production_smoke.sh "$IMAGE" "$SAMPLE" 2>&1)" || { echo "$PROD"; exit 1; }
echo "$PROD" | tail -2

echo "== 5/6 save the image"
OUT="release.new"; rm -rf "$OUT"; mkdir -p "$OUT/production"
docker save "$IMAGE" | gzip > "$OUT/visa-ocr-$TAG-amd64.tar.gz"
gzip -t "$OUT/visa-ocr-$TAG-amd64.tar.gz"

echo "== 6/6 release folder"
cp deploy/production/visa-ocr.env deploy/production/docker-compose.yml deploy/production/make_keys.py "$OUT/production/"
for f in README.txt DEPLOYMENT.txt; do sed -e "s/@TAG@/$TAG/g" -e "s/@VERSION@/$VERSION/g" "deploy/release/$f" > "$OUT/$f"; done
sed -e "s/<commit>/$BUILD/g" docs/GO_LIVE_RECORD.md > "$OUT/GO_LIVE_RECORD.md"
tar czf "$OUT/docs.tar.gz" --exclude='.DS_Store' docs
find app -type f \( -name '*.py' -o -path 'app/static/*' \) -not -path '*/__pycache__/*' > "$OUT/.list"
ls config/* tools/system_test.py tools/perf_test.py tools/load_test.py tools/suggest_deadline.py tools/accuracy_matrix.py \
   tools/accuracy_real.py tools/capacity_test.py tools/read_folder.py tools/make_scenarios.py tools/make_checksums.py tools/specimens.py \
   requirements-lite.txt requirements-dev.txt VERSION 2>/dev/null >> "$OUT/.list"
tar czf "$OUT/tools.tar.gz" -T "$OUT/.list"; rm -f "$OUT/.list"
{
  echo "visa-ocr $VERSION, build $BUILD"
  echo "built:        $(date -u '+%Y-%m-%d %H:%M UTC')"
  echo "image:        $IMAGE"
  echo "image id:     $(docker image inspect "$IMAGE" --format '{{.Id}}')"
  echo "platform:     $PLATFORM"
  echo "base image:   $DIGEST"
  echo "packages:     installed from requirements.lock, every package checked by SHA-256 (LOCKED=1)"
  echo "offline proof:     $(echo "$OFFLINE" | tail -1)"
  echo "production proof:  $(echo "$PROD" | tail -1)"
  echo "                   $(echo "$PROD" | grep '^status:' | tail -1)"
} > "$OUT/BUILD_INFO.txt"
( cd "$OUT" && { if command -v sha256sum >/dev/null 2>&1; then sha256sum visa-ocr-*.tar.gz docs.tar.gz tools.tar.gz *.txt *.md production/*; \
    else shasum -a 256 visa-ocr-*.tar.gz docs.tar.gz tools.tar.gz *.txt *.md production/*; fi; } > ../SHA256SUMS.tmp && mv ../SHA256SUMS.tmp SHA256SUMS.txt )
rm -rf release.previous; [ -d release ] && mv release release.previous; mv "$OUT" release
echo
cat release/BUILD_INFO.txt
echo
SUMS_SHA="$( (command -v sha256sum >/dev/null 2>&1 && sha256sum release/SHA256SUMS.txt || shasum -a 256 release/SHA256SUMS.txt) | cut -d' ' -f1)"
# The checksum file travels with the release, so it proves nothing about the transfer on its own. Its own SHA-256 and the
# image id are recorded here, outside the folder: send them to the server team by a different route (e-mail, ticket, phone)
# from the release folder itself, and they compare before loading (DEPLOYMENT.txt step 1).
{ echo "$(date -u +%Y-%m-%dT%H:%MZ)  visa-ocr $TAG  SHA256SUMS.txt=$SUMS_SHA  image=$(docker image inspect "$IMAGE" --format '{{.Id}}')"; } >> "$HOME/visa-ocr-release-records.txt"
echo "SHA-256 of release/SHA256SUMS.txt: $SUMS_SHA"
echo "Send that value and the image id to the server team by a different route from the release folder (kept in ~/visa-ocr-release-records.txt)."
echo "DONE: release/ is ready to hand over. The folder from before is in release.previous/ (delete it when you are happy)."
