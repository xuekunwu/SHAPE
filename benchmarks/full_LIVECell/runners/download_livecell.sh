#!/usr/bin/env bash
# Download LIVECell images.zip and the 24 single-cell COCO files into <root>/raw/ and record SOURCE.json.
# Resumable: complete files (size equal to the server's Content-Length) are kept; partial ones continue.
# Usage: bash download_livecell.sh <root>
set -eo pipefail

ROOT="$1"
if [ -z "$ROOT" ]; then
  echo "usage: bash download_livecell.sh <root>" >&2
  exit 2
fi
BASE="http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021"
RAW="$ROOT/raw"
IMAGES_ZIP_BYTES=1241715651
mkdir -p "$RAW"

FILES=("images.zip")
for t in a172 bt474 bv2 huh7 mcf7 shsy5y skbr3 skov3; do
  for s in train val test; do
    FILES+=("annotations/LIVECell_single_cells/$t/$s.json")
  done
done

remote_size() {
  curl -sSfI --retry 5 --retry-delay 10 "$BASE/$1" | tr -d '\r' \
    | awk 'tolower($1) == "content-length:" { size = $2 } END { print size }'
}

for rel in "${FILES[@]}"; do
  dest="$RAW/$rel"
  mkdir -p "$(dirname "$dest")"
  size=$(remote_size "$rel")
  if [ -z "$size" ]; then
    echo "no Content-Length for $rel" >&2
    exit 1
  fi
  if [ "$rel" = "images.zip" ] && [ "$size" != "$IMAGES_ZIP_BYTES" ]; then
    echo "images.zip is $size bytes on the server, expected $IMAGES_ZIP_BYTES" >&2
    exit 1
  fi
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$size" ]; then
    echo "HAVE $rel"
    continue
  fi
  rm -f "$dest"
  part="$dest.part"
  if [ -f "$part" ] && [ "$(stat -c %s "$part")" -gt "$size" ]; then
    rm -f "$part"
  fi
  if [ ! -f "$part" ] || [ "$(stat -c %s "$part")" != "$size" ]; then
    echo "GET $rel ($size bytes)"
    curl -fSL --retry 10 --retry-delay 15 --retry-all-errors -C - -o "$part" "$BASE/$rel"
  fi
  got=$(stat -c %s "$part")
  if [ "$got" != "$size" ]; then
    echo "size mismatch for $rel: $got != $size" >&2
    exit 1
  fi
  mv "$part" "$dest"
done

# SOURCE.json: bytes and SHA256 of every downloaded file (selection and preparation verify them).
tmp="$RAW/SOURCE.json.tmp"
{
  echo "{"
  echo "  \"base_url\": \"$BASE/\","
  echo "  \"files\": {"
  n=${#FILES[@]}
  i=0
  for rel in "${FILES[@]}"; do
    i=$((i + 1))
    bytes=$(stat -c %s "$RAW/$rel")
    sha=$(sha256sum "$RAW/$rel" | awk '{print $1}')
    sep=","
    [ "$i" = "$n" ] && sep=""
    echo "    \"$rel\": {\"bytes\": $bytes, \"sha256\": \"$sha\"}$sep"
  done
  echo "  }"
  echo "}"
} > "$tmp"
mv "$tmp" "$RAW/SOURCE.json"
echo "SOURCE_COMPLETE $RAW/SOURCE.json"
