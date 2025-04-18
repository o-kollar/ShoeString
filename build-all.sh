#!/bin/bash

set -e

PLATFORMS=(
  "linux amd64"
  "darwin amd64"
  "darwin arm64"
  "windows amd64"
  "linux arm64"
)

for platform in "${PLATFORMS[@]}"
do
  os=$(echo $platform | cut -d ' ' -f 1)
  arch=$(echo $platform | cut -d ' ' -f 2)
  suffix="${os}-${arch}"
  output="shoestring-${suffix}"

  [ "$os" == "windows" ] && output="${output}.exe"

  echo "Building: $output"
  GOOS=$os GOARCH=$arch go build -o "./build/$output"
done

echo "✅ Done! Binaries in ./build"
