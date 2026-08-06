#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT/models/checkpoints/hierarchy/layer3/layer3_12_4_3"

for f in mashup_softfill_api_big12_mid4_big12*_model_weights.pth; do
  new=""
  if [[ "$f" == "mashup_softfill_api_big12_mid4_big12_model_weights.pth" ]]; then
    new="mashup_softfill_api_all_layer3_12_model_weights.pth"

  elif [[ "$f" =~ ^mashup_softfill_api_big12_mid4_big12_mid([0-9]+)-([0-9]+)_model_weights\.pth$ ]]; then
    k="${BASH_REMATCH[1]}"
    s="${BASH_REMATCH[2]}"
    new="mashup_softfill_api_all_layer3_12_${k}_${k}_${s}_model_weights.pth"

  elif [[ "$f" =~ ^mashup_softfill_api_big12_mid4_big12_mid([0-9]+)_model_weights\.pth$ ]]; then
    k="${BASH_REMATCH[1]}"
    new="mashup_softfill_api_all_layer3_12_${k}_model_weights.pth"
  fi

  if [[ -n "$new" ]]; then
    echo "mv -n -- '$f' '$new'"
  else
    echo "SKIP: $f"
  fi
done
