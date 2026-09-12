#!/usr/bin/env bash
# Re-fetch the raw OpenADMET CYP release into data/raw/.
set -euo pipefail
BASE="https://huggingface.co/datasets/openadmet/Octant_CYP_inhibition_reactivity_blog_release/resolve/main"
DEST="$(dirname "$0")/../data/raw"
mkdir -p "$DEST"
for f in reactivity.tsv reactivity_wells.tsv inhibition.tsv \
         inhibition_wells.tsv will_it_fly_in_mass_spec.tsv; do
  echo "fetching $f"
  curl -fsSL "$BASE/$f" -o "$DEST/$f"
done
curl -fsSL "$BASE/README.md" -o "$DEST/DATASET_CARD.md"
echo "done -> $DEST"
