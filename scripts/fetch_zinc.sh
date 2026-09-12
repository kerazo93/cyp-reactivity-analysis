#!/usr/bin/env bash
# Fetch purchasable, drug-like ZINC20 2D tranches into data/external/zinc/.
#
# Tranche codes are [MW][logP][reactivity][purchasability]:
#   MW   F-J  ~350-500 Da   ) chosen to bracket the assayed library
#   logP F-J  ~2.5-5        )   (median MW 421, median cLogP 3.6)
#   reactivity A/B/C = anodyne / bother / clean
#   purchasability A/B = IN STOCK  (C=via agent, D=make-on-demand,
#                                   E=boutique, F=not for sale)
#
# Only in-stock tranches are fetched, so every candidate is orderable today
# from a vendor catalogue (predominantly Enamine and MolPort).
#
# NOTE: ZINC's licence permits sharing search/screen *results* but not
# redistribution of major portions of the database, so data/external/ is
# gitignored and never committed.
set -euo pipefail
DEST="$(dirname "$0")/../data/external/zinc"
mkdir -p "$DEST"
UA="Mozilla/5.0"
for a in F G H I J; do
  for b in F G H I J; do
    for r in A B C; do
      for p in A B; do
        code="${a}${b}${r}${p}"
        url="https://files.docking.org/2D/${a}${b}/${code}.smi"
        if curl -sfIL --max-time 25 -A "$UA" "$url" >/dev/null 2>&1; then
          if [ ! -s "$DEST/${code}.smi" ]; then
            echo "fetching ${code}"
            curl -sfL --max-time 300 -A "$UA" "$url" -o "$DEST/${code}.smi"
          fi
        fi
      done
    done
  done
done
echo "done: $(ls -1 "$DEST" | wc -l | tr -d ' ') files, $(du -sh "$DEST" | cut -f1)"
