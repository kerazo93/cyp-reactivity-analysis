# Part 3 — What we buy, and what each purchase is supposed to teach us

## The framing

The obvious move is to rank purchasable space by predicted reactivity and buy the top 1,000. We
did not do that, and the reason is the analysis itself: **68.6% of the assayed library are already
CYP3A4 substrates after correction, and the model's scaffold-split ROC-AUC is 0.82.** A thousand
high-confidence predicted substrates would, if the model is right, return ~900 hits and change
nothing we believe. That is a very expensive way to be told we were correct.

The budget is spent instead on **resolving specific uncertainties**, each with a stated prediction
that the experiment can falsify. Every bucket below answers the question "what would we learn if
this came back the other way?"

| Bucket | n | Question it answers | What a surprise would mean |
|---|---|---|---|
| Model validation | 200 | Do the predicted values mean what they say across the whole range? | Miscalibration → the model ranks but cannot be trusted as a probability |
| Cliff resolution | 250 | Which side of a known activity cliff does new chemistry land on? | The cliff is a local artefact, not a transferable SAR rule |
| Substructure tests | 250 | Are the enriched/depleted groups causal or confounded? | The Part 2 enrichments are scaffold-driven, not group-driven |
| Uncertainty sampling | 200 | Where does the model genuinely not know? | Largest expected reduction in future model error |
| Space expansion | 100 | Does anything transfer outside the training domain? | Honest bounds on where the model may be applied |

## Two constraints applied to every bucket

**MS-detectability.** A compound that does not ionise yields no depletion measurement at all —
the assay simply fails for it. The OpenADMET write-up is explicit that pre-profiling for
MS-compatible molecules "effectively blinds us to a subset of chemical space." We train a
detectability model on the 11,353-compound `will_it_fly` screen and require predicted-detectable
candidates. The threshold is data-derived rather than assumed: a compound is treated as
assay-ready if its ionisation peak area reaches **2,824** — the weakest control well that actually
supported a reactivity measurement in this screen.

**Diversity.** Every bucket is filled greedily subject to a Tanimoto ceiling against compounds
already picked, so no single series can consume a bucket. Without it, the cliff and substructure
buckets collapse onto a handful of analogue series and the 1,000 compounds carry far less than
1,000 compounds' worth of information.

We deliberately **did not** apply the Brenk unwanted-functionality filter. Brenk rejects anilines,
nitroaromatics and halides — and anilines are the strongest depleted substructure in our own Part 2
results. Screening them out would remove exactly the compounds able to test our conclusions. (It
also rejected ~99% of the in-stock pool.) PAINS is retained, since it targets assay interference,
which is a real concern here.

## Why these particular experiments

**Model validation (200).** Picks are stratified across the predicted range rather than taken from
the top, including compounds predicted *inactive*. Calibration can only be measured where
predictions exist, and the low end is where a follow-up campaign would otherwise never look. This
is the bucket that tells us whether the model's numbers can be used for decisions or only for
ranking.

**Cliff resolution (250).** Part 2 found 572 statistically significant activity cliffs — pairs of
near-identical compounds with genuinely different turnover, where the difference survives the
measurement noise. A cliff is the highest-information region of an SAR landscape: it is exactly
where a similarity-based model is most likely to be wrong, and where a small structural change
carries real metabolic consequence. We buy near-neighbours of confirmed cliff compounds to find
out whether the cliff generalises into fresh chemistry or was a property of those two molecules.

**Substructure hypothesis tests (250).** Enrichment is correlational. Anilines are depleted among
substrates, but anilines also co-occur with particular scaffolds and property ranges, so the
association may not be causal. We buy compounds carrying each implicated group *spanning the
predicted-activity range*, so the group's effect can be estimated in chemistry that differs from
the original library. This is the bucket most likely to overturn a Part 2 conclusion, which is
precisely why it is funded.

**Uncertainty sampling (200).** Selection is by **disagreement between trees**, not prediction
entropy. Entropy peaks at p = 0.5 even when every tree agrees — that is a compound the model
confidently believes is borderline, which teaches nothing. Tree variance instead marks regions
where the training data does not determine the answer, which is what actually reduces future model
error.

**Space expansion (100).** Compounds with nearest-neighbour Tanimoto < 0.35 to anything assayed.
These are, by construction, outside the applicability domain and the model's predictions for them
should be distrusted — which is the point. Without them we can never state how far the model
transfers, only assert it.

## Deliverable

[`results/followup_1000.csv`](../results/followup_1000.csv) carries, per compound: ZINC ID and
SMILES, assigned bucket, predicted log10 fold-change with the tree-disagreement uncertainty,
predicted MS-detectability, nearest-neighbour similarity to the training set and an
in-applicability-domain flag, similarity to the nearest confirmed cliff compound, MW/cLogP, and
the tranche's purchasability code.

## Sourcing note

Candidates come from **ZINC20 in-stock tranches** (purchasability codes A and B), restricted to
MW 300–550 and cLogP 1–6 to bracket the assayed library. The tranches hold 2,723,363 in-stock
compounds; since only 1,000 are bought, a fixed-seed uniform sample is drawn before the expensive
structural filtering. The pool is homogeneous by construction — every tranche already spans the
same property region — so the sample is representative, and `--sample 0` scores all of it. ZINC aggregates vendor catalogues —
predominantly Enamine and MolPort, the two the prompt names — so every candidate is orderable
today, and each row carries a `vendor_lookup` URL resolving the ZINC ID to its current suppliers
and catalogue numbers.

This route was chosen for a practical reason worth stating: Enamine's own download links redirect
to a request form and MolPort's catalogue requires registration, so neither can be fetched by a
script. Selecting through ZINC keeps the entire pipeline reproducible from a clean clone while
still delivering Enamine/MolPort-purchasable compounds. Bulk resolution of catalogue numbers for
an order would go through [CartBlanche](https://cartblanche22.docking.org/).
