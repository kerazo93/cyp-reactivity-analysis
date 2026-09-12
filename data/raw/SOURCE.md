# Raw data provenance

Downloaded verbatim from the OpenADMET / Octant Bio public release:

- **Dataset**: https://huggingface.co/datasets/openadmet/Octant_CYP_inhibition_reactivity_blog_release
- **Accompanying write-up**: https://openadmet.github.io/Octant_CYP_blog_post/
- **Upstream source**: https://github.com/OpenADMET/Octant_CYP_blog_post
- **Licence**: CC-BY-4.0 — © OpenADMET consortium / Octant Bio
- **Retrieved**: 2026-09-12
- **DOI**: 10.57967/hf/9647

Files are committed unmodified so the analysis is reproducible from a clean
clone. Re-fetch with `make data`.

## SHA-256

```
1859f335ccd9bee9407282b66bc4a690bc8bbf74abbf80d0c5fae27120cadce4  DATASET_CARD.md
19e537166a17a42dd50cc262dd6eb0a963c181830fdc52db0fba98533e01c9c6  inhibition.tsv
531a3dc8d175a9bf0be69cc8594ad310cd64e0997ed4fcd0da2067057737175c  inhibition_wells.tsv
efca9d85202f215edeb929d3786693359589ea83071a3ebcbe1de79318e24834  reactivity.tsv
967c9589e19bc91d8a731adaa8feae038419d57bffe560be8242db87108d008f  reactivity_wells.tsv
afb8482cad6910fce18c14661810b2c3342797fd60833e6daba9065f5cb18d37  will_it_fly_in_mass_spec.tsv
```

## Row counts

| File | Rows | Contents |
|---|---|---|
| `reactivity.tsv` | 2,446 | compound x enzyme summary (no SMILES, no uncertainty) |
| `reactivity_wells.tsv` | 19,344 | well-level peak areas + standardised SMILES |
| `inhibition.tsv` | 1,340 | CYP3A4 pIC50 + QC flags |
| `inhibition_wells.tsv` | 16,931 | well-level dose-response |
| `will_it_fly_in_mass_spec.tsv` | 11,353 | MS ionisation screen |
