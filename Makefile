# Reproduce the whole analysis:  make all
#
# Requires the conda environment:
#   conda env create -f environment.yml && conda activate cyp-reactivity
PY ?= python
export PYTHONPATH := src

RESULTS := results
RAW     := data/raw/reactivity_wells.tsv
CALLS   := $(RESULTS)/reactivity_calls_all.csv
CLIFFS  := $(RESULTS)/part2_activity_cliffs.csv
CAND    := data/processed/zinc_candidates.parquet
FOLLOW  := $(RESULTS)/followup_1000.csv

.PHONY: all test part1 part2 part3 figures notebooks data zinc clean distclean

all: part1 part2 figures part3

## --- data ------------------------------------------------------------
data: $(RAW)
$(RAW):
	./scripts/fetch_data.sh

zinc:
	./scripts/fetch_zinc.sh

$(CAND):
	@test -d data/external/zinc || (echo ">> run 'make zinc' first" && exit 1)
	$(PY) scripts/prepare_candidates.py

## --- analysis --------------------------------------------------------
part1: $(CALLS)
$(CALLS): $(RAW) src/octant_cyp/*.py
	$(PY) scripts/run_part1.py

part2: $(CLIFFS)
$(CLIFFS): $(CALLS) src/octant_cyp/*.py
	$(PY) scripts/run_part2.py

part3: $(FOLLOW)
$(FOLLOW): $(CLIFFS) $(CAND) src/octant_cyp/*.py
	$(PY) scripts/run_part3.py

figures: $(CALLS) $(CLIFFS)
	$(PY) scripts/make_figures.py

test:
	$(PY) -m pytest tests/ -q

## Re-execute the notebooks in place (they are committed with outputs)
notebooks: $(CALLS) $(CLIFFS) $(FOLLOW)
	cd notebooks && for nb in *.ipynb; do \
	  jupyter nbconvert --to notebook --execute --inplace \
	    --ExecutePreprocessor.timeout=2400 "$$nb" || exit 1; \
	done

## --- housekeeping ----------------------------------------------------
clean:
	rm -rf $(RESULTS)/*.csv $(RESULTS)/figures/*.png

distclean: clean
	rm -rf data/processed data/external
