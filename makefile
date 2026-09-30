# MBTA subway reliability & ridership analytics (CS 506 final project)
#
#   make all          install dependencies, build the data, train, cluster, plot
#   make test         run the test suite
#
# `PYTHONPATH` is exported so `python -m mbta_ds.cli` resolves without an install
# step. Every target is safe to re-run: collection is cached and incremental.

PYTHON ?= python

export PYTHONPATH := src

#: Service days of history. `make DAYS=180 data` fetches a longer window.
DAYS ?= 90
#: Last service date. Pinned to the published analysis so every machine studies
#: the same 90 days; `make END=latest data` follows the newest data instead.
END ?= 2026-06-30
#: `make NO_RIDERSHIP=1 ...` analyses delays without the ridership source.
NO_RIDERSHIP ?=
#: Optional run name: results go to data/runs/$(RUN)/ and reports/$(RUN)/.
ifdef RUN
export MBTA_RUN := $(RUN)
endif

.PHONY: help all setup data live model model-full cluster figures report test clean distclean

help:
	@echo "Targets:"
	@echo "  setup        check Python 3.11-3.13 and install the pinned dependencies"
	@echo "  data         download, clean, build features, validate (DAYS=$(DAYS), END=$(END))"
	@echo "  live         poll the MBTA V3 API and record live snapshots"
	@echo "  model        train and evaluate the delay and 10+ minute models (Track A)"
	@echo "  model-full   as 'model', plus random forest and KNN"
	@echo "  cluster      cluster stations by reliability and demand (Track B)"
	@echo "  figures      render interactive and static figures"
	@echo "  report       write reports/report.html, reports/story.html and reports/tables/*.csv"
	@echo "  all          setup + data + model + cluster + figures + report"
	@echo "  test         run the test suite"
	@echo "  clean        remove generated data and figures"
	@echo "  distclean    as 'clean', plus downloaded caches"

all: setup data model cluster figures report
	@echo ""
	@echo "Pipeline complete. Open reports/report.html."

setup:
	@echo "==> checking the Python version (3.11-3.13; the pinned wheels stop at 3.13)"
	$(PYTHON) -c "import sys; v=sys.version_info[:2]; sys.exit(0 if (3,11) <= v <= (3,13) else 'Python 3.11-3.13 required, found %d.%d' % v)"
	@echo "==> installing the exact dependency versions the results were produced with"
	$(PYTHON) -m pip install -r requirements-lock.txt

data:
	@echo "==> collecting $(DAYS) days of MBTA data"
	$(PYTHON) -m mbta_ds.cli collect --days $(DAYS) --end $(END) $(if $(NO_RIDERSHIP),--no-ridership)
	@echo "==> cleaning"
	$(PYTHON) -m mbta_ds.cli clean --refresh
	@echo "==> extracting features"
	$(PYTHON) -m mbta_ds.cli features --refresh
	@echo "==> validating data"
	$(PYTHON) -m mbta_ds.cli validate

live:
	@echo "==> polling the V3 API (Ctrl-C to stop)"
	$(PYTHON) -m mbta_ds.cli live --minutes 10 --interval 60

model:
	@echo "==> training and evaluating delay models"
	$(PYTHON) -m mbta_ds.cli train
	$(PYTHON) -m mbta_ds.cli tail

model-full:
	@echo "==> training with the costly model set"
	$(PYTHON) -m mbta_ds.cli train --full
	$(PYTHON) -m mbta_ds.cli tail

cluster:
	@echo "==> clustering stations"
	$(PYTHON) -m mbta_ds.cli cluster

figures:
	@echo "==> rendering figures"
	$(PYTHON) -m mbta_ds.cli figures

report:
	@echo "==> supporting analyses"
	$(PYTHON) -m mbta_ds.cli extras
	@echo "==> writing the report"
	$(PYTHON) -m mbta_ds.cli report
	@echo "==> writing the plain-language story"
	$(PYTHON) -m mbta_ds.cli story

test:
	@echo "==> running tests"
	$(PYTHON) -m pytest -q

clean:
	@echo "==> removing generated outputs"
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('data/processed','reports')]"

distclean: clean
	@echo "==> removing downloaded caches"
	$(PYTHON) -c "import shutil, pathlib; shutil.rmtree('data/raw', ignore_errors=True); pathlib.Path('data/analysis_window.json').unlink(missing_ok=True)"
