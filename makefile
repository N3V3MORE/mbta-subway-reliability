# MBTA subway reliability & ridership analytics (CS 506 final project)
#
#   make all          install dependencies, build the data, train, cluster, plot
#   make test         run the test suite
#
# `PYTHONPATH` is exported so `python -m mbta_ds.cli` resolves without an install
# step. Every target is safe to re-run: collection is cached and incremental.

PYTHON ?= python
ifeq ($(OS),Windows_NT)
PYTHON := python
endif

export PYTHONPATH := src

#: Service days of history. `make DAYS=180 data` fetches a longer window.
DAYS ?= 90

.PHONY: help all setup data live model model-full cluster figures report test clean distclean

help:
	@echo "Targets:"
	@echo "  setup        install Python dependencies"
	@echo "  data         download, clean, build features, validate (DAYS=$(DAYS))"
	@echo "  live         poll the MBTA V3 API and record live snapshots"
	@echo "  model        train and evaluate the delay models (Track A)"
	@echo "  model-full   as 'model', plus random forest and KNN"
	@echo "  cluster      cluster stations by reliability and demand (Track B)"
	@echo "  figures      render interactive and static figures"
	@echo "  report       write reports/report.html and reports/tables/*.csv"
	@echo "  all          setup + data + model + cluster + figures + report"
	@echo "  test         run the test suite"
	@echo "  clean        remove generated data and figures"
	@echo "  distclean    as 'clean', plus downloaded caches"

all: setup data model cluster figures report
	@echo ""
	@echo "Pipeline complete. Open reports/report.html."

setup:
	@echo "==> installing dependencies"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data:
	@echo "==> collecting $(DAYS) days of MBTA data"
	$(PYTHON) -m mbta_ds.cli collect --days $(DAYS)
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

model-full:
	@echo "==> training with the costly model set"
	$(PYTHON) -m mbta_ds.cli train --full

cluster:
	@echo "==> clustering stations"
	$(PYTHON) -m mbta_ds.cli cluster

figures:
	@echo "==> rendering figures"
	$(PYTHON) -m mbta_ds.cli figures

report:
	@echo "==> writing the report"
	$(PYTHON) -m mbta_ds.cli report

test:
	@echo "==> running tests"
	$(PYTHON) -m pytest -q

clean:
	@echo "==> removing generated outputs"
	$(PYTHON) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ('data/processed','reports')]"

distclean: clean
	@echo "==> removing downloaded caches"
	$(PYTHON) -c "import shutil, pathlib; shutil.rmtree('data/raw', ignore_errors=True); pathlib.Path('data/analysis_window.json').unlink(missing_ok=True)"
