PY ?= python3
export OPENBLAS_NUM_THREADS = 1
export MPLBACKEND = Agg
.PHONY: all test verify audit results tables figures animations check
all: results tables figures animations check
test:
	$(PY) -m pytest -q code/tests
verify:
	$(PY) code/scripts/verify_green_symbolic.py
audit:
	$(PY) code/scripts/audit_revision.py
results:
	$(PY) code/scripts/compute_results.py
tables:
	$(PY) code/scripts/export_tables.py
figures:
	$(PY) code/scripts/make_figures.py
animations:
	$(PY) code/scripts/make_animations.py
check:
	$(PY) code/scripts/check_package.py
