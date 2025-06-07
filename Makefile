build:
	@set -e; \
	cp pyproject.toml pyproject.toml.bak; \
	trap 'mv pyproject.toml.bak pyproject.toml' EXIT; \
	python tools/sync_deps.py; \
	hatch build