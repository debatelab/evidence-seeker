build:
	# uncomment if you want to use pinned versions in dev_env.lock as dependencies in the deployed pkg 
	# @set -e; \
	# cp pyproject.toml pyproject.toml.bak; \
	# trap 'mv pyproject.toml.bak pyproject.toml' EXIT; \
	# python tools/sync_deps.py; \
	hatch build