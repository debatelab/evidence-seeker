import toml

# Read lock file and extract dependencies
with open("locks/dev_env.lock") as f:
    deps = [
        line.strip()
        for line in f
        if line.strip() and not line.strip().startswith("#")
    ]

# Read pyproject.toml
pyproject = toml.load("pyproject.toml")

# Remove existing dependencies
pyproject["project"]["dependencies"] = []
# Update dependencies
pyproject["project"]["dependencies"] = deps

# Write back to pyproject.toml
with open("pyproject.toml", "w") as f:
    toml.dump(pyproject, f)
