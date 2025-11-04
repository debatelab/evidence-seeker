#!/bin/bash

# See: https://github.com/orgs/quarto-dev/discussions/13611
# Remove files in target directory
rm -rf ../docs/*
# Create .nojekyll file in output directory
touch ../docs/.nojekyll
