#!/bin/bash

set -euo pipefail

pip uninstall lib-ml -y || true

PACKAGE_NAME=$(ls -1 ../lib-ml/dist/*.whl | tail -n1)
echo "Installing package: $PACKAGE_NAME"
pip install $PACKAGE_NAME
