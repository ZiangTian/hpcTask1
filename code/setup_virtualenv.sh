#!/bin/#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && pwd)"
(cd $DIR && git submodule update --init --recursive)
(cd $DIR && python3 -m venv python3 venv)
source $DIR/venv/bin/activate
(cd $DIR && $DIR/venv/bin/pip install --editable .)
(cd $DIR && $DIR/venv/bin/pip install --editable dace)