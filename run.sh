#!/bin/bash

# Build and run Genetica
# This script configures CMake, builds the project, and runs the app bundle
# (cmake -S . -B build && cd build && make && ./Genetica.app/Contents/MacOS/Genetica)


(cmake -S . -B build && cd build && make && ./WebGPUExample)