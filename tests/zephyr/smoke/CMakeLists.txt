# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Minimal Zephyr smoke app for the ns-cmsis-nn module.
#
# Builds against an external Zephyr workspace (the CI workflow does the
# `west init`/`west update` and points west at the parent of this repo).
# The app links a single CMSIS-NN kernel and prints the version macro so
# any link- or include-path regression in the Zephyr integration fails the
# build instead of slipping through silently.

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(ns_cmsis_nn_zephyr_smoke LANGUAGES C)

target_sources(app PRIVATE src/main.c)
