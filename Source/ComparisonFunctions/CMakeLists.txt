#
# SPDX-FileCopyrightText: Copyright 2024-2025
# SPDX-License-Identifier: Apache-2.0
#

# Numeric comparisons and Boolean logical entries.
file(GLOB SRC "./*.c")
target_sources(cmsis-nn PRIVATE ${SRC})
