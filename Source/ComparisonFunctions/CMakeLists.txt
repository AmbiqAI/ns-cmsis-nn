#
# SPDX-FileCopyrightText: Copyright 2024-2025
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB SRC "./*.c")
target_sources(cmsis-nn PRIVATE ${SRC})
