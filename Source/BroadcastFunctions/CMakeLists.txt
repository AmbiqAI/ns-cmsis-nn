#
# SPDX-FileCopyrightText: 2025 Ambiq
#
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Licensed under the Ambiq Apollo SDK License.
# See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
#

file(GLOB SRC "./*_*.c")
target_sources(cmsis-nn PRIVATE ${SRC})
