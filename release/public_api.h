/*
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CMSIS_NN_PUBLIC_API_H
#define CMSIS_NN_PUBLIC_API_H

#if defined(_MSC_VER)
    #define CMSIS_API
#elif defined(__clang__) || defined(__GNUC__) || defined(__ARMCC_VERSION)
    #define CMSIS_API __attribute__((visibility("default")))
#else
    #define CMSIS_API
#endif

#endif /* CMSIS_NN_PUBLIC_API_H */
