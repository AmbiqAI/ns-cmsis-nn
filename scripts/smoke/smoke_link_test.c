/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>

#include "arm_nnfunctions.h"

int main(void)
{
    int8_t data[4] = {-3, 5, -7, 9};

    arm_relu_q7(data, 4);

    return (data[0] == 0 && data[1] == 5 && data[2] == 0 && data[3] == 9) ? 0 : 1;
}
