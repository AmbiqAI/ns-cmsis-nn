/*
 * SPDX-FileCopyrightText: Copyright 2010-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nntables.c
 * Description:  Converts the elements of the Q7 vector to Q15 vector without left-shift
 *
 * $Date:        28 October 2022
 * $Revision:    V.2.1.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_lut_s16.h"

/**
 * @brief tables for various activation functions
 *
 * This file include the declaration of common tables.
 * Most of them are used for activation functions
 *
 */

// Table of sigmoid(i/24) at 0.16 format - 256 elements.
// Combined sigmoid and tanh look-up table, since
// tanh(x) = 2*sigmoid(2*x) -1.
// Both functions are symmetric, so the LUT table is only needed
// for the absolute value of the input.
/* Keep the public Q0.16 samples and the DSP interpolation knots in sync.
 * Entry zero is the midpoint and has no distinct negative counterpart. */
// clang-format off
#define SIGMOID_NONZERO_SAMPLES(X) \
    X(1, 33451) X(2, 34133) X(3, 34813) X(4, 35493) X(5, 36169) X(6, 36843) \
    X(7, 37513) X(8, 38180) X(9, 38841) X(10, 39498) X(11, 40149) X(12, 40794) \
    X(13, 41432) X(14, 42064) X(15, 42688) X(16, 43304) X(17, 43912) X(18, 44511) \
    X(19, 45102) X(20, 45683) X(21, 46255) X(22, 46817) X(23, 47369) X(24, 47911) \
    X(25, 48443) X(26, 48964) X(27, 49475) X(28, 49975) X(29, 50464) X(30, 50942) \
    X(31, 51409) X(32, 51865) X(33, 52311) X(34, 52745) X(35, 53169) X(36, 53581) \
    X(37, 53983) X(38, 54374) X(39, 54755) X(40, 55125) X(41, 55485) X(42, 55834) \
    X(43, 56174) X(44, 56503) X(45, 56823) X(46, 57133) X(47, 57433) X(48, 57724) \
    X(49, 58007) X(50, 58280) X(51, 58544) X(52, 58800) X(53, 59048) X(54, 59288) \
    X(55, 59519) X(56, 59743) X(57, 59959) X(58, 60168) X(59, 60370) X(60, 60565) \
    X(61, 60753) X(62, 60935) X(63, 61110) X(64, 61279) X(65, 61441) X(66, 61599) \
    X(67, 61750) X(68, 61896) X(69, 62036) X(70, 62172) X(71, 62302) X(72, 62428) \
    X(73, 62549) X(74, 62666) X(75, 62778) X(76, 62886) X(77, 62990) X(78, 63090) \
    X(79, 63186) X(80, 63279) X(81, 63368) X(82, 63454) X(83, 63536) X(84, 63615) \
    X(85, 63691) X(86, 63765) X(87, 63835) X(88, 63903) X(89, 63968) X(90, 64030) \
    X(91, 64090) X(92, 64148) X(93, 64204) X(94, 64257) X(95, 64308) X(96, 64357) \
    X(97, 64405) X(98, 64450) X(99, 64494) X(100, 64536) X(101, 64576) X(102, 64614) \
    X(103, 64652) X(104, 64687) X(105, 64721) X(106, 64754) X(107, 64786) X(108, 64816) \
    X(109, 64845) X(110, 64873) X(111, 64900) X(112, 64926) X(113, 64950) X(114, 64974) \
    X(115, 64997) X(116, 65019) X(117, 65039) X(118, 65060) X(119, 65079) X(120, 65097) \
    X(121, 65115) X(122, 65132) X(123, 65149) X(124, 65164) X(125, 65179) X(126, 65194) \
    X(127, 65208) X(128, 65221) X(129, 65234) X(130, 65246) X(131, 65258) X(132, 65269) \
    X(133, 65280) X(134, 65291) X(135, 65301) X(136, 65310) X(137, 65319) X(138, 65328) \
    X(139, 65337) X(140, 65345) X(141, 65352) X(142, 65360) X(143, 65367) X(144, 65374) \
    X(145, 65381) X(146, 65387) X(147, 65393) X(148, 65399) X(149, 65404) X(150, 65410) \
    X(151, 65415) X(152, 65420) X(153, 65425) X(154, 65429) X(155, 65433) X(156, 65438) \
    X(157, 65442) X(158, 65445) X(159, 65449) X(160, 65453) X(161, 65456) X(162, 65459) \
    X(163, 65462) X(164, 65465) X(165, 65468) X(166, 65471) X(167, 65474) X(168, 65476) \
    X(169, 65479) X(170, 65481) X(171, 65483) X(172, 65485) X(173, 65488) X(174, 65489) \
    X(175, 65491) X(176, 65493) X(177, 65495) X(178, 65497) X(179, 65498) X(180, 65500) \
    X(181, 65501) X(182, 65503) X(183, 65504) X(184, 65505) X(185, 65507) X(186, 65508) \
    X(187, 65509) X(188, 65510) X(189, 65511) X(190, 65512) X(191, 65513) X(192, 65514) \
    X(193, 65515) X(194, 65516) X(195, 65517) X(196, 65517) X(197, 65518) X(198, 65519) \
    X(199, 65520) X(200, 65520) X(201, 65521) X(202, 65522) X(203, 65522) X(204, 65523) \
    X(205, 65523) X(206, 65524) X(207, 65524) X(208, 65525) X(209, 65525) X(210, 65526) \
    X(211, 65526) X(212, 65526) X(213, 65527) X(214, 65527) X(215, 65528) X(216, 65528) \
    X(217, 65528) X(218, 65529) X(219, 65529) X(220, 65529) X(221, 65529) X(222, 65530) \
    X(223, 65530) X(224, 65530) X(225, 65530) X(226, 65531) X(227, 65531) X(228, 65531) \
    X(229, 65531) X(230, 65531) X(231, 65532) X(232, 65532) X(233, 65532) X(234, 65532) \
    X(235, 65532) X(236, 65532) X(237, 65533) X(238, 65533) X(239, 65533) X(240, 65533) \
    X(241, 65533) X(242, 65533) X(243, 65533) X(244, 65533) X(245, 65534) X(246, 65534) \
    X(247, 65534) X(248, 65534) X(249, 65534) X(250, 65534) X(251, 65534) X(252, 65534) \
    X(253, 65534) X(254, 65534) X(255, 65535)
// clang-format on

#define SIGMOID_VALUE(i, value) value,
const uint16_t sigmoid_table_uint16[256] = {32768, SIGMOID_NONZERO_SAMPLES(SIGMOID_VALUE)};
#undef SIGMOID_VALUE

#ifdef ARM_NN_ACTIVATION_S16_DSP_LUT
    /* Signed knots support interpolation in either direction with a single word
     * load. LOGISTIC rounds both final Q0.16 values to the same Q0.15 result. */
    #define TANH_KNOTS(i, value) [256 + (i)] = (value) - 32768, [256 - (i)] = 32768 - (value),
const int16_t arm_nn_tanh_knots_s16[513] = {[0] = -32767, [512] = 32767, SIGMOID_NONZERO_SAMPLES(TANH_KNOTS)};
    #undef TANH_KNOTS

    /* Adding one to both knots includes the 512-unit rounding bias in the dot product. */
    #define LOGISTIC_VALUE(value) ((value) < 65535 ? (value) - 32768 : 32766)
    #define LOGISTIC_KNOTS(i, value) [256 + (i)] = 1 + LOGISTIC_VALUE(value), [256 - (i)] = 1 - LOGISTIC_VALUE(value),
const int16_t arm_nn_logistic_knots_s16[513] =
    {[0] = -32765, [256] = 1, [512] = 32767, SIGMOID_NONZERO_SAMPLES(LOGISTIC_KNOTS)};
    #undef LOGISTIC_KNOTS
    #undef LOGISTIC_VALUE
#endif
#undef SIGMOID_NONZERO_SAMPLES
