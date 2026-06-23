# Changelog

## [7.27.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.26.0...v7.27.0) (2026-06-22)


### Features

* add arm_where_s16 kernel + unit test ([634aa2d](https://github.com/AmbiqAI/ns-cmsis-nn/commit/634aa2d458698beee7f163a0383223131ca782ed))
* add tensor shaping kernels (tile, broadcast_to, scatter_nd, mirror_pad, select_v2, where, reverse_sequence, dynamic_update_slice) ([8335274](https://github.com/AmbiqAI/ns-cmsis-nn/commit/8335274b2ed1a9b9476b24b4cd34b4f2e2704b8b))
* add tensor shaping kernels (tile, broadcast_to, scatter_nd, mirror_pad, select_v2, where, reverse_sequence, dynamic_update_slice) ([8335274](https://github.com/AmbiqAI/ns-cmsis-nn/commit/8335274b2ed1a9b9476b24b4cd34b4f2e2704b8b))
* add tensor shaping kernels (tile, broadcast_to, scatter_nd, mirror_pad, select_v2, where, reverse_sequence, dynamic_update_slice) ([dd04ea4](https://github.com/AmbiqAI/ns-cmsis-nn/commit/dd04ea4ced273861dc8674b50c60e7c2c4a0a115))
* Update docs with MVE vs DSP vs Scalar C kernel-benchmarks.md ([8246c60](https://github.com/AmbiqAI/ns-cmsis-nn/commit/8246c607b37e8f44f1bd553e6b899b42f7dd95b0))
* Update docs with MVE vs DSP vs Scalar C kernel-benchmarks.md ([f33a188](https://github.com/AmbiqAI/ns-cmsis-nn/commit/f33a188c851938ff8c557e6947e0acf45f647e4e))


### Bug Fixes

* add new operator groups to CMake build options ([3a3c1d7](https://github.com/AmbiqAI/ns-cmsis-nn/commit/3a3c1d703ba2ef05069b2cf22b5d344196a22c5d))
* avoid single-rounding requantize overflow ([#197](https://github.com/AmbiqAI/ns-cmsis-nn/issues/197)) ([d6735fd](https://github.com/AmbiqAI/ns-cmsis-nn/commit/d6735fd06049574e1cec2821a3cff8ff33738636))
* **broadcast_to:** decouple stride computation from broadcast mask ([0b14668](https://github.com/AmbiqAI/ns-cmsis-nn/commit/0b1466805044ce62ef9bab803aea4a012c9ab856))
* Correct formatting in arm_tile_s8.c ([6266e9e](https://github.com/AmbiqAI/ns-cmsis-nn/commit/6266e9e9005ac99660c363366f2bdaee65a574e1))
* Correct kernel-benchmarks.md cycle counts ([9e75bea](https://github.com/AmbiqAI/ns-cmsis-nn/commit/9e75beafe109bbfd5d0f4505aab6ca6b022bc359))
* pass CI contracts (formatting, PDSC, SSoT, Zephyr/NSX wiring) ([1cd2050](https://github.com/AmbiqAI/ns-cmsis-nn/commit/1cd20502d6cfa7fe17b1ae78300fe567654c6631))
* Update benchmark with LP mode results ([8c1eebc](https://github.com/AmbiqAI/ns-cmsis-nn/commit/8c1eebc23a370e781390b06a8fc7c80aedce7bea))
* Update conv benchmark paragraph and add extra prj.conf setting to Zephyr doc ([c2fa2c5](https://github.com/AmbiqAI/ns-cmsis-nn/commit/c2fa2c5a0a6a47af88a544c883507d50c28c7e4a))
* **zephyr:** align Kconfig with renamed-knob contract ([e2b1a61](https://github.com/AmbiqAI/ns-cmsis-nn/commit/e2b1a61c78e4039fde039c77fab816c1060f5038))


### Refactoring

* **nsx:** make ns-cmsis-nn SoC compatibility wildcard ([#200](https://github.com/AmbiqAI/ns-cmsis-nn/issues/200)) ([3dc5bf2](https://github.com/AmbiqAI/ns-cmsis-nn/commit/3dc5bf2fba329674c3c4e9d2b527169be6e6a5d2))
* use arm_memcpy_s8/s16 instead of raw memcpy ([ea005a6](https://github.com/AmbiqAI/ns-cmsis-nn/commit/ea005a631b7b908f90721602636cfaccd8aaa5a7))

## [7.26.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.25.0...v7.26.0) (2026-05-19)


### Features

* **ci:** publish multi-toolchain staticlibs ([0c33b87](https://github.com/AmbiqAI/ns-cmsis-nn/commit/0c33b879c57388a2102b38ccc1b35768b1bde910))


### Bug Fixes

* Add missing endif to zephyr/Kconfig ([#192](https://github.com/AmbiqAI/ns-cmsis-nn/issues/192)) ([46b6b39](https://github.com/AmbiqAI/ns-cmsis-nn/commit/46b6b39d4ab5f73da2a220d72c212f6368943237))
* **release:** harden pack changelog and CI image tagging ([#183](https://github.com/AmbiqAI/ns-cmsis-nn/issues/183)) ([fa8e5ea](https://github.com/AmbiqAI/ns-cmsis-nn/commit/fa8e5ea828f45e4095e3bbdd00f60ce4be41bcf9))

## [7.25.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.24.1...v7.25.0) (2026-05-15)


### Features

* **release:** complete static releases — SDK tarball, find_package, Zephyr & CMSIS-Pack prebuilt ([#182](https://github.com/AmbiqAI/ns-cmsis-nn/issues/182)) ([aa304cb](https://github.com/AmbiqAI/ns-cmsis-nn/commit/aa304cbb8540aaa6ac68d3bcef072270d288917c))


### Bug Fixes

* repair devcontainer build failures ([#136](https://github.com/AmbiqAI/ns-cmsis-nn/issues/136)) ([47a075a](https://github.com/AmbiqAI/ns-cmsis-nn/commit/47a075a755f84f062f4cd3e878b477cc2ce6ee5f))
* **toolchain:** propagate NS_CMSIS_NN_TARGET_CPU to try_compile ([#180](https://github.com/AmbiqAI/ns-cmsis-nn/issues/180)) ([c46b443](https://github.com/AmbiqAI/ns-cmsis-nn/commit/c46b4437a93a7ceb23bb881e54cad93c801f67d3))

## [7.24.1](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.24.0...v7.24.1) (2026-05-08)


### Bug Fixes

* Change vld1q_s32() to vldrwq_s32() in arm_rsqrt_s16.c ([45e120e](https://github.com/AmbiqAI/ns-cmsis-nn/commit/45e120e506a80ec085cb7053e2a83473987e7ba9))
* Change vld1q_s32() to vldrwq_s32() in arm_rsqrt_s16.c ([45e120e](https://github.com/AmbiqAI/ns-cmsis-nn/commit/45e120e506a80ec085cb7053e2a83473987e7ba9))
* Change vld1q_s32() to vldrwq_s32() in arm_rsqrt_s16.c ([95ffa0d](https://github.com/AmbiqAI/ns-cmsis-nn/commit/95ffa0dbcc6b94083850757676a8fed432871096))
* correct pointer increment in per-row scalar ([0c94945](https://github.com/AmbiqAI/ns-cmsis-nn/commit/0c949456e546c73b29bd0e56a433cab68c5dce1d))
* correct pointer increment in per-row scalar broadcast path for arm_squared_difference_s16 and s8 ([055edf9](https://github.com/AmbiqAI/ns-cmsis-nn/commit/055edf945688008103f25a2c66707afeeec61288))
* correct pointer increments in elementwise functions to ensure proper data handling ([70d893d](https://github.com/AmbiqAI/ns-cmsis-nn/commit/70d893d91fa74c3f5606dc7180884ca88b119022))
* Ensure helia-core-tester.yml fetches submodule tag ([996f0f4](https://github.com/AmbiqAI/ns-cmsis-nn/commit/996f0f42457f15dc2c9683769bfbff1ead367772))
* force submodule tag fetch in coverage-merge-summary job ([817b9d2](https://github.com/AmbiqAI/ns-cmsis-nn/commit/817b9d234635dd071fa9361d605d6581ed3302d5))
* update helia tag retrieval to use abbrev=0 and handle missing tags ([853fa42](https://github.com/AmbiqAI/ns-cmsis-nn/commit/853fa42b632ec053f1cc078cffc3aa00b9bede72))
* update subproject commit reference in helia-core-tester ([9951017](https://github.com/AmbiqAI/ns-cmsis-nn/commit/9951017b04816706fe3f01e99802835e4c7559f0))

## [7.24.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.23.0...v7.24.0) (2026-04-24)


### Features

* add NSX module support (source + prebuilt modes) ([#128](https://github.com/AmbiqAI/ns-cmsis-nn/issues/128)) ([9d8badc](https://github.com/AmbiqAI/ns-cmsis-nn/commit/9d8badc6825e0ca0788d9043f308cabbede4ef33))
* add pattern to dependabot to group bot PRs ([#127](https://github.com/AmbiqAI/ns-cmsis-nn/issues/127)) ([56722a1](https://github.com/AmbiqAI/ns-cmsis-nn/commit/56722a1df32974af81bf748577f72fbd2c2bc4d4))
* route ATfE/Clang to ACLE intrinsics path ([#131](https://github.com/AmbiqAI/ns-cmsis-nn/issues/131)) ([99d4435](https://github.com/AmbiqAI/ns-cmsis-nn/commit/99d4435c645b8a3e7523ce61405e839fe381e778))

## [7.23.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.22.0...v7.23.0) (2026-04-10)


### Features

* add arm_sqrt_s16 kernel with MVE-optimized path ([af1429c](https://github.com/AmbiqAI/ns-cmsis-nn/commit/af1429c92c2979b1480496f91b85fb0eb955e12a))
* add arm_sqrt_s16 kernel with MVE-optimized path ([af1429c](https://github.com/AmbiqAI/ns-cmsis-nn/commit/af1429c92c2979b1480496f91b85fb0eb955e12a))
* add arm_sqrt_s16 kernel with MVE-optimized path ([3c97bcd](https://github.com/AmbiqAI/ns-cmsis-nn/commit/3c97bcd08658216aecd238568a704b1e9272c806))
* add depthwise fast path to arm_convolve_wrapper_s16 ([45d68a3](https://github.com/AmbiqAI/ns-cmsis-nn/commit/45d68a31872ebe318e99c9817dc848b4b8663697))
* add int16 rsqrt kernels and generated tests ([5b4e8ac](https://github.com/AmbiqAI/ns-cmsis-nn/commit/5b4e8ac4504c02605cb6960dae9aef4b372925eb))


### Bug Fixes

* add arm_convolve_s16_depthwise.c to PDSC ([c78ce6b](https://github.com/AmbiqAI/ns-cmsis-nn/commit/c78ce6bb8b453b86c5a3283c7428656ff75f0594))
* add arm_sqrt_s16.c to PDSC source file listing ([2e87b7e](https://github.com/AmbiqAI/ns-cmsis-nn/commit/2e87b7e9e207cf1a918f9f40ce5012e8e8afb688))
* depthwise scalar path uses arm_nn_requantize for int32 bias, add uint16 offset overflow guard ([932650a](https://github.com/AmbiqAI/ns-cmsis-nn/commit/932650a3840b90ace258ba378e9b5b27339461d5))
* remove unnecessary casts discarding const from LUT pointers ([576da80](https://github.com/AmbiqAI/ns-cmsis-nn/commit/576da80b6ec3ba5a3cf2aa5b89e53389273afe1c))


### Refactoring

* remove duplicate static helper, call arm_convolve_s16_group_ch_mult_1 from wrapper ([73679b3](https://github.com/AmbiqAI/ns-cmsis-nn/commit/73679b39fb3da7b8166b73778e722b1921184327))
* rename arm_convolve_s16_depthwise to arm_convolve_s16_group_ch_mult_1 ([03b2f86](https://github.com/AmbiqAI/ns-cmsis-nn/commit/03b2f868de6df0a2eeb686d10647563ac2139f83))

## [7.22.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.21.0...v7.22.0) (2026-03-24)


### Features

* Add int8 SQRT kernel. ([2aa4bc5](https://github.com/AmbiqAI/ns-cmsis-nn/commit/2aa4bc5b5fa48b818dc34fdc7f500246c71de194))
* Add int8 SQRT kernel. ([2aa4bc5](https://github.com/AmbiqAI/ns-cmsis-nn/commit/2aa4bc5b5fa48b818dc34fdc7f500246c71de194))
* Add int8 SQRT kernel. Update requirements.txt to tensorflow 2.21 to support quantizing SQRT. Use modern Keras for compatibility with modern Tensorflow ([5f43a09](https://github.com/AmbiqAI/ns-cmsis-nn/commit/5f43a09cf79f292ce681a2c5670b5752573e6b37))
* Add int8/int16 squared difference ([13199ba](https://github.com/AmbiqAI/ns-cmsis-nn/commit/13199baca247421639269d662a0ac07a72d9f473))
* Add int8/int16 Squared Difference ([fa6ba21](https://github.com/AmbiqAI/ns-cmsis-nn/commit/fa6ba212b72f6797e582cccf8f590cabe34bd9d5))
* Add MVE path to squared difference, add int16 tests ([75e7d06](https://github.com/AmbiqAI/ns-cmsis-nn/commit/75e7d065bb7de8a6d105a96ab69b7595034fe4eb))
* Update ARM.CMSIS-NN.pdsc with squared_differnece ([f69896a](https://github.com/AmbiqAI/ns-cmsis-nn/commit/f69896abaf3105233518958094a07d254f8c0638))
* Vectorize arm_sqrt_s8.c and place LUT into TCM ([6d38e45](https://github.com/AmbiqAI/ns-cmsis-nn/commit/6d38e458087a0aa9fbc12710b43adce9e7382900))


### Bug Fixes

* Default RefactoredTestGen to legacy Keras and fall back to modern Keras only when tf_keras is unavailable ([9344acf](https://github.com/AmbiqAI/ns-cmsis-nn/commit/9344acf216046a24c27d692d8f247dff7b386c49))
* Scope KEras/Tensorflow upgrade requirements to SQRT test generation only by restoring baseline UnitTest deps and reverting non-SQRT generators to tf_keras ([010a198](https://github.com/AmbiqAI/ns-cmsis-nn/commit/010a198fcd3549c27e6df5d565bc61921a5c1db7))

## [7.21.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.20.0...v7.21.0) (2026-02-18)


### Features

* Add s32 variant of concatenation ([7e95d01](https://github.com/AmbiqAI/ns-cmsis-nn/commit/7e95d01a604cba15d8157399f8da54d4892be8e7))
* Add s32 variant of concatenation ([7e95d01](https://github.com/AmbiqAI/ns-cmsis-nn/commit/7e95d01a604cba15d8157399f8da54d4892be8e7))
* Add s32 variant of concatenation ([825d406](https://github.com/AmbiqAI/ns-cmsis-nn/commit/825d4069a717555a1d5724542e05bb7f79d56249))
* Add support for s32 strided slice ([1df6b34](https://github.com/AmbiqAI/ns-cmsis-nn/commit/1df6b3446c4e68cd94960f456020cc58951b6183))
* Add support for s32 strided slice. Add int32 io support to test.py ([4ba775d](https://github.com/AmbiqAI/ns-cmsis-nn/commit/4ba775dc9696a221501aa524cac44b6408d32afa))
* Changes arm_strided_slice_s32() to use arm_memcpy_s32() ([5afc9df](https://github.com/AmbiqAI/ns-cmsis-nn/commit/5afc9df96eb62b728be6dd43341a6ea3b13564ec))


### Bug Fixes

* Add shape checks, correct loop counter type to prevent overflow, move copy size out of loop, and add unit tests ([02217c5](https://github.com/AmbiqAI/ns-cmsis-nn/commit/02217c54cd5c4baeb983907db1fd50e8b2f911bb))
* Move validate_s32() to Utils/validate.h and update zephyr/CMakeLists.txt to include s32 variant of strided slice ([049714d](https://github.com/AmbiqAI/ns-cmsis-nn/commit/049714d2e72b42ddde32a966459e1a58214a9a50))
* Remove duplicate output_tensor.h test data ([aa9f026](https://github.com/AmbiqAI/ns-cmsis-nn/commit/aa9f026ae83c15ec3fd42a6d0c5e566e21adf4b5))
* Update ARM&gt;CMSIS-NN.pdsc file with arm_concatenation_s32.c ([46dccd6](https://github.com/AmbiqAI/ns-cmsis-nn/commit/46dccd6624a5a6b4067e183ca74460181afa6316))

## [7.20.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.19.0...v7.20.0) (2026-02-06)


### Features

* Add int8/int16 absolute value ([09e1b8d](https://github.com/AmbiqAI/ns-cmsis-nn/commit/09e1b8de996be41e156677d9a7291b1e35a8ba9e))

## [7.19.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.18.0...v7.19.0) (2026-01-14)


### Features

* Add resize nearest neighbor operator ([c35f844](https://github.com/AmbiqAI/ns-cmsis-nn/commit/c35f844e7e2372a7b2c03dacce5bf52e7d4e9397), [3c49c3f](https://github.com/AmbiqAI/ns-cmsis-nn/commit/3c49c3f41b522c29532c768be58e6e68527ee607))
* Move nearest neighbor coordinate mapping into precomputed x/y arrays stored in ctx-&gt;buf ([feeb063](https://github.com/AmbiqAI/ns-cmsis-nn/commit/feeb06317f37d2c1b71452c322183940f37787bf))
* Move scale and offset computation out of GetNearestNeighbor ([b37d6e1](https://github.com/AmbiqAI/ns-cmsis-nn/commit/b37d6e182f67f91141bec44a6f755d091b843e1c))


### Bug Fixes

* Add tflite-micro back to requirements.txt ([16b2195](https://github.com/AmbiqAI/ns-cmsis-nn/commit/16b2195b09264962b033305beb2762d07a98f7d8))
* Correct resize functions to pass in correct size to arm_memcpy invocation. ([8723c80](https://github.com/AmbiqAI/ns-cmsis-nn/commit/8723c804f42b26d372aae6f2401afcf1601a6474))
* Correct whitespace inconsistencies ([d6861ec](https://github.com/AmbiqAI/ns-cmsis-nn/commit/d6861ecffb19b3892a9b9a2600c31aaccf561dec))
* Include ARM_CMSIS_NN_ARG_ERROR as a possible return code ([51d77d1](https://github.com/AmbiqAI/ns-cmsis-nn/commit/51d77d1e03cce80a8cb119b47ef35dbe2292c8c3))
* Update pdsc file ([04c8839](https://github.com/AmbiqAI/ns-cmsis-nn/commit/04c8839ed066dfa573c29074e8c02c5bca130055))
* Update repository URL for Ethos-U core platform in test setup script ([c160efe](https://github.com/AmbiqAI/ns-cmsis-nn/commit/c160efe2ca4d5f6ac58df42f556be6f6163a1f21))
* Update URL for Ethos-U core platform ([7e0fd73](https://github.com/AmbiqAI/ns-cmsis-nn/commit/7e0fd731ef43ef77798f19e8acefac89b2aa9cfe))

## [7.18.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.17.0...v7.18.0) (2025-11-18)


### Features

* Add PReLU operator for s8 ([de8c1dd](https://github.com/AmbiqAI/ns-cmsis-nn/commit/de8c1dd1d67adce6b1c1a0d249a0dc6f11d37e60))
* Add scalar and broadcastable implementation to s8 prelu ([71a6a11](https://github.com/AmbiqAI/ns-cmsis-nn/commit/71a6a115c3c806496e88ce9725e2671f226c7391))


### Bug Fixes

* Correct arm_elementwise_prelu_s8.c Title ([9e38e55](https://github.com/AmbiqAI/ns-cmsis-nn/commit/9e38e55e2c63b33a365326d830ad89b0623f4d18))
* Correct prelu function signature to match leaky relu ([07dd810](https://github.com/AmbiqAI/ns-cmsis-nn/commit/07dd810da4dbaa54e23e41b854e69abebab1ffdc))
* Correct prelu shift type to int32_t ([540f159](https://github.com/AmbiqAI/ns-cmsis-nn/commit/540f159992c49f92ab9389d7152bb1750743f323))
* Remove accidental commit to gitignore ([77f2554](https://github.com/AmbiqAI/ns-cmsis-nn/commit/77f2554d2b4c9701cdd58be2b2b8be42c9d76b2d))

## [7.17.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.16.0...v7.17.0) (2025-11-10)


### Features

* Add GATHER and GATHER_ND operators for s8 and s16. ([#86](https://github.com/AmbiqAI/ns-cmsis-nn/issues/86)) ([f95d70d](https://github.com/AmbiqAI/ns-cmsis-nn/commit/f95d70d235ac4611897857295bcbf36614bd893a))


### Bug Fixes

* Correct arm_convolve_s16 w/ groups &gt; 1 ([#88](https://github.com/AmbiqAI/ns-cmsis-nn/issues/88)) ([#75](https://github.com/AmbiqAI/ns-cmsis-nn/issues/75)) ([1d4356f](https://github.com/AmbiqAI/ns-cmsis-nn/commit/1d4356ff4db7efc558338d10a9cbcf19a945344e))

## [7.16.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.15.0...v7.16.0) (2025-11-04)


### Features

* Add baseline implementations for argmin and argmax. ([#83](https://github.com/AmbiqAI/ns-cmsis-nn/issues/83)) ([e65f8ca](https://github.com/AmbiqAI/ns-cmsis-nn/commit/e65f8caa3f3f6265e1f91b47b1f268f5310c767c))

## [7.15.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.14.0...v7.15.0) (2025-11-04)


### Features

* Add SUB operator for s8 and s16. ([#81](https://github.com/AmbiqAI/ns-cmsis-nn/issues/81)) ([6c0b213](https://github.com/AmbiqAI/ns-cmsis-nn/commit/6c0b213ddee6df7a6c8df6d01ebd233cedc55c5e))

## [7.14.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.13.1...v7.14.0) (2025-11-03)


### Features

* Add comparison functions and unit tests for int8 and int16 tensors ([#79](https://github.com/AmbiqAI/ns-cmsis-nn/issues/79)) ([4f7b2dc](https://github.com/AmbiqAI/ns-cmsis-nn/commit/4f7b2dc8e4c1f03c76a519d04804f16950133932))

## [7.13.1](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.13.0...v7.13.1) (2025-10-28)


### Bug Fixes

* Correctly adjust quantization multiplier for arm_fully_connected_per_channel_s16  ([#76](https://github.com/AmbiqAI/ns-cmsis-nn/issues/76)) (Closes [#77](https://github.com/AmbiqAI/ns-cmsis-nn/issues/77)) ([1876ec6](https://github.com/AmbiqAI/ns-cmsis-nn/commit/1876ec66bc7cb753ba6b813a507614c4934e43d6))

## [7.13.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.12.0...v7.13.0) (2025-10-08)


### Features

* Update ARM.CMSIS-NN.pdsc ([#72](https://github.com/AmbiqAI/ns-cmsis-nn/issues/72)) ([e61a443](https://github.com/AmbiqAI/ns-cmsis-nn/commit/e61a443820a95e497ea492f1a02263a5b041c346))

## [7.12.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.11.0...v7.12.0) (2025-10-08)


### Features

* Add batch/space/depth transforms. ([#68](https://github.com/AmbiqAI/ns-cmsis-nn/issues/68)) ([0f08e53](https://github.com/AmbiqAI/ns-cmsis-nn/commit/0f08e536e76aab9082109a333422110a50009f8c))

## [7.11.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.10.2...v7.11.0) (2025-10-03)


### Features

* Add REDUCE_MIN operator. ([58ee9ee](https://github.com/AmbiqAI/ns-cmsis-nn/commit/58ee9ee53a9637c0c6f9f795454ba1302a16202e))

## [7.10.2](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.10.1...v7.10.2) (2025-10-03)


### Bug Fixes

* Correct pack build. ([#63](https://github.com/AmbiqAI/ns-cmsis-nn/issues/63)) ([bbb76c9](https://github.com/AmbiqAI/ns-cmsis-nn/commit/bbb76c9cdac669efd9b3eee2f3ba644d3a065e20))

## [7.10.1](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.10.0...v7.10.1) (2025-10-03)


### Bug Fixes

* Correct pack build. ([#61](https://github.com/AmbiqAI/ns-cmsis-nn/issues/61)) ([a2eebdf](https://github.com/AmbiqAI/ns-cmsis-nn/commit/a2eebdfd1a556b98344c2fbfd6887f4077d9eb28))

## [7.10.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.9.0...v7.10.0) (2025-10-03)


### Features

* Streamline releases. ([#59](https://github.com/AmbiqAI/ns-cmsis-nn/issues/59)) ([b2ab7cf](https://github.com/AmbiqAI/ns-cmsis-nn/commit/b2ab7cfce80bed7c90dba92894a32c45068b0336))

## [7.9.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.8.0...v7.9.0) (2025-10-03)


### Features

* Add additional activation support for S16 ([#50](https://github.com/AmbiqAI/ns-cmsis-nn/issues/50)) ([7f81aa7](https://github.com/AmbiqAI/ns-cmsis-nn/commit/7f81aa7c26c3869d2b2f63d32da0db11da2168cd))
* REDUCE_MIN Operator ([#52](https://github.com/AmbiqAI/ns-cmsis-nn/issues/52)) ([31153c7](https://github.com/AmbiqAI/ns-cmsis-nn/commit/31153c70938f1baa2e5a286973dd886db17d0d75))
* Streamline releases. ([#57](https://github.com/AmbiqAI/ns-cmsis-nn/issues/57)) ([8aa0f1d](https://github.com/AmbiqAI/ns-cmsis-nn/commit/8aa0f1d8cc0bf3e323419cd193abfa2eaa194d7e))

## [7.8.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/ns-cmsis-nn-v7.7.0...ns-cmsis-nn-v7.8.0) (2025-10-03)


### Features

* Add additional activation support for S16 ([#50](https://github.com/AmbiqAI/ns-cmsis-nn/issues/50)) ([7f81aa7](https://github.com/AmbiqAI/ns-cmsis-nn/commit/7f81aa7c26c3869d2b2f63d32da0db11da2168cd))
* REDUCE_MIN Operator ([#52](https://github.com/AmbiqAI/ns-cmsis-nn/issues/52)) ([31153c7](https://github.com/AmbiqAI/ns-cmsis-nn/commit/31153c70938f1baa2e5a286973dd886db17d0d75))
