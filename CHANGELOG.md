# Changelog

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
