# Changelog

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
