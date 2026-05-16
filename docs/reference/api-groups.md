# API

heliaCORE keeps the CMSIS-NN C API shape where compatibility matters, then adds
Ambiq-tuned kernels and integration paths for HELIA workloads on Apollo-class
targets. Start with the operator families below, filter to the kernels you care
about, then drill into exact generated signatures when you need implementation
detail.

```{toctree}
:hidden:
:maxdepth: 1

../api/library_root
doxygen
```

<div class="api-story-grid">
  <a class="api-story-card" href="../why.html"><span class="api-tag">positioning</span><strong>What it is</strong><em>A CMSIS-NN-compatible kernel layer packaged for Ambiq firmware and HELIA runtime integration.</em></a>
  <a class="api-story-card" href="../why.html#what-stays-compatible"><span class="api-tag">compatibility</span><strong>How it differs</strong><em>Inherited CMSIS-NN APIs stay recognizable while Ambiq coverage fills model-graph gaps around real deployments.</em></a>
  <a class="api-story-card" href="../getting-started/neuralspot-x.html"><span class="api-tag">integration</span><strong>How it fits HELIA</strong><em>Use heliaCORE through neuralSPOT-X or direct CMake/CMSIS-Pack flows depending on your firmware stack.</em></a>
</div>

## Browse By Kernel Family

<div class="api-group-grid">
  <a class="api-group-card" href="#convolution-functions" data-filter-kind="family" data-filter-value="convolution"><span class="api-tag">compute</span><strong>Convolution</strong><em>Conv2D, depthwise, transpose convolution, wrappers, and buffer helpers.</em></a>
  <a class="api-group-card" href="#fully-connected-layer-functions" data-filter-kind="family" data-filter-value="fully-connected"><span class="api-tag">compute</span><strong>Fully connected</strong><em>Dense layers, batch matmul paths, and scratch sizing helpers.</em></a>
  <a class="api-group-card" href="#elementwise-functions" data-filter-kind="family" data-filter-value="elementwise"><span class="api-tag">graph glue</span><strong>Elementwise</strong><em>Add, sub, mul, square difference, min/max, mean, and arithmetic glue.</em></a>
  <a class="api-group-card" href="#basic-math-and-reduction" data-filter-kind="family" data-filter-value="reduction-comparison"><span class="api-tag">selection</span><strong>Reduction and comparison</strong><em>Argmin/argmax, min/max reductions, comparisons, means, and vector sums.</em></a>
  <a class="api-group-card" href="#activation-functions" data-filter-kind="family" data-filter-value="activation"><span class="api-tag">activation</span><strong>Activation</strong><em>ReLU, LeakyReLU, PReLU, Hard-Swish, Logistic, Tanh, and clamp.</em></a>
  <a class="api-group-card" href="#data-movement" data-filter-kind="family" data-filter-value="data-movement"><span class="api-tag">layout</span><strong>Data movement</strong><em>Pad, reshape, transpose, concatenate, gather, resize, and strided slice.</em></a>
  <a class="api-group-card" href="#classifier-tail" data-filter-kind="family" data-filter-value="classifier-tail"><span class="api-tag">tail</span><strong>Pooling, softmax, quantization</strong><em>Classifier tail APIs plus dtype conversion and requantization utilities.</em></a>
  <a class="api-group-card" href="#sequence-functions" data-filter-kind="family" data-filter-value="sequence"><span class="api-tag">sequence</span><strong>Sequence</strong><em>LSTM and SVDF functions for temporal workloads.</em></a>
  <a class="api-group-card" href="../api/library_root.html"><span class="api-tag">index</span><strong>Types and support</strong><em>Structs, enums, files, and low-level support helpers in the full generated index.</em></a>
</div>

<div class="api-filter-panel" data-api-filter-root>
  <label class="api-filter-search" for="api-filter-query">
    <span>Find a kernel</span>
    <input id="api-filter-query" type="search" placeholder="arm_convolve_s8, softmax, s16, buffer" autocomplete="off">
  </label>
  <div class="api-filter-row" aria-label="Filter by family">
    <button type="button" class="api-filter-chip is-active" data-filter-kind="family" data-filter-value="all" aria-pressed="true">All families</button>
    <button type="button" class="api-filter-chip" data-filter-kind="family" data-filter-value="convolution" aria-pressed="false">Convolution</button>
    <button type="button" class="api-filter-chip" data-filter-kind="family" data-filter-value="fully-connected" aria-pressed="false">Fully connected</button>
    <button type="button" class="api-filter-chip" data-filter-kind="family" data-filter-value="elementwise" aria-pressed="false">Elementwise</button>
    <button type="button" class="api-filter-chip" data-filter-kind="family" data-filter-value="reduction-comparison" aria-pressed="false">Reduction</button>
    <button type="button" class="api-filter-chip" data-filter-kind="family" data-filter-value="activation" aria-pressed="false">Activation</button>
    <button type="button" class="api-filter-chip" data-filter-kind="family" data-filter-value="data-movement" aria-pressed="false">Data movement</button>
    <button type="button" class="api-filter-chip" data-filter-kind="family" data-filter-value="classifier-tail" aria-pressed="false">Classifier tail</button>
    <button type="button" class="api-filter-chip" data-filter-kind="family" data-filter-value="sequence" aria-pressed="false">Sequence</button>
  </div>
  <div class="api-filter-row" aria-label="Filter by data type">
    <button type="button" class="api-filter-chip is-active" data-filter-kind="dtype" data-filter-value="all" aria-pressed="true">All dtypes</button>
    <button type="button" class="api-filter-chip" data-filter-kind="dtype" data-filter-value="s4" aria-pressed="false">s4</button>
    <button type="button" class="api-filter-chip" data-filter-kind="dtype" data-filter-value="s8" aria-pressed="false">s8</button>
    <button type="button" class="api-filter-chip" data-filter-kind="dtype" data-filter-value="s16" aria-pressed="false">s16</button>
    <button type="button" class="api-filter-chip" data-filter-kind="dtype" data-filter-value="s32" aria-pressed="false">s32</button>
    <button type="button" class="api-filter-chip" data-filter-kind="dtype" data-filter-value="u8" aria-pressed="false">u8</button>
    <button type="button" class="api-filter-chip" data-filter-kind="dtype" data-filter-value="fp16" aria-pressed="false">fp16</button>
    <button type="button" class="api-filter-chip" data-filter-kind="dtype" data-filter-value="f32" aria-pressed="false">f32</button>
  </div>
  <div class="api-filter-row" aria-label="Filter by API role">
    <button type="button" class="api-filter-chip is-active" data-filter-kind="role" data-filter-value="all" aria-pressed="true">All roles</button>
    <button type="button" class="api-filter-chip" data-filter-kind="role" data-filter-value="kernel" aria-pressed="false">Kernels</button>
    <button type="button" class="api-filter-chip" data-filter-kind="role" data-filter-value="wrapper" aria-pressed="false">Wrappers</button>
    <button type="button" class="api-filter-chip" data-filter-kind="role" data-filter-value="buffer" aria-pressed="false">Buffer helpers</button>
    <button type="button" class="api-filter-chip" data-filter-kind="role" data-filter-value="optimized" aria-pressed="false">Optimized paths</button>
  </div>
  <p class="api-filter-status" data-api-filter-status></p>
</div>

For file-by-file browsing, structs, enums, and the complete generated tree, open
the <a href="../api/library_root.html">generated C API reference</a>.

## Convolution Functions

```{eval-rst}
.. api-group-index:: convolution
```

## Fully-Connected Layer Functions

```{eval-rst}
.. api-group-index:: fully-connected
```

## Elementwise Functions

```{eval-rst}
.. api-group-index:: elementwise
```

## Basic Math And Reduction

```{eval-rst}
.. api-group-index:: reduction-comparison
```

## Activation Functions

```{eval-rst}
.. api-group-index:: activation
```

## Data Movement

```{eval-rst}
.. api-group-index:: data-movement
```

## Classifier Tail

```{eval-rst}
.. api-group-index:: classifier-tail
```

## Sequence Functions

```{eval-rst}
.. api-group-index:: sequence
```

## Types And Support

Structs, enums, files, and lower-level support helpers are intentionally kept in
the <a href="../api/library_root.html">full generated C API index</a> to avoid
duplicating declarations across the grouped operator view and the complete
Exhale tree.
