# C API By Group

The public headers already define Doxygen group names for the major operator
families. This page uses those families as the primary navigation layer before
the full generated index.

<div class="api-group-grid">
  <a class="api-group-card" href="#convolution-functions"><span class="api-tag">compute</span><strong>Convolution</strong><em>Conv2D, depthwise, transpose convolution, wrappers, and buffer helpers.</em></a>
  <a class="api-group-card" href="#fully-connected-layer-functions"><span class="api-tag">compute</span><strong>Fully connected</strong><em>Dense layers, batch matmul paths, and scratch sizing helpers.</em></a>
  <a class="api-group-card" href="#elementwise-functions"><span class="api-tag">graph glue</span><strong>Elementwise</strong><em>Add, sub, mul, square difference, min/max, mean, and reductions.</em></a>
  <a class="api-group-card" href="#activation-functions"><span class="api-tag">activation</span><strong>Activation</strong><em>ReLU, LeakyReLU, PReLU, Hard-Swish, Logistic, Tanh, and clamp.</em></a>
  <a class="api-group-card" href="#data-movement"><span class="api-tag">layout</span><strong>Data movement</strong><em>Pad, reshape, transpose, concatenate, gather, and strided slice.</em></a>
  <a class="api-group-card" href="#classifier-tail"><span class="api-tag">tail</span><strong>Pooling, softmax, quantization</strong><em>Classifier tail and dtype conversion APIs.</em></a>
  <a class="api-group-card" href="#sequence-functions"><span class="api-tag">sequence</span><strong>Sequence</strong><em>LSTM and SVDF functions for temporal workloads.</em></a>
  <a class="api-group-card" href="../api/library_root.html"><span class="api-tag">index</span><strong>Types and support</strong><em>Structs, enums, files, and low-level support helpers in the full generated index.</em></a>
</div>

For file-by-file browsing, structs, enums, and the complete generated tree, open
the <a href="../api/library_root.html">full generated C API index</a>.

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