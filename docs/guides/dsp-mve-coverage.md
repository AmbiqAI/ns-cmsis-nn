# DSP/MVE Coverage

heliaCORE's acceleration work focuses on Ambiq silicon. MVE is a primary target
where it is available, especially for Cortex-M55-class Apollo devices, while DSP
optimized paths remain important for Apollo-class devices that do not include
MVE.

This page summarizes why heliaCORE includes Ambiq-specific operator additions
around the CMSIS-NN-compatible surface. The goal is integration into HELIA AI
workflows for Ambiq devices, with continued respect for the Arm CMSIS ecosystem
and CMSIS-Pack delivery.

## Why coverage matters

Arm CMSIS-NN provides the trusted foundation for efficient neural-network kernels
on Cortex-M. In Ambiq's HELIA workflows, internal profiling across a field-like
model suite highlighted additional Ambiq-specific coverage needs: real models can
spend measurable time in PAD, LeakyReLU, and other glue operators that are easy
to overlook when focusing only on the largest MAC-heavy layers.

heliaCORE responds to those Ambiq silicon needs by adding 200+ DSP/MVE optimized
operators and extra variants around the CMSIS-NN-compatible surface. The numbers
below describe Ambiq's internal coverage target for HELIA workflows on Ambiq
devices.

<div class="workflow workflow--accel" markdown>

<div class="workflow-step" markdown>
<span>1</span>
<strong>MVE-first where available</strong>
<p>Cortex-M55 paths are a primary optimization target, with vectorized kernels
for Ambiq workloads where MVE can move latency.</p>
</div>

<div class="workflow-step" markdown>
<span>2</span>
<strong>DSP coverage for Apollo-class MCUs</strong>
<p>Cortex-M DSP paths remain important for Apollo targets that do not have MVE,
so the library keeps DSP-optimized variants in the release surface.</p>
</div>

<div class="workflow-step" markdown>
<span>3</span>
<strong>Glue operators count too</strong>
<p>Coverage extends beyond obvious MAC-heavy layers to operators that shape real
model latency in HELIA deployments.</p>
</div>

</div>

<div class="chart-card" markdown>
<canvas id="operator-coverage-chart" aria-label="Operator coverage comparison for Ambiq field-like models and MLPerf Tiny" role="img" data-chart-config='{
  "type": "bar",
  "data": {
    "labels": ["Operator types", "Unique operators", "Operator instances"],
    "datasets": [
      {
        "label": "Ambiq field-like suite",
        "data": [53, 247, 963],
        "backgroundColor": "rgba(0, 193, 179, 0.82)",
        "borderColor": "#00c1b3",
        "borderWidth": 1
      },
      {
        "label": "MLPerf Tiny",
        "data": [7, 34, 80],
        "backgroundColor": "rgba(245, 166, 35, 0.78)",
        "borderColor": "#f5a623",
        "borderWidth": 1
      }
    ]
  },
  "options": {
    "responsive": true,
    "maintainAspectRatio": false,
    "plugins": {
      "legend": { "position": "bottom" },
      "title": { "display": true, "text": "Coverage pressure from Ambiq field-like workloads" },
      "tooltip": { "mode": "index", "intersect": false }
    },
    "scales": {
      "x": { "grid": { "display": false } },
      "y": { "beginAtZero": true, "title": { "display": true, "text": "Count" } }
    }
  }
}'></canvas>
</div>

!!! note "Scope"
    This chart is a coverage-oriented view of Ambiq's internal model suite. It
    is not a performance benchmark and is not a statement about the broader Arm
    CMSIS-NN ecosystem. For vendor-neutral Cortex-M kernel work, Arm CMSIS-NN
    remains the upstream ecosystem reference.
