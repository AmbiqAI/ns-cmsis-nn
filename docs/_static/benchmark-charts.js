// Benchmark chart configs — loaded after Chart.js and chart-init.js via conf.py
(function () {
  function detectDark() {
    // Furo theme: data-theme can be "light", "dark", or "auto"
    var theme = document.body && document.body.dataset.theme;
    if (theme === 'dark') return true;
    if (theme === 'light') return false;
    // "auto" or unset — follow OS preference
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  function themeColors(isDark) {
    return {
      text: isDark ? '#edf7f6' : '#102033',
      muted: isDark ? 'rgba(237,247,246,0.7)' : 'rgba(16,32,51,0.68)',
      grid: isDark ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.07)'
    };
  }

  var colors = themeColors(detectDark());
  var textColor = colors.text;
  var mutedColor = colors.muted;
  var gridColor = colors.grid;

  var teal = 'rgba(0, 184, 148, 0.82)';
  var tealBorder = 'rgba(0, 184, 148, 1)';
  var dspColor = 'rgba(99, 110, 250, 0.78)';
  var dspBorder = 'rgba(99, 110, 250, 1)';
  var gray = 'rgba(160, 160, 160, 0.65)';
  var grayBorder = 'rgba(160, 160, 160, 1)';

  function scaleOpts(titleText, extra) {
    return Object.assign({
      type: 'logarithmic',
      title: { display: true, text: titleText, color: textColor },
      ticks: { color: mutedColor },
      grid: { color: gridColor, tickColor: gridColor }
    }, extra || {});
  }

  var yOpts = {
    ticks: { font: { family: 'monospace', size: 11 }, color: mutedColor },
    grid: { color: gridColor, tickColor: gridColor }
  };

  function pluginOpts(titleText, showLegend) {
    var opts = {
      title: { display: true, text: titleText, font: { size: 14 }, color: textColor }
    };
    if (showLegend !== false) {
      opts.legend = { labels: { color: textColor } };
    } else {
      opts.legend = { display: false };
    }
    return opts;
  }

  var kernelLabelMap = {
    'arm_convolve_s8': 'convolve_s8',
    'arm_convolve_s4': 'convolve_s4',
    'arm_convolve_s16': 'convolve_s16',
    'arm_convolve_1x1_s8_fast': 'conv_1x1_s8',
    'arm_depthwise_conv_s8_opt': 'dw_conv_s8',
    'arm_depthwise_conv_s4_opt': 'dw_conv_s4',
    'arm_depthwise_conv_fast_s16': 'dw_conv_s16',
    'arm_nn_mat_mult_nt_t_s8': 'mat_mult_s8',
    'arm_nn_mat_mult_nt_t_s4': 'mat_mult_s4',
    'arm_nn_vec_mat_mult_t_s8': 'vec_mat_s8',
    'arm_nn_vec_mat_mult_t_s16': 'vec_mat_s16',
    'arm_nn_vec_mat_mult_t_s4': 'vec_mat_s4',
    'arm_avgpool_s8': 'avgpool_s8',
    'arm_avgpool_s16': 'avgpool_s16',
    'arm_elementwise_add_s8': 'add_s8',
    'arm_elementwise_add_s16': 'add_s16',
    'arm_elementwise_mul_s8': 'mul_s8',
    'arm_elementwise_mul_s16': 'mul_s16',
    'arm_elementwise_sub_s8': 'sub_s8',
    'arm_elementwise_mul_acc_s16': 'mul_acc_s16',
    'arm_elementwise_mul_s16_s8': 'mul_s16_s8',
    'arm_elementwise_mul_s16_batch_offset': 'mul_s16_batch',
    'arm_add_scalar_s8': 'add_scalar_s8',
    'arm_sub_scalar_s8': 'sub_scalar_s8',
    'arm_mul_scalar_s8': 'mul_scalar_s8',
    'arm_mul_scalar_s16': 'mul_scalar_s16',
    'arm_comparison_s8': 'comparison_s8',
    'arm_comparison_s16': 'comparison_s16'
  };

  function compactKernelLabel(rawLabel) {
    var base = rawLabel.replace(/\s*\(.*\)$/, '');
    return kernelLabelMap[base] || base;
  }

  function parseCycles(text) {
    return parseInt(String(text).replace(/,/g, '').trim(), 10);
  }

  function nextTableAfterCanvas(canvas) {
    var node = canvas && canvas.parentElement;
    while (node && node.nextElementSibling) {
      node = node.nextElementSibling;
      if (node.tagName === 'TABLE') return node;
      if (node.tagName === 'DIV') {
        var nestedTable = node.querySelector('table');
        if (nestedTable) return nestedTable;
      }
    }
    return null;
  }

  function buildSpeedupChartFromTable(canvasId, seriesDefs, titleText, xMin) {
    var canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    var table = nextTableAfterCanvas(canvas);
    if (!table || !table.tBodies || !table.tBodies.length) return null;

    var rows = table.tBodies[0].rows;
    var labels = [];
    var datasets = seriesDefs.map(function (series) {
      return {
        label: series.label,
        data: [],
        backgroundColor: series.backgroundColor,
        borderColor: series.borderColor,
        borderWidth: 1
      };
    });

    for (var i = 0; i < rows.length; i++) {
      var cells = rows[i].cells;
      if (!cells || !cells.length) continue;
      labels.push(compactKernelLabel(cells[0].textContent || ''));
      var refValue = parseCycles(cells[seriesDefs[0].referenceColumn].textContent);
      for (var j = 0; j < seriesDefs.length; j++) {
        var currentValue = parseCycles(cells[seriesDefs[j].column].textContent);
        datasets[j].data.push(refValue / currentValue);
      }
    }

    return {
      type: 'bar',
      data: { labels: labels, datasets: datasets },
      options: {
        indexAxis: 'y',
        responsive: true,
        plugins: pluginOpts(titleText),
        scales: {
          x: scaleOpts('Speedup vs REF (log scale)', { min: xMin }),
          y: yOpts
        }
      }
    };
  }

  var chartBuilders = {
    'chart-conv-cycles': function () {
      return buildSpeedupChartFromTable(
        'chart-conv-cycles',
        [
          { label: 'Reference (GCC)', column: 2, referenceColumn: 2, backgroundColor: gray, borderColor: grayBorder },
          { label: 'DSP (GCC)', column: 3, referenceColumn: 2, backgroundColor: dspColor, borderColor: dspBorder },
          { label: 'MVE (GCC)', column: 4, referenceColumn: 2, backgroundColor: teal, borderColor: tealBorder }
        ],
        'REF speedup — Convolution & MatMul (GCC, 250 MHz)',
        0.5
      );
    },
    'chart-elem-cycles': function () {
      return buildSpeedupChartFromTable(
        'chart-elem-cycles',
        [
          { label: 'Reference (GCC)', column: 2, referenceColumn: 2, backgroundColor: gray, borderColor: grayBorder },
          { label: 'DSP (GCC)', column: 3, referenceColumn: 2, backgroundColor: dspColor, borderColor: dspBorder },
          { label: 'MVE (GCC)', column: 4, referenceColumn: 2, backgroundColor: teal, borderColor: tealBorder }
        ],
        'REF speedup — Elementwise & Scalar (GCC, 250 MHz)',
        0.5
      );
    }
  };

  function initBenchmarkCharts(attempt) {
    attempt = attempt || 0;
    if (!window.Chart) return;
    var pending = false;
    Object.keys(chartBuilders).forEach(function (id) {
      var canvas = document.getElementById(id);
      if (!canvas || canvas.chartjsInitialized) return;
      var config = chartBuilders[id]();
      if (!config) {
        pending = true;
        return;
      }
      canvas.chartjsInitialized = true;
      try {
        new Chart(canvas.getContext('2d'), config);
      } catch (e) {
        console.error('benchmark chart error:', id, e);
      }
    });
    if (pending && attempt < 20) {
      window.setTimeout(function () { initBenchmarkCharts(attempt + 1); }, 100);
    }
  }

  function refreshBenchmarkTheme() {
    var c = themeColors(detectDark());
    Object.keys(chartBuilders).forEach(function (id) {
      var canvas = document.getElementById(id);
      if (!canvas) return;
      var inst = Chart.getChart(canvas);
      if (!inst) return;
      try {
        var opts = inst.options;
        if (opts.plugins && opts.plugins.title) opts.plugins.title.color = c.text;
        if (opts.plugins && opts.plugins.legend && opts.plugins.legend.labels) opts.plugins.legend.labels.color = c.text;
        Object.keys(opts.scales || {}).forEach(function (k) {
          var s = opts.scales[k];
          if (s.ticks) s.ticks.color = c.muted;
          if (s.title) s.title.color = c.text;
          if (s.grid) { s.grid.color = c.grid; s.grid.tickColor = c.grid; }
        });
        inst.update('none');
      } catch (e) { /* ignore theme update errors */ }
    });
  }

  // Watch for Furo theme toggle (mutates data-theme on <body>)
  if (typeof MutationObserver !== 'undefined') {
    new MutationObserver(function () { refreshBenchmarkTheme(); })
      .observe(document.body || document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  }
  // Also watch OS-level changes
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', refreshBenchmarkTheme);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initBenchmarkCharts);
  } else {
    initBenchmarkCharts();
  }
})();
