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
  var purple = 'rgba(99, 110, 250, 0.78)';
  var purpleBorder = 'rgba(99, 110, 250, 1)';
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

  var convLabels = [
    'convolve_s8', 'convolve_s4', 'convolve_s16', 'conv_1x1_s8',
    'dw_conv_s8', 'dw_conv_s4', 'dw_conv_s16',
    'mat_mult_s8', 'mat_mult_s4',
    'vec_mat_s8', 'vec_mat_s16', 'vec_mat_s4',
    'avgpool_s8', 'avgpool_s16'
  ];

  var charts = {
    'chart-conv-cycles': {
      type: 'bar',
      data: {
        labels: convLabels,
        datasets: [
          {
            label: 'Reference (GCC)',
            data: [12189, 13206, 12964, 1635, 2453, 1134, 2374, 2643, 3551, 69, 76, 66, 1736, 858],
            backgroundColor: gray,
            borderColor: grayBorder,
            borderWidth: 1
          },
          {
            label: 'DSP (GCC)',
            data: [8096, 17028, 11283, 1273, 1070, 1345, 1414, 1860, 3493, 49, 60, 77, 535, 551],
            backgroundColor: dspColor,
            borderColor: dspBorder,
            borderWidth: 1
          },
          {
            label: 'MVE (GCC)',
            data: [1106, 2432, 4177, 286, 236, 294, 914, 396, 634, 15, 18, 13, 288, 318],
            backgroundColor: teal,
            borderColor: tealBorder,
            borderWidth: 1
          }
        ]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        plugins: pluginOpts('REF vs DSP vs MVE \u2014 Convolution & MatMul (avg cycles, GCC, 250 MHz)'),
        scales: {
          x: scaleOpts('Avg cycles (log scale)', { min: 10 }),
          y: yOpts
        }
      }
    },

    'chart-elem-cycles': {
      type: 'bar',
      data: {
        labels: [
          'add_s8', 'add_s16', 'mul_s8', 'mul_s16', 'sub_s8',
          'mul_acc_s16', 'mul_s16_s8', 'mul_s16_batch',
          'add_scalar_s8', 'sub_scalar_s8', 'mul_scalar_s8', 'mul_scalar_s16',
          'comparison_s8', 'comparison_s16'
        ],
        datasets: [
          {
            label: 'Reference (GCC)',
            data: [50, 43, 18, 14, 49, 17, 17, 16, 36, 37, 17, 15, 43, 43],
            backgroundColor: gray,
            borderColor: grayBorder,
            borderWidth: 1
          },
          {
            label: 'DSP (GCC)',
            data: [39, 43, 21, 20, 41, 20, 19, 18, 28, 28, 20, 17, 32, 36],
            backgroundColor: dspColor,
            borderColor: dspBorder,
            borderWidth: 1
          },
          {
            label: 'MVE (GCC)',
            data: [8, 7, 3, 3, 8, 4, 3, 3, 5, 5, 2, 3, 10, 10],
            backgroundColor: teal,
            borderColor: tealBorder,
            borderWidth: 1
          }
        ]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        plugins: pluginOpts('REF vs DSP vs MVE \u2014 Elementwise & Scalar (avg cycles, GCC, 250 MHz)'),
        scales: {
          x: scaleOpts('Avg cycles (log scale)', { min: 1 }),
          y: yOpts
        }
      }
    },

    'chart-atfe-cycles': {
      type: 'bar',
      data: {
        labels: convLabels,
        datasets: [
          {
            label: 'MVE GCC',
            data: [1106, 2432, 4177, 286, 236, 294, 914, 396, 634, 15, 18, 13, 288, 318],
            backgroundColor: teal,
            borderColor: tealBorder,
            borderWidth: 1
          },
          {
            label: 'MVE ATfE',
            data: [1009, 1971, 2188, 222, 212, 288, 1048, 337, 496, 14, 17, 8, 230, 257],
            backgroundColor: purple,
            borderColor: purpleBorder,
            borderWidth: 1
          }
        ]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        plugins: pluginOpts('MVE: GCC vs ATfE (avg cycles, 250 MHz)'),
        scales: {
          x: scaleOpts('Avg cycles (log scale)', { min: 5 }),
          y: yOpts
        }
      }
    }
  };

  function initBenchmarkCharts() {
    if (!window.Chart) return;
    Object.keys(charts).forEach(function (id) {
      var canvas = document.getElementById(id);
      if (!canvas || canvas.chartjsInitialized) return;
      canvas.chartjsInitialized = true;
      try {
        new Chart(canvas.getContext('2d'), charts[id]);
      } catch (e) {
        console.error('benchmark chart error:', id, e);
      }
    });
  }

  function refreshBenchmarkTheme() {
    var c = themeColors(detectDark());
    Object.keys(charts).forEach(function (id) {
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
