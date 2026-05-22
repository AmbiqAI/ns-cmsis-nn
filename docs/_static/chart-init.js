function heliaCoreApplyChartTheme(chart) {
  if (!chart || !chart.options) return;

  const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  const colors = {
    text: isDark ? '#edf7f6' : '#102033',
    muted: isDark ? 'rgba(237,247,246,0.7)' : 'rgba(16,32,51,0.68)',
    grid: isDark ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.07)',
  };

  chart.options.plugins = chart.options.plugins || {};
  chart.options.plugins.legend = chart.options.plugins.legend || {};
  chart.options.plugins.legend.labels = chart.options.plugins.legend.labels || {};
  chart.options.plugins.legend.labels.color = colors.text;

  if (chart.options.plugins.title) {
    chart.options.plugins.title.color = colors.text;
  }

  if (chart.options.scales) {
    Object.values(chart.options.scales).forEach((scale) => {
      if (!scale) return;
      scale.ticks = scale.ticks || {};
      scale.ticks.color = colors.muted;
      if (scale.title) scale.title.color = colors.text;
      scale.grid = scale.grid || {};
      scale.grid.color = colors.grid;
      scale.grid.tickColor = colors.grid;
    });
  }

  chart.update('none');
}

function heliaCoreInitCharts() {
  if (!window.Chart) return;

  document.querySelectorAll('canvas[data-chart-config]').forEach((canvas) => {
    if (canvas.chartjsInitialized) return;
    canvas.chartjsInitialized = true;

    try {
      const config = JSON.parse(canvas.getAttribute('data-chart-config'));
      const chart = new Chart(canvas.getContext('2d'), config);
      canvas.heliaCoreChart = chart;
      heliaCoreApplyChartTheme(chart);
    } catch (e) {
      console.error('heliaCORE chart-init: failed to initialize chart', canvas.id, e);
    }
  });
}

window.addEventListener('DOMContentLoaded', heliaCoreInitCharts);
if (window.matchMedia) {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    document.querySelectorAll('canvas[data-chart-config]').forEach((canvas) => {
      heliaCoreApplyChartTheme(canvas.heliaCoreChart);
    });
  });
}
