document$.subscribe(() => {
  document.querySelectorAll('canvas[data-chart-config]').forEach((canvas) => {
    if (canvas.chartjsInitialized) return;
    canvas.chartjsInitialized = true;

    const config = JSON.parse(canvas.getAttribute('data-chart-config'));
    const chart = new Chart(canvas.getContext('2d'), config);
    canvas.__chart = chart;
    applyChartTheme(chart);
  });
});

function getChartThemeColors() {
  const scheme = document.body.getAttribute('data-md-color-scheme');
  const dark = scheme === 'slate';

  return {
    text: dark ? '#e6edf3' : '#0b1220',
    muted: dark ? 'rgba(230,237,243,0.7)' : 'rgba(11,18,32,0.65)',
    grid: dark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)',
  };
}

function applyChartTheme(chart) {
  if (!chart || !chart.options) return;

  const colors = getChartThemeColors();

  if (!chart.options.plugins) chart.options.plugins = {};
  if (!chart.options.plugins.legend) chart.options.plugins.legend = {};
  if (!chart.options.plugins.legend.labels) chart.options.plugins.legend.labels = {};
  chart.options.plugins.legend.labels.color = colors.text;

  if (chart.options.plugins.title) {
    chart.options.plugins.title.color = colors.text;
  }

  if (chart.options.scales) {
    Object.values(chart.options.scales).forEach((scale) => {
      if (!scale) return;
      if (!scale.ticks) scale.ticks = {};
      scale.ticks.color = colors.muted;
      if (scale.title) scale.title.color = colors.text;
      if (!scale.grid) scale.grid = {};
      scale.grid.color = colors.grid;
      scale.grid.tickColor = colors.grid;
    });
  }

  chart.update('none');
}

new MutationObserver(() => {
  document.querySelectorAll('canvas[data-chart-config]').forEach((canvas) => {
    if (canvas.__chart) applyChartTheme(canvas.__chart);
  });
}).observe(document.body, { attributes: true, attributeFilter: ['data-md-color-scheme'] });
