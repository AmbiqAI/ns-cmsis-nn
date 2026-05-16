(() => {
  const root = document.querySelector('[data-api-filter-root]');
  if (!root) {
    return;
  }

  const links = Array.from(document.querySelectorAll('.api-function-link'));
  const search = root.querySelector('#api-filter-query');
  const status = root.querySelector('[data-api-filter-status]');
  const chips = Array.from(root.querySelectorAll('[data-filter-kind]'));
  const sections = Array.from(document.querySelectorAll('article section[id]'));
  const state = {
    family: 'all',
    dtype: 'all',
    role: 'all',
    query: '',
  };

  const updateActive = (kind, value) => {
    chips
      .filter((chip) => chip.dataset.filterKind === kind)
      .forEach((chip) => {
        const active = chip.dataset.filterValue === value;
        chip.classList.toggle('is-active', active);
        chip.setAttribute('aria-pressed', active ? 'true' : 'false');
      });
  };

  const linkMatches = (link) => {
    const name = link.dataset.apiFunction || '';
    const family = link.dataset.apiFamily || '';
    const dtype = link.dataset.apiDtype || '';
    const role = link.dataset.apiRole || '';
    const tokens = state.query.split(/\s+/).filter(Boolean);
    const haystack = `${name} ${name.replaceAll('_', ' ')} ${family} ${dtype} ${role}`;

    return (state.family === 'all' || family === state.family)
      && (state.dtype === 'all' || dtype === state.dtype)
      && (state.role === 'all' || role === state.role)
      && (!tokens.length || tokens.every((token) => haystack.includes(token)));
  };

  const applyFilters = () => {
    let visible = 0;
    links.forEach((link) => {
      const match = linkMatches(link);
      link.hidden = !match;
      if (match) {
        visible += 1;
      }
    });

    sections.forEach((section) => {
      const sectionLinks = Array.from(section.querySelectorAll('.api-function-link'));
      if (!sectionLinks.length) {
        return;
      }
      section.hidden = sectionLinks.every((link) => link.hidden);
    });

    if (status) {
      status.textContent = `${visible} kernel${visible === 1 ? '' : 's'} shown`;
    }
  };

  chips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const kind = chip.dataset.filterKind;
      const value = chip.dataset.filterValue;
      state[kind] = value;
      updateActive(kind, value);
      applyFilters();
    });
  });

  if (search) {
    search.addEventListener('input', () => {
      state.query = search.value.trim().toLowerCase();
      applyFilters();
    });
  }

  applyFilters();
})();
