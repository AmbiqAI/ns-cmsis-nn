// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
import { useMemo, useState } from 'react';
import { Input } from '@ambiqai/helia-ui/react/input';
import type { RefIndexRow } from '@ambiqai/helia-ui/ref-index-model';

const pageSize = 25;

export default function KernelBrowser({ rows }: { rows: RefIndexRow[] }) {
  const [query, setQuery] = useState('');
  const [group, setGroup] = useState('');
  const [dtype, setDtype] = useState('');
  const [sort, setSort] = useState('name');
  const [page, setPage] = useState(0);
  const groups = useMemo(() => [...new Set(rows.map(row => row.group))].sort(), [rows]);
  const dtypes = useMemo(() => [...new Set(rows.flatMap(row => row.facets.dtypes ?? []))].sort(), [rows]);
  const filtered = useMemo(() => {
    const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    return rows.filter(row => (!group || row.group === group) &&
      (!dtype || row.facets.dtypes?.includes(dtype)) &&
      words.every(word => `${row.name} ${row.summary}`.toLowerCase().includes(word)))
      .sort((a, b) => sort === 'reverse' ? b.name.localeCompare(a.name) :
        sort === 'group' ? a.group.localeCompare(b.group) || a.name.localeCompare(b.name) : a.name.localeCompare(b.name));
  }, [rows, query, group, dtype, sort]);
  const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const currentPage = Math.min(page, pages - 1);
  const visible = filtered.slice(currentPage * pageSize, (currentPage + 1) * pageSize);
  const active = query || group || dtype;
  const reset = () => { setQuery(''); setGroup(''); setDtype(''); setPage(0); };

  return <section className="kernel-browser" aria-label="Find a kernel">
    <div className="kernel-controls">
      <label className="kernel-search">Search kernels
        <Input type="search" value={query} placeholder="Name or description" onChange={e => { setQuery(e.target.value); setPage(0); }} />
      </label>
      <label>Operator group<select aria-label="Operator group" value={group} onChange={e => { setGroup(e.target.value); setPage(0); }}>
        <option value="">All groups</option>{groups.map(value => <option key={value}>{value}</option>)}
      </select></label>
      <label>Data type<select aria-label="Data type" value={dtype} onChange={e => { setDtype(e.target.value); setPage(0); }}>
        <option value="">All data types</option>{dtypes.map(value => <option key={value} value={value}>{value === 'f16' ? 'FP16' : value === 'f32' ? 'FP32' : value}</option>)}
      </select></label>
      <label>Sort by<select aria-label="Sort by" value={sort} onChange={e => { setSort(e.target.value); setPage(0); }}>
        <option value="name">Name: A–Z</option><option value="reverse">Name: Z–A</option><option value="group">Operator group</option>
      </select></label>
    </div>
    <div className="kernel-results">
      <p role="status">{filtered.length} of {rows.length} functions{filtered.length > 0 && ` · Showing ${currentPage * pageSize + 1}–${Math.min((currentPage + 1) * pageSize, filtered.length)}`}</p>
      {active && <button type="button" onClick={reset}>Clear filters</button>}
    </div>
    {visible.length ? <div className="kernel-table"><table>
      <caption className="sr-only">Kernel functions matching your filters</caption>
      <thead><tr><th scope="col">Function</th><th scope="col">Description</th></tr></thead>
      <tbody>{visible.map(row => <tr key={row.id}>
        <th scope="row"><a href={row.href}>{row.name}</a><small>{row.group}</small></th>
        <td>{row.summary || 'See the function reference for details.'}</td>
      </tr>)}</tbody>
    </table></div> : <div className="kernel-empty"><p>No kernels match these filters.</p><button type="button" onClick={reset}>Clear filters</button></div>}
    <nav className="kernel-pagination" aria-label="Kernel results pages">
      <button type="button" disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>Previous</button>
      <span>Page {currentPage + 1} of {pages}</span>
      <button type="button" disabled={currentPage + 1 >= pages} onClick={() => setPage(currentPage + 1)}>Next</button>
    </nav>
  </section>;
}
