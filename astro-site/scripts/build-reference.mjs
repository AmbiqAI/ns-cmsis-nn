#!/usr/bin/env node
// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
/*
 * Build the generated C API reference: Doxygen XML, then helia-ui-doxyref,
 * then the operator-family index page (AmbiqAI/ns-cmsis-nn#519).
 *
 * The Doxyfile is the one the Sphinx site already uses,
 * Documentation/Doxygen/nn.dxy.in, with an override block appended. Doxygen
 * takes the last assignment of a tag, so the template stays the single
 * definition of what is extracted (PREDEFINED, EXCLUDE_SYMBOLS,
 * OPTIMIZE_OUTPUT_FOR_C) and the block only turns the output from HTML into
 * XML and redirects it into the scratch tree.
 *
 * The families come from reference.config.json rather than a directive
 * because doxyref has no grouping option; see README.md.
 */
import { execFileSync, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const siteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = path.resolve(siteRoot, '..');
const cacheDir = path.join(siteRoot, '.cache', 'reference');
const xmlDir = path.join(cacheDir, 'xml');
const outDir = path.join(siteRoot, 'src', 'content', 'docs', 'reference', 'api');
const publicDir = path.join(siteRoot, 'public');
const routePrefix = 'reference/api';
const indexPage = path.join(outDir, 'index.mdx');

/* The reader in @ambiqai/helia-ui was written against this Doxygen XML
   schema. A different release usually still parses, so this warns rather
   than refusing, but a silently thinner model is worth a line of output. */
const EXPECTED_DOXYGEN = '1.17.0';

const config = JSON.parse(
  fs.readFileSync(path.join(siteRoot, 'reference.config.json'), 'utf8'),
);

const ifMissing = process.argv.includes('--if-missing');
if (ifMissing && fs.existsSync(path.join(outDir, '.doxyref-manifest.json'))) {
  process.exit(0);
}

function git(...args) {
  try {
    return execFileSync('git', args, { cwd: repoRoot, encoding: 'utf8' }).trim();
  } catch {
    return '';
  }
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, { stdio: 'inherit', ...options });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${command} exited with ${result.status}`);
  }
}

/* The Doxyfile is upstream Arm's, written for 1.9.6, so every run under the
   version doxyref targets repeats the same list of retired and
   not-compiled-in tags. Those lines are dropped and everything else,
   including any real error, is passed through. */
const DOXYGEN_TAG_NOISE =
  /(has become obsolete|belongs to an option that was not enabled|^\s+To avoid this warning)/;

function runDoxygen(doxyfile) {
  const result = spawnSync('doxygen', [doxyfile], {
    cwd: cacheDir,
    encoding: 'utf8',
  });
  if (result.error) throw result.error;
  const noise = (result.stderr ?? '')
    .split('\n')
    .filter((line) => line.trim() !== '' && !DOXYGEN_TAG_NOISE.test(line));
  if (noise.length > 0) console.error(noise.join('\n'));
  if (result.stdout) process.stdout.write(result.stdout);
  if (result.status !== 0) {
    throw new Error(`doxygen exited with ${result.status}`);
  }
}

function requireDoxygen() {
  const probe = spawnSync('doxygen', ['--version'], { encoding: 'utf8' });
  if (probe.error || probe.status !== 0) {
    throw new Error(
      'doxygen is not on PATH. Install it (brew install doxygen, or the ' +
        'tarball .github/workflows/docs.yml pins) and run again.',
    );
  }
  const version = probe.stdout.trim().split(/\s+/)[0];
  if (version !== EXPECTED_DOXYGEN) {
    console.warn(
      `build-reference: doxygen ${version} found, helia-ui-doxyref targets ` +
        `${EXPECTED_DOXYGEN}; check the extracted counts below.`,
    );
  }
  return version;
}

function writeDoxyfile(projectNumber) {
  const template = fs.readFileSync(
    path.join(repoRoot, config.doxyfileTemplate),
    'utf8',
  );
  /* The HTML file tags are cleared, not just GENERATE_HTML: doxygen resolves
     and rejects a missing HTML_HEADER before it looks at whether HTML is
     wanted, and those paths are relative to Documentation/Doxygen/. */
  const overrides = [
    '',
    '# Appended by astro-site/scripts/build-reference.mjs. Doxygen takes the',
    '# last assignment of a tag, so these win over the template above.',
    `INPUT                  = ${path.join(repoRoot, config.input)}`,
    'RECURSIVE              = YES',
    `OUTPUT_DIRECTORY       = ${cacheDir}`,
    'GENERATE_XML           = YES',
    'XML_OUTPUT             = xml',
    'GENERATE_HTML          = NO',
    'GENERATE_LATEX         = NO',
    'GENERATE_TREEVIEW      = NO',
    'SEARCHENGINE           = NO',
    'HAVE_DOT               = NO',
    'HTML_HEADER            =',
    'HTML_FOOTER            =',
    'HTML_STYLESHEET        =',
    'HTML_EXTRA_STYLESHEET  =',
    'HTML_EXTRA_FILES       =',
    'LAYOUT_FILE            =',
    'PROJECT_LOGO           =',
    'IMAGE_PATH             =',
    'EXAMPLE_PATH           =',
    'CITE_BIB_FILES         =',
    'QUIET                  = YES',
    'WARNINGS               = NO',
    'WARN_IF_UNDOCUMENTED   = NO',
    'WARN_IF_DOC_ERROR      = NO',
    'WARN_IF_INCOMPLETE_DOC = NO',
    'WARN_NO_PARAMDOC       = NO',
    '',
  ].join('\n');
  const doxyfile = path.join(cacheDir, 'nn.dxy');
  fs.writeFileSync(
    doxyfile,
    template.replaceAll('{projectNumber}', projectNumber) + overrides,
  );
  return doxyfile;
}

/* Prose reaches the page through MDX, where a brace opens an expression and
   `<` an element. The renderer escapes what it emits; these cells are ours. */
function mdxCell(text) {
  return String(text ?? '')
    .replace(/\s+/g, ' ')
    .replace(/([{}])/g, '\\$1')
    .replace(/</g, '&lt;')
    .replace(/\|/g, '\\|')
    .trim();
}

function attr(text) {
  return String(text ?? '').replace(/"/g, '&quot;');
}

/* github-slugger's result for the plain ASCII headings this page uses. */
function slug(heading) {
  return heading
    .toLowerCase()
    .replace(/[^a-z0-9 -]/g, '')
    .replace(/ /g, '-');
}

/* `path` on a module is the Doxygen group id as written -- NNConv and LSTM
   among them -- while the page is served from the slugged route. */
function moduleRoute(modulePath) {
  return `${config.base}${routePrefix}/${modulePath
    .split('.')
    .map((segment) => segment.toLowerCase())
    .join('/')}/`;
}

function collectFunctions(model) {
  const functions = [];
  const visit = (module) => {
    for (const symbol of module.symbols ?? []) {
      if (symbol.kind === 'function' && /^Include\/arm_nnfunctions.*\.h$/.test(symbol.source?.path ?? '')) {
        functions.push({
          id: symbol.id,
          name: symbol.name,
          summary: symbol.summary ?? '',
          header: symbol.source?.path ?? '',
          modulePath: module.path,
          moduleName: module.name,
        });
      }
    }
    for (const child of module.submodules ?? []) visit(child);
  };
  for (const module of model.modules ?? []) visit(module);
  functions.sort((a, b) => a.name.localeCompare(b.name));
  return functions;
}

function renderIndex(model, functions) {
  const compiled = config.groups.map((group) => ({
    ...group,
    regexes: group.patterns.map((pattern) => new RegExp(pattern)),
  }));
  const matches = (group, name) => group.regexes.some((re) => re.test(name));

  /* Each family filters the whole set independently, as the Sphinx directive
     did, so a rename that makes two families claim one kernel shows up as a
     duplicate on the page instead of being hidden by a first-match rule. */
  const membership = compiled.map((group) => ({
    group,
    members: functions.filter((fn) => matches(group, fn.name)),
  }));
  const grouped = new Set(
    membership.flatMap(({ members }) => members.map((fn) => fn.id)),
  );
  const ungrouped = functions.filter((fn) => !grouped.has(fn.id));
  const duplicated = functions.filter(
    (fn) => membership.filter(({ group }) => matches(group, fn.name)).length > 1,
  );

  const headers = new Set(functions.map((fn) => fn.header).filter(Boolean));
  const artifact = (file) => `${config.base}${routePrefix}/${file}`;

  const lines = [
    '---',
    'title: API',
    'description: The heliaCORE C API, grouped by operator family.',
    'sidebar:',
    '  order: 0',
    '---',
    '',
    "import CardGrid from '@ambiqai/helia-ui/astro/CardGrid';",
    "import LinkCard from '@ambiqai/helia-ui/astro/LinkCard';",
    '',
    `Public kernel functions declared in \`arm_nnfunctions*.h\`: ` +
      `${functions.length} across ${headers.size} headers. Start with an ` +
      'operator family, then follow a function through to its parameters, ' +
      'return values and source.',
    '',
    `[Machine-readable model](${artifact('reference.json')}) · ` +
      `[llms.txt](${artifact('llms.txt')}) · ` +
      `[llms-full.txt](${artifact('llms-full.txt')})`,
    '',
    '<CardGrid columns={3} density="compact">',
  ];

  for (const { group, members } of membership) {
    lines.push(
      `  <LinkCard href="#${slug(group.heading)}" ` +
        `title="${attr(group.label)}" ` +
        `eyebrow="${attr(group.eyebrow)}" ` +
        `meta="${members.length} functions">`,
      `    ${mdxCell(group.blurb)}`,
      '  </LinkCard>',
    );
  }
  lines.push('</CardGrid>', '');

  for (const { group, members } of membership) {
    lines.push(
      `## ${group.heading}`,
      '',
      `${mdxCell(group.blurb)} ${members.length} ` +
        `${members.length === 1 ? 'function' : 'functions'}.`,
      '',
      '| Function | Summary | Module |',
      '| --- | --- | --- |',
    );
    for (const fn of members) {
      const href = `${moduleRoute(fn.modulePath)}#${fn.id}`;
      lines.push(
        `| [\`${fn.name}\`](${href}) | ${mdxCell(fn.summary)} | ` +
          `[${mdxCell(fn.moduleName)}](${moduleRoute(fn.modulePath)}) |`,
      );
    }
    lines.push('');
  }

  lines.push(
    '## Types and support',
    '',
    'Structs, enums, macros and typedefs stay on the module pages rather ' +
      `than being repeated here. Open [${mdxCell(model.name)}]` +
      `(${moduleRoute(model.modules[0].path)}) for the full tree.`,
    '',
  );

  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(indexPage, lines.join('\n'));
  return { membership, ungrouped, duplicated, headers };
}

/* One page per operator family, so the Reference section's sidebar can list
   the eight families as its top level instead of the Doxygen module tree
   (AmbiqAI/ns-cmsis-nn#524). The family page carries the same rows as the
   matching section of the index; the index stays the one-page view. */
function renderGroupPages(membership) {
  membership.forEach(({ group, members }, order) => {
    const dir = path.join(outDir, group.id);
    const modules = [
      ...new Map(
        members.map((fn) => [fn.modulePath, fn.moduleName]),
      ).entries(),
    ].sort((a, b) => a[1].localeCompare(b[1]));
    const lines = [
      '---',
      `title: ${JSON.stringify(group.label)}`,
      `description: ${JSON.stringify(`${group.heading}: ${group.blurb}`)}`,
      'sidebar:',
      `  order: ${order + 1}`,
      '---',
      '',
      `${mdxCell(group.blurb)} ${members.length} ` +
        `${members.length === 1 ? 'function' : 'functions'}, declared in ` +
        `${modules.length} ${modules.length === 1 ? 'module' : 'modules'}.`,
      '',
      '| Function | Summary | Module |',
      '| --- | --- | --- |',
    ];
    for (const fn of members) {
      const href = `${moduleRoute(fn.modulePath)}#${fn.id}`;
      lines.push(
        `| [\`${fn.name}\`](${href}) | ${mdxCell(fn.summary)} | ` +
          `[${mdxCell(fn.moduleName)}](${moduleRoute(fn.modulePath)}) |`,
      );
    }
    lines.push(
      '',
      `[All ${config.name} functions in one page]` +
        `(${config.base}${routePrefix}/)`,
      '',
    );
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'index.mdx'), lines.join('\n'));
  });
}

/* The sidebar is declared in astro.config.mjs, and the module pages are not
   in the checkout to declare. This is the list the config reads: a module
   sits under the family that holds most of its functions, and a module whose
   functions are spread across families -- the two public headers, which
   declare kernels for every family -- sits under the headers entry instead of
   being repeated eight times. */
function writeNav(membership, model) {
  const modules = [];
  const visit = (module, depth) => {
    modules.push({
      path: module.path,
      label: module.name,
      depth,
      slug: moduleRoute(module.path)
        .slice(config.base.length)
        .replace(/\/$/, ''),
      total: (module.symbols ?? []).filter((s) => s.kind === 'function').length,
    });
    for (const child of module.submodules ?? []) visit(child, depth + 1);
  };
  for (const module of model.modules ?? []) visit(module, 0);

  const owner = new Map();
  for (const entry of modules) {
    const counts = membership.map(({ group, members }) => ({
      id: group.id,
      count: members.filter((fn) => fn.modulePath === entry.path).length,
    }));
    const best = counts.reduce((a, b) => (b.count > a.count ? b : a));
    if (entry.total > 0 && best.count * 2 >= entry.total) owner.set(entry.path, best.id);
  }

  const nav = {
    note:
      'Generated by astro-site/scripts/build-reference.mjs. The Reference ' +
      'sidebar in astro.config.mjs reads it. Do not edit by hand.',
    groups: membership.map(({ group }) => ({
      id: group.id,
      label: group.label,
      slug: `${routePrefix}/${group.id}`,
      modules: modules
        .filter((entry) => owner.get(entry.path) === group.id)
        .map(({ label, slug }) => ({ label, slug })),
    })),
    /* Only the top of the tree: a module page nested deeper than its family
       lists its own children, and the Internal shim headers do not belong in
       a reader's first view of the API. */
    other: modules
      .filter((entry) => !owner.has(entry.path) && entry.depth <= 1)
      .map(({ label, slug }) => ({ label, slug })),
  };
  const navFile = path.join(siteRoot, 'src', 'data', 'reference-nav.json');
  fs.mkdirSync(path.dirname(navFile), { recursive: true });
  fs.writeFileSync(navFile, `${JSON.stringify(nav, null, 2)}\n`);
  return nav;
}

const doxygenVersion = requireDoxygen();
fs.rmSync(cacheDir, { recursive: true, force: true });
fs.mkdirSync(cacheDir, { recursive: true });

const commit = git('rev-parse', 'HEAD');
const describe = git('describe', '--tags', '--match', 'v*', '--always');
const doxyfile = writeDoxyfile(describe || '0.0.0');

console.log(`build-reference: doxygen ${doxygenVersion} -> ${xmlDir}`);
runDoxygen(doxyfile);

/* The package exports map does not cover the bin scripts, so the shim npm
   links is the supported way in; realpath turns it back into a module path
   the current node can run. */
const doxyref = fs.realpathSync(
  path.join(siteRoot, 'node_modules', '.bin', 'helia-ui-doxyref'),
);
const doxyrefArgs = [
  doxyref,
  '--xml', xmlDir,
  '--out', outDir,
  '--public', publicDir,
  '--base', config.base,
  '--site', config.site,
  '--name', config.name,
  '--language', config.language,
  '--route-prefix', routePrefix,
  '--source-root', repoRoot,
  '--quiet',
];
if (commit) {
  doxyrefArgs.push('--commit', commit);
  doxyrefArgs.push(
    '--source-url',
    config.sourceUrlTemplate.replace('{commit}', commit),
  );
}
run(process.execPath, doxyrefArgs);

const model = JSON.parse(
  fs.readFileSync(path.join(publicDir, routePrefix, 'reference.json'), 'utf8'),
);
// Keep low-level contracts linkable without presenting them as a kernel entry point.
function describeSupport(module) {
  if (module.path.split('.').at(-1).toLowerCase() === 'groupsupport') {
    module.name = 'Internal support';
    const page = path.join(outDir, ...module.path.toLowerCase().split('.'), 'index.mdx');
    const mdx = fs.readFileSync(page, 'utf8')
      .replace(/^title:.*$/m, 'title: "Internal support"\npagefind: false');
    fs.writeFileSync(page, mdx);
  }
  for (const child of module.submodules ?? []) describeSupport(child);
}
for (const module of model.modules ?? []) describeSupport(module);
const functions = collectFunctions(model);
const { membership, ungrouped, duplicated, headers } = renderIndex(model, functions);
renderGroupPages(membership);
const nav = writeNav(membership, model);

console.log(
  `build-reference: ${functions.length} functions across ${headers.size} headers`,
);
for (const { group, members } of membership) {
  console.log(`  ${group.id}: ${members.length}`);
}
if (duplicated.length > 0) {
  console.log(
    `  in more than one family: ${duplicated.length} ` +
      `(${duplicated.slice(0, 5).map((fn) => fn.name).join(', ')})`,
  );
}
console.log(`  no family: ${ungrouped.length}`);
console.log(
  `build-reference: ${nav.groups.length} family pages, ` +
    `${nav.groups.reduce((n, g) => n + g.modules.length, 0)} module pages ` +
    `under a family, ${nav.other.length} under headers and types`,
);
