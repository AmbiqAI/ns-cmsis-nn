// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
const variants = {
  note: ['note', 'Note'],
  tip: ['tip', 'Tip'],
  caution: ['warning', 'Caution'],
  danger: ['critical', 'Danger'],
};

export default function markdownCallouts() {
  let imported = false;
  return {
    name: 'helia-markdown-callouts',
    containerDirective(node, ctx) {
      if (ctx.sourceFormat !== 'mdx' || !variants[node.name]) return;
      const [tone, defaultTitle] = variants[node.name];
      const children = [...node.children];
      const title = children[0]?.data?.directiveLabel
        ? ctx.textContent(children.shift()) : defaultTitle;
      if (!imported) {
        let root = node;
        while (ctx.parent(root)) root = ctx.parent(root);
        ctx.prependChild(root, {
          type: 'mdxjsEsm',
          value: 'import HeliaMarkdownCallout from "@ambiqai/helia-ui/astro/Callout";',
          data: {
            estree: {
              type: 'Program', sourceType: 'module',
              body: [{
                type: 'ImportDeclaration',
                source: { type: 'Literal', value: '@ambiqai/helia-ui/astro/Callout' },
                specifiers: [{
                  type: 'ImportDefaultSpecifier',
                  local: { type: 'Identifier', name: 'HeliaMarkdownCallout' },
                }],
              }],
            },
          },
        });
        imported = true;
      }
      return {
        type: 'mdxJsxFlowElement',
        name: 'HeliaMarkdownCallout',
        attributes: [
          { type: 'mdxJsxAttribute', name: 'title', value: title },
          { type: 'mdxJsxAttribute', name: 'tone', value: tone },
        ],
        children,
      };
    },
  };
}
