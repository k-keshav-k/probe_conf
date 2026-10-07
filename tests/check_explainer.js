// Regression check for the explainer's recorded-data position control.
// This executes page logic with a small DOM stub; it does not test browser layout.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const page = fs.readFileSync(path.join(__dirname, '../report/EXPLAINER.html'), 'utf8');
const script = page.match(/<script>([\s\S]*?)<\/script>/)[1];
const nodes = new Map();
function element() {
  return {innerHTML: '', textContent: '', value: '', children: [],
    classList: {toggle() {}}, addEventListener() {},
    appendChild(child) {this.children.push(child);}};
}
const document = {
  querySelector(selector) {
    if (!nodes.has(selector)) nodes.set(selector, element());
    return nodes.get(selector);
  },
  createElement: element,
};
const context = {document, location: {search: ''}, URLSearchParams};
vm.createContext(context);
vm.runInContext(script + '\nglobalThis.testPage = {D, state, refresh};', context);
const {D, state, refresh} = context.testPage;
let cases = 0;
for (const example of ['correct', 'wrong']) {
  for (const pos of ['last', 'ans']) {
    for (let layer = 0; layer < 28; layer++) {
      Object.assign(state, {example, pos, layer, step: 0});
      refresh();
      const maximum = D.gate[example][pos].top[layer][0].toFixed(2);
      assert(nodes.get('#step-data').innerHTML.includes(`<b>${maximum}</b>`));
      assert(!nodes.get('#stats-groups').innerHTML.includes('NaN'));
      cases++;
    }
  }
}
console.log(`Explainer logic: ${cases} layer/position/example combinations pass.`);
