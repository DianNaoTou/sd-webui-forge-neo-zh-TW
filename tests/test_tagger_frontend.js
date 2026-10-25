// Execute WD14's shipped browser code in a minimal DOM, without model downloads.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../extensions/stable-diffusion-webui-tagger-fork/javascript/tagger.js'), 'utf8');
function load(root, extra = {}) {
    const context = vm.createContext({
        onUiLoaded() {}, gradioApp: () => root,
        setTimeout, clearTimeout, clearInterval, Event, ...extra,
        document: {addEventListener() {}, ...extra.document, querySelector() {throw Error("DOM elements are not selectors");}},
    });
    vm.runInContext(source, context);
    return context;
}
test('existing WD14 controls resolve immediately without invalid querySelector', async () => {
    const element = {};
    const context = load({querySelector: () => element});
    assert.equal(await context.waitQuerySelector('#keep-tags'), element);
});
test('lazy Gradio tabs resolve after mounting', async () => {
    let element = null, notify, disconnected = false;
    const context = load({querySelector: () => element}, {
        MutationObserver: class {
            constructor(callback) {notify = callback;}
            observe() {}
            disconnect() {disconnected = true;}
        },
    });
    const pending = context.waitQuerySelector('#keep-tags');
    element = {};
    notify();
    assert.equal(await pending, element);
    assert.equal(disconnected, true);
});
test('missing Gradio controls reject on timeout', async () => {
    let disconnected = false;
    const context = load({querySelector: () => null}, {
        MutationObserver: class {observe() {} disconnect() {disconnected = true;}}
    });
    await assert.rejects(context.waitQuerySelector('#missing', 1), /timeout/);
    assert.equal(disconnected, true);
});
test('tag clicks update prompt state once and keep regex characters literal', () => {
    const events = [];
    const textarea = {value: '', dispatchEvent: event => events.push(event.type)};
    const element = {querySelector: () => textarea};
    const context = load({}, {document: {getElementById: () => element}});
    context.tag_clicked('pose (test)+', true);
    context.tag_clicked('pose (test)+', true);
    assert.equal(textarea.value, 'pose (test)+');
    assert.deepEqual(events, ['input', 'blur']);
});
