// Production compiler and UI event smoke tests; minimal DOM, no browser/GPU.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const base = path.join(__dirname, '../extensions-builtin/sd_forge_json_builder/javascript');
const code = name => fs.readFileSync(path.join(base, name + '.js'), 'utf8');
function compiler() {
    const context = vm.createContext({});
    vm.runInContext(code('compile') + code('decompile'), context);
    return context;
}
function draft() {
    return {width:1024, height:768, bbox_coord_space:'abs', bbox_format:'xyxy',
        summary:{caption:'room (red)', meta:['anime'], canvas:'single'},
        items:[{label:'text',caption:'sign',text:'hello (world)',bbox:[32,48,256,160]}]};
}
test('compiled MOSS coordinates and parenthesis captions round-trip', () => {
    const c = compiler(), d = draft();
    const compiled = c.compilePrompt(JSON.stringify(d));
    const result = c.decompilePrompt(compiled, d.width, d.height);
    assert.equal(result.summary.caption, d.summary.caption);
    assert.equal(result.items[0].text, d.items[0].text);
    assert.equal(JSON.stringify(result.items[0].bbox), JSON.stringify(d.items[0].bbox));
    const recompiled=c.compilePrompt(JSON.stringify({...result,bbox_coord_space:'abs',bbox_format:'xyxy'}));
    assert.deepEqual(JSON.parse(JSON.stringify(c.decompilePrompt(recompiled,d.width,d.height))),JSON.parse(JSON.stringify(result)));
});
test('duplicate JSON keys are rejected before compiling', () => {
    assert.throws(() => compiler().compilePrompt('{"caption":"a","caption":"b"}'), /Duplicate JSON key/);
});
test('invalid dimensions, out-of-canvas boxes and missing literal text fail', () => {
    const c = compiler();
    for (const mutate of [d=>d.width=1000, d=>d.items[0].bbox=[-1,0,64,64], d=>delete d.items[0].text]) {
        const d = draft(); mutate(d);
        assert.throws(() => c.compilePrompt(JSON.stringify(d)));
    }
});
test('plain caption mode and malformed imports are handled without evaluation', () => {
    const c = compiler();
    assert.equal(c.compilePrompt('{"caption":"a (cat)"}'), 'a \\(cat\\)');
    assert.throws(() => c.decompilePrompt('globalThis.infected = true',1024,1024));
    assert.equal(c.infected, undefined);
    assert.throws(() => c.decompilePrompt('{"items":[]}',NaN,NaN));
});

// Record the real controls/events without replacing compiler/UI functions.
class Element {
    constructor(tag='div') {
        this.tagName=tag.toUpperCase(); this.children=[]; this.events={}; this.attrs={};
        this.style={setProperty(){}}; this.dataset={}; this.value=''; this.className='';
        this.clientWidth=900; this.clientHeight=600; this.scrollHeight=20;
        this.classList={add:n=>this.classes.add(n),remove:n=>this.classes.delete(n)};
        this.classes=new Set(); this.isConnected=true;
    }
    appendChild(x) {this.children.push(x);return x;}
    append(...xs) {xs.forEach(x=>this.appendChild(x));}
    setAttribute(k,v) {this.attrs[k]=v;if(k==='class')this.className=v;}
    addEventListener(k,fn) {this.events[k]=fn;}
    getBoundingClientRect(){return {width:800,height:500,left:0,top:0};}
    querySelectorAll(selector) {
        return this.children.flatMap(child => {
            if (!(child instanceof Element)) return [];
            const matches=selector.startsWith('.') && child.className.split(' ').includes(selector.slice(1));
            return [...(matches?[child]:[]),...child.querySelectorAll(selector)];
        });
    }
    querySelector(selector) {return this.querySelectorAll(selector)[0] || null;}
    focus(){} remove(){} click(){this.events.click?.({});}
}
function frontend(translation={}) {
    const elements = new Map();
    const tab=new Element(); elements.set('tab_sd-forge-json-builder',tab);
    for(const id of ['txt2img_prompt','txt2img_width','txt2img_height','img2img_prompt','img2img_width','img2img_height']) {
        const control=new Element(); control.input=new Element(id.endsWith('prompt')?'textarea':'input');
        control.querySelector=()=>control.input; elements.set(id,control);
    }
    const updates=[], switches=[], alerts=[];
    const document={getElementById:id=>elements.get(id)||null,createElement:t=>new Element(t),createTextNode:t=>({textContent:t}),body:new Element()};
    const context=vm.createContext({document,window:{localization:translation},
        onUiLoaded:fn=>fn(), requestAnimationFrame:fn=>fn(),
        ResizeObserver:class{observe(){}unobserve(){}},getComputedStyle:()=>({borderTopWidth:'0',borderBottomWidth:'0'}),
        updateInput:el=>updates.push(el),switch_to_txt2img:()=>switches.push('txt2img'),switch_to_img2img:()=>switches.push('img2img'),
        alert:e=>alerts.push(String(e)),prompt:()=>null,console});
    vm.runInContext(code('compile')+code('decompile')+code('main'),context);
    return {tab,elements,updates,switches,alerts,context};
}
test('disabled/missing Builder tab does not break other frontend scripts',()=> {
    const context=vm.createContext({onUiLoaded:fn=>fn(),document:{getElementById:()=>null}});
    assert.doesNotThrow(()=>vm.runInContext(code('main'),context));
});
test('dynamic labels translate while schema radio values stay English',()=> {
    const ui=frontend({'Add Item':'新增項目','object':'物件'});
    assert.equal(ui.tab.querySelector('.jb-add').children[0].textContent,'新增項目');
    const object=ui.tab.querySelectorAll('.jb-radio').find(x=>x.children.some(c=>c.value==='object'));
    assert.equal(object.children[0].value,'object');
    assert.equal(object.children[1].textContent,'物件');
    vm.runInContext(code('main'),ui.context);
    assert.equal(ui.tab.querySelectorAll('.jb-root').length,1);
});
test('loaded layout sends prompt/dimensions to both Gradio tabs via input updates',()=> {
    const ui=frontend();
    ui.tab.querySelector('.jb-load-prompt').click();
    ui.tab.querySelector('.jb-dialog-input').value=JSON.stringify(draft());
    ui.tab.querySelector('.jb-dialog-load').click();
    ui.tab.querySelector('.jb-send-t2i').click();
    ui.tab.querySelector('.jb-send-i2i').click();
    assert.deepEqual(ui.switches,['txt2img','img2img']);
    assert.equal(ui.updates.length,6);
    assert.equal(ui.elements.get('txt2img_width').input.value,1024);
    assert.equal(ui.elements.get('img2img_height').input.value,768);
    assert.match(ui.elements.get('txt2img_prompt').input.value,/abs_centralized/);
    assert.deepEqual(ui.alerts,[]);
});
test('invalid import keeps the dialog open and displays a localized error',()=> {
    const ui=frontend({'Unable to load layout: ':'無法載入構圖：'});
    ui.tab.querySelector('.jb-load-prompt').click();
    ui.tab.querySelector('.jb-dialog-input').value='invalid json';
    ui.tab.querySelector('.jb-dialog-load').click();
    assert.equal(ui.tab.querySelector('.jb-modal').classes.has('jb-hidden'),false);
    assert.match(ui.tab.querySelector('.jb-error').textContent,/^無法載入構圖：/);
});
test('import/edit/export preserves optional MOSS tags and palettes',()=> {
    const ui=frontend(), d=draft();
    d.summary.palette=['red','gold']; d.items[0].tags=['sign']; d.items[0].palette=['blue'];
    ui.tab.querySelector('.jb-load-prompt').click();
    ui.tab.querySelector('.jb-dialog-input').value=JSON.stringify(d);
    ui.tab.querySelector('.jb-dialog-load').click();
    ui.tab.querySelector('.jb-send-t2i').click();
    const exported=compiler().decompilePrompt(ui.elements.get('txt2img_prompt').input.value,1024,768);
    assert.equal(JSON.stringify(exported.summary.palette),JSON.stringify(d.summary.palette));
    assert.equal(JSON.stringify(exported.items[0].tags),JSON.stringify(d.items[0].tags));
    assert.equal(JSON.stringify(exported.items[0].palette),JSON.stringify(d.items[0].palette));
});
test('imports reject invalid dimensions and out-of-canvas coordinates before editing',()=> {
    const c=compiler();
    for (const mutate of [d=>d.width=1000,d=>d.items[0].bbox=[-1,0,64,64]]) {
        const d=draft(); mutate(d);
        assert.throws(()=>c.decompilePrompt(JSON.stringify(d),1024,768));
    }
});
