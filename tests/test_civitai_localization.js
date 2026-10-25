// Exercise the shipped card handlers without contacting Civitai or deleting files.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const root=path.join(__dirname,'..');
const dictionary=JSON.parse(fs.readFileSync(path.join(root,'localizations/zh_Hant.json'),'utf8'));
const source=fs.readFileSync(path.join(root,'extensions/sd-webui-civitai-downloader/javascript/card_buttons.js'),'utf8')
    .replace('waitForUiLoadedAndSetup();','globalThis.cards={getButtonsHtml,processCard};');
function setup(locale) {
    const notices=[],confirms=[];
    const app={querySelector:()=>null};
    const ctx=vm.createContext({window:{localization:locale},opts:{},gradioApp:()=>app,
        alert:message=>notices.push(message),confirm:message=>{confirms.push(message);return false;}});
    vm.runInContext(source,ctx);
    const open={},remove={};
    const buttons={querySelector:selector=>selector==='.open-url-btn'?open:remove};
    let inserted=false;
    const card={dataset:{},insertAdjacentHTML:()=>{inserted=true;},
        querySelector:selector=>selector==='.custom-card-buttons'&&inserted?buttons:null};
    ctx.cards.processCard(card);
    return {ctx,open,remove,notices,confirms};
}
const event={stopPropagation(){},preventDefault(){}};
test('Civitai notifications and deletion confirmation use the selected dictionary',async()=>{
    const ui=setup(dictionary);
    await ui.open.onclick(event);await ui.remove.onclick(event);
    assert.equal(ui.notices[0],dictionary['Cannot determine the model path to open its information page.']);
    assert.equal(ui.confirms[0],dictionary['Are you sure you want to delete this model and its related files? This cannot be undone.']);
    ui.ctx.window.localization={};
    await ui.open.onclick(event);await ui.remove.onclick(event);
    assert.equal(ui.notices[1],'Cannot determine the model path to open its information page.');
    assert.equal(ui.confirms[1],'Are you sure you want to delete this model and its related files? This cannot be undone.');
});
test('card tooltip source stays English for normal localization and English reload',()=>{
    const ui=setup({});
    const html=ui.ctx.cards.getButtonsHtml();
    const engine=fs.readFileSync(path.join(root,'javascript/localization.js'),'utf8');
    ui.ctx.document={addEventListener(){},createTreeWalker:()=>({nextNode:()=>null})};
    ui.ctx.NodeFilter={SHOW_TEXT:4};vm.runInContext(engine,ui.ctx);
    for(const match of html.matchAll(/title="([^"]+)"/g)) {
        ui.ctx.window.localization=dictionary;ui.ctx.localization=dictionary;
        ui.ctx.node={title:match[1]};vm.runInContext('processNode(node)',ui.ctx);
        assert.equal(ui.ctx.node.title,dictionary[match[1]]);
        ui.ctx.window.localization={};ui.ctx.localization={};
        ui.ctx.node={title:match[1]};vm.runInContext('processNode(node)',ui.ctx);
        assert.equal(ui.ctx.node.title,match[1]);
    }
});
