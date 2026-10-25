// Use the shipped localization engine on actual ImageStitch HTML text nodes.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const root=path.join(__dirname,'..');
const dictionary=JSON.parse(fs.readFileSync(path.join(root,'localizations/zh_Hant.json'),'utf8'));
const source=fs.readFileSync(path.join(root,'extensions-builtin/sd_forge_image_stitch/scripts/image_stitch.py'),'utf8');
const engine=fs.readFileSync(path.join(root,'javascript/localization.js'),'utf8');
function context(locale) {
    return vm.createContext({localization:locale,window:{localization:locale},document:{addEventListener(){}}});
}
function messages(name) {
    const html=source.match(new RegExp(name+' = f"""([\\s\\S]*?)"""'))[1];
    return html.split(/<[^>]*>/g).filter(s=>s.trim());
}
function render(context,parts) {
    return parts.map(text=>{
        const node={textContent:text,parentElement:{nodeName:'P'}};
        context.node=node;
        vm.runInContext('processTextNode(node)',context);
        return node.textContent;
    });
}
test('both ImageStitch help blocks translate actual HTML-split phrases',()=>{
    const c=context(dictionary); vm.runInContext(engine,c);
    for(const name of ['t2i_info','i2i_info']) {
        const parts=messages(name),translated=render(c,parts);
        parts.forEach((part,index)=>{
            if(part.trim()==='Wan 2.2 I2V') return; // model identifier
            assert.ok(dictionary[part.trim()], 'missing text node: '+part.trim());
            assert.equal(translated[index],dictionary[part.trim()]);
        });
        assert.match(translated.join(''),/提示：請使用「上傳圖片」加入圖片/);
    }
});
test('English UI reload leaves original ImageStitch text intact after Chinese',()=>{
    const c=context(dictionary); vm.runInContext(engine,c);
    render(c,messages('t2i_info'));
    c.localization={}; c.window.localization=c.localization;
    for(const name of ['t2i_info','i2i_info']) {
        const original=messages(name);
        assert.deepEqual(render(c,original),original);
    }
});
