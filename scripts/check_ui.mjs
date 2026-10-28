/* Real-browser checks, distilled from .build/ reader/home CDP scripts.
   Node 22+ and Chrome; screenshots/reports stay in .build/ui. No npm dependencies. */
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { readFile, writeFile, mkdir, stat, rm } from 'node:fs/promises';
import path from 'node:path';

const option = name => process.argv.find(a => a.startsWith(name + '='))?.slice(name.length + 1);
const site = path.resolve(option('--site-dir') || '.build/site');
const artifacts = path.resolve(option('--artifacts') || '.build/ui');
const serverTransfers=[];
await mkdir(artifacts, { recursive: true });
const mime = {'.html':'text/html','.css':'text/css','.js':'text/javascript','.json':'application/json','.svg':'image/svg+xml','.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg','.JPG':'image/jpeg','.woff2':'font/woff2','.pdf':'application/pdf'};
const server = createServer(async (req,res) => {
  try {
    let file = path.resolve(site,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname));
    if(!file.startsWith(site+path.sep) && file!==site) throw Error('Invalid path');
    if((await stat(file)).isDirectory()) file=path.join(file,'index.html');
    const data=await readFile(file);
    serverTransfers.push({path:new URL(req.url,'http://localhost').pathname,bodyBytes:data.length});
    res.writeHead(200,{'Content-Type':(mime[path.extname(file)] || 'application/octet-stream')+'; charset=utf-8'});res.end(data);
  } catch {res.writeHead(404);res.end('Not found');}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const root='http://127.0.0.1:'+server.address().port;
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const assert=(test,message)=>{if(!test)throw Error(message);};
let browser,socket,target,cdp=option('--cdp');
const checks=[],errors=[],network=new Map();
let homeTransfer;
try {
  if(!cdp) {
    const profile=path.join(artifacts,'chrome-'+process.pid+'-'+Date.now());
    await mkdir(profile,{recursive:true});
    const executable=process.env.CHROME_BIN || (process.platform==='win32'?'C:/Program Files/Google/Chrome/Application/chrome.exe':'google-chrome');
    browser=spawn(executable,['--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-port=0','--user-data-dir='+profile,...(process.platform==='linux'?['--no-sandbox','--disable-dev-shm-usage']:[]),'about:blank'],{stdio:['ignore','ignore','pipe'],windowsHide:true});
    let launchError,launchOutput='';browser.on('error',error=>{launchError=error;});
    browser.stderr.on('data',data=>{launchOutput=(launchOutput+data.toString()).slice(-16000);});
    for(let i=0;i<300;i++){
      if(launchError || browser.exitCode!==null || browser.signalCode!==null)break;
      try{cdp='http://127.0.0.1:'+(await readFile(path.join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0];break;}catch{await delay(100);}
    }
    assert(cdp,'Chrome failed to start within 30s: '+executable+'\n'+(launchError?.message || (browser.exitCode!==null?'Exit code: '+browser.exitCode:browser.signalCode || 'DevToolsActivePort was not created'))+'\n'+launchOutput);
  }
  target=await(await fetch(cdp+'/json/new?about:blank',{method:'PUT'})).json();
  socket=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
  let id=0;const pending=new Map();
  socket.onmessage=({data})=>{
    const message=JSON.parse(data);
    if(pending.has(message.id)){const task=pending.get(message.id);pending.delete(message.id);message.error?task.reject(Error(JSON.stringify(message.error))):task.resolve(message.result);}
    if(message.method==='Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.text);
    if(message.method==='Network.requestWillBeSent') network.set(message.params.requestId,{url:message.params.request.url,encodedBytes:0});
    if(message.method==='Network.loadingFinished' && network.has(message.params.requestId)) network.get(message.params.requestId).encodedBytes=message.params.encodedDataLength;
  };
  const send=(method,params={})=>new Promise((resolve,reject)=>{const key=++id;pending.set(key,{resolve,reject});socket.send(JSON.stringify({id:key,method,params}));});
  const evaluate=async(fn,args=[])=>{
    const result=await send('Runtime.evaluate',{expression:'('+fn.toString()+')(...'+JSON.stringify(args)+')',returnByValue:true,awaitPromise:true});
    if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  await send('Page.enable');await send('Runtime.enable');await send('Network.enable');
  const capture=async name=>{const image=await send('Page.captureScreenshot',{format:'png'});await writeFile(path.join(artifacts,name+'.png'),Buffer.from(image.data,'base64'));};
  async function load(route,width,theme='default') {
    await send('Network.setCacheDisabled',{cacheDisabled:!homeTransfer && route==='/'});
    network.clear();
    serverTransfers.length=0;
    await send('Emulation.setDeviceMetricsOverride',{width,height:width===1280?800:width===1920?1080:900,deviceScaleFactor:1,mobile:width<650});
    await send('Emulation.setTouchEmulationEnabled',{enabled:width<650});
    await send('Emulation.setEmulatedMedia',{media:'screen',features:[{name:'prefers-reduced-motion',value:'no-preference'}]});
    await send('Page.navigate',{url:root+route});
    await delay(150);
    await evaluate(()=>new Promise((resolve,reject)=>{
      const start=Date.now();
      const poll=()=>{
        if(document.querySelector('article') && document.readyState==='complete')return resolve();
        if(Date.now()-start>20000)return reject(Error('Page load timeout'));
        setTimeout(poll,100);
      };poll();
    }));
    await evaluate(theme=>document.querySelector('input[data-md-color-scheme="'+theme+'"]').click(),[theme]);
    await evaluate(()=>new Promise((resolve,reject)=>{
      const start=Date.now();const poll=()=>{
        if(!document.querySelector('.arithmatex'))return resolve();
        if(window.MathJax?.startup?.document)return Promise.resolve(MathJax.startup.promise).then(()=>document.fonts.ready).then(resolve,reject);
        if(Date.now()-start>20000)return reject(Error('MathJax unavailable'));
        setTimeout(poll,150);
      };poll();
    }));
    await delay(100);
    const state=await evaluate(()=>{
      const article=document.querySelector('article'),toc=document.querySelector('.mobile-toc'),aside=document.querySelector('.md-sidebar--secondary');
      const visible=n=>n && getComputedStyle(n).display!=='none';
      const css=getComputedStyle(document.body);
      const rgb=name=>{let hex=css.getPropertyValue(name).trim().slice(1);if(hex.length===3)hex=[...hex].map(c=>c+c).join('');return [0,2,4].map(i=>parseInt(hex.slice(i,i+2),16)/255);};
      const luminance=color=>color.map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((sum,v,i)=>sum+v*[.2126,.7152,.0722][i],0);
      const ratio=(a,b)=>{const values=[luminance(rgb(a)),luminance(rgb(b))].sort((a,b)=>b-a);return (values[0]+.05)/(values[1]+.05);};
      const walker=document.createTreeWalker(article,NodeFilter.SHOW_TEXT);const raw=[];
      while(walker.nextNode()){const n=walker.currentNode;if(!n.parentElement.closest('code,pre,.arithmatex') && /\$\$|\$[^\n$]+\$/.test(n.textContent))raw.push(n.textContent.slice(0,100));}
      return {width:innerWidth,scrollWidth:document.documentElement.scrollWidth,bodyFont:parseFloat(getComputedStyle(article).fontSize),
        rawMath:raw,math:document.querySelectorAll('.arithmatex').length,rendered:document.querySelectorAll('mjx-container').length,mathErrors:[...document.querySelectorAll('mjx-merror')].map(n=>n.textContent),
        mobileToc:visible(toc),tocOpen:toc?.open,tocHeight:toc?.getBoundingClientRect().height,tocAfterTitle:toc?toc.getBoundingClientRect().top>=article.querySelector('h1').getBoundingClientRect().bottom:false,
        desktopToc:visible(aside),tocLeft:aside?aside.getBoundingClientRect().right<=article.getBoundingClientRect().left+1:false,
        rows:[...article.querySelectorAll('.article-row')].map(n=>n.dataset.section),
        auxiliaryFonts:[...article.querySelectorAll('.article-row__meta,.article-row__description,.article-tag,.hero-caption,.atlas-caption,.atlas-hint,.visual-caption')].filter(visible).map(n=>parseFloat(getComputedStyle(n).fontSize)),
        images:[...article.querySelectorAll('img')].map(img=>({dimensions:img.hasAttribute('width') && img.hasAttribute('height'),loaded:img.complete && img.naturalWidth>0,lazy:img.loading==='lazy'})),
        contrast:[ratio('--md-default-fg-color','--md-default-bg-color'),ratio('--md-default-fg-color--light','--note-surface'),ratio('--md-typeset-a-color','--md-default-bg-color'),ratio('--button-ink','--button-fill'),ratio('--button-ink','--button-hover')],
        brand:document.querySelector('.header-home-link').href,loadedMath:!!document.querySelector('script[src*="tex-mml-chtml"]')};
    });
    assert(state.width===width && state.scrollWidth<=width+1,route+' overflow: '+JSON.stringify(state));
    assert(state.bodyFont>=16 && state.bodyFont<=18,'Body font outside reading range');
    assert(!state.rawMath.length && !state.mathErrors.length && state.math===state.rendered,'Unrendered math: '+route);
    assert(state.auxiliaryFonts.every(size=>size>=12),'Small reading labels: '+route);
    assert(state.contrast.every(ratio=>ratio>=4.5),'Reading colors below WCAG AA: '+JSON.stringify(state.contrast));
    assert(state.images.every(img=>img.dimensions && (img.loaded || img.lazy)),'Image dimensions/decoding: '+route);
    assert(state.brand===root+'/','Header name must return home');
    const inlineToc=width<960;
    if(state.tocHeight!==undefined)assert(inlineToc?state.mobileToc && !state.tocOpen && state.tocAfterTitle && state.tocHeight<=64 && !state.desktopToc:!state.mobileToc && state.desktopToc && state.tocLeft,'Outline layout: '+JSON.stringify(state));
    checks.push({route,width,theme,...state});
    if(route==='/' && !homeTransfer) {
      await delay(500);
      homeTransfer={environment:'local HTTP, uncompressed, cache disabled',resources:[...network.values()]};
      homeTransfer.encodedBytes=homeTransfer.resources.reduce((total,r)=>total+r.encodedBytes,0);
      homeTransfer.sameOriginResources=[...serverTransfers];
      homeTransfer.sameOriginBodyBytes=serverTransfers.reduce((total,r)=>total+r.bodyBytes,0);
      homeTransfer.measurementWindow='initial load plus 500ms; server bytes include worker requests not fully reported by the main-frame CDP target';
      assert(!homeTransfer.resources.some(r=>new URL(r.url).pathname.endsWith('.pdf')),'Home unexpectedly downloads PDFs');
    }
    console.log('PASS',width,theme,route,'math',state.rendered);
    return state;
  }
  if(option('--only')!=='interaction') {
  for(const width of [320,390,768,1280,1440,1920])for(const theme of ['default','slate'])for(const route of ['/','/papers/','/papers/library/','/translations/','/translations/cs146s/','/translations/robotic-manipulation/','/deep-learning/','/engineering/','/journal/','/about/','/deep-learning/Linear_Regression/','/papers/orion/','/translations/robotic-manipulation/pick-and-place/','/papers/lit/']){
    const state=await load(route,width,theme);
    // The keyword canvas was replaced by accessible links into real paper notes.
    // Preserve discovery, touch scrolling, anchors, course entry and sticky branding.
    if(route==='/'){
      assert(await evaluate(()=>!document.querySelector('.home-recent,.recent-notes,.home-threads,.reading-note') && document.querySelectorAll('.topic-card').length===3),'Homepage collection coverage changed');
      assert(!state.loadedMath,'MathJax must not load on home');
      assert(await evaluate(()=>document.querySelectorAll('.research-paths > a').length===4 && !document.querySelector('canvas')),'Research paths must be usable without a canvas');
      assert(await evaluate(()=>!!document.querySelector('.featured-course a.md-button') && document.querySelector('.featured-course h2').innerText.includes('Robotic Manipulation')),'Course title/links did not render');
      assert(await evaluate(()=>document.getElementById('_2')?.closest('.featured-course') && document.getElementById('_3')?.closest('.featured-course')),'Home bookmark anchors changed');
      if([320,390,1280,1440,1920].includes(width))await capture('home-'+width+'-'+theme);
      if(width===320 && theme==='default') {
        const drag=await evaluate(()=>{const r=document.querySelector('.home-intro').getBoundingClientRect();return {x:r.left+r.width/2,y:r.bottom-20};});
        await send('Page.bringToFront');
        await send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[drag]});
        for(const offset of [35,70,110]){await send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:drag.x,y:drag.y-offset}]});await delay(40);}
        await send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await delay(300);
        assert(await evaluate(()=>scrollY>30),'Homepage must preserve vertical touch scrolling');
      }
      await evaluate(()=>document.querySelector('.featured-course').scrollIntoView({block:'center'}));await delay(300);
      assert(await evaluate(()=>{const a=document.querySelector('.header-home-link'),r=a.getBoundingClientRect(),s=getComputedStyle(a.parentElement);return s.opacity==='1' && s.pointerEvents!=='none' && r.width>0 && r.left>=0 && r.top>=0;}),'Homepage brand must remain clickable after scrolling');
      if(width===320 || width===1440)await capture('home-course-'+width+'-'+theme);
    }
    if([390,1440].includes(width))await capture(route.replaceAll('/','-').replace(/^-|-$/g,'')+'-'+width+'-'+theme);
    if(route==='/papers/library/'){
      const library=await evaluate(()=>{const content=document.querySelector('.md-content'),r=content.getBoundingClientRect();return {center:Math.abs((r.left+r.right)/2-innerWidth/2),width:r.width,emptySidebar:[...document.querySelectorAll('.md-sidebar')].some(n=>{const b=n.getBoundingClientRect();return getComputedStyle(n).display!=='none' && b.right>1 && b.left<innerWidth;}),rows:document.querySelectorAll('.resource-row').length};});
      assert(library.rows>0 && !library.emptySidebar && (width<960 || (library.center<15 && library.width<=1041)),'PDF library centering: '+JSON.stringify(library));
      if(width===320 || width===1440)await capture('library-'+width+'-'+theme);
    }
    if(route==='/papers/') {
      assert(await evaluate(()=>{const all=[...document.querySelectorAll('.article-list .article-row')].map(a=>a.getAttribute('href'));const quick=[...document.querySelectorAll('.collection-directory a')].map(a=>a.getAttribute('href'));return all.length>0 && all.length===quick.length && all.every(href=>quick.includes(href));}),'Quick paper directory must cover every public paper note');
      if(width===1440 && theme==='default') {
        await evaluate(()=>document.querySelector('.collection-directory summary').focus());
        await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,text:'\r'});
        await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});
        assert(await evaluate(()=>document.querySelector('.collection-directory').open),'Paper directory keyboard disclosure failed');
        await capture('paper-quick-directory-1440');
      }
    }
    if(route==='/papers/orion/' && width===1440 && theme==='default')await capture('paper-outline-1440');
    if(route==='/translations/'){
      assert(await evaluate(()=>{const cards=[...document.querySelectorAll('.translation-shelf')];return cards.length>=2 && cards.every(a=>a.querySelector('.shelf-cover') && a.querySelector('.shelf-copy')) && ['robotic-manipulation/','cs146s/'].every(href=>cards.some(a=>a.getAttribute('href')===href));}),'Translation shelves must link complete course cards');
    }
    if(route==='/translations/cs146s/'){
      assert(await evaluate(()=>{const links=new Set([...document.querySelectorAll('.course-week__links a')].map(a=>a.getAttribute('href')));return [1,2,3,4,5].every(week=>['assignment','writeup'].every(kind=>links.has('week'+week+'/'+kind+'/'))) && !!document.querySelector('.course-preface[href="overview/"]');}),'CS146S course must link its README and five assignment/template pairs');
      assert(await evaluate(()=>[...document.querySelectorAll('.course-week')].length===5 && [...document.querySelectorAll('.course-week')].every(week=>{
        const links=week.querySelector('.course-week__links'),buttons=[...links.querySelectorAll('a')];
        const container=links.getBoundingClientRect(),article=week.closest('article').getBoundingClientRect(),a=buttons[0]?.getBoundingClientRect(),b=buttons[1]?.getBoundingClientRect();
        return !week.querySelector('p') && buttons.length===2 && Math.abs(container.width-article.width)<2 && Math.abs(a.width-b.width)<2 && Math.abs(a.left-container.left)<2 && Math.abs(b.right-container.right)<2 && Math.abs(a.top-b.top)<2;
      })),'CS146S weekly links must fill the row equally without descriptions');
    }
    if(route.includes('pick-and-place')){
      if(width<960){
        await evaluate(()=>document.querySelector('.mobile-toc summary').focus());await send('Page.bringToFront');
        await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,text:'\r'});
        await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});
        const expanded=await evaluate(()=>({open:document.querySelector('.mobile-toc').open,scroll:getComputedStyle(document.querySelector('.mobile-toc nav')).overflowY,focus:getComputedStyle(document.activeElement).outlineWidth}));
        assert(expanded.open && !['auto','scroll'].includes(expanded.scroll) && expanded.focus!=='0px','Outline keyboard/no-inner-scroll failed');
        await evaluate(()=>document.querySelector('.mobile-toc a[href="#33"]').click());await delay(180);
        assert(await evaluate(()=>location.hash==='#33' && !document.querySelector('.mobile-toc').open && document.activeElement.id==='33' && !!document.querySelector('.mobile-toc a[href="#33"][aria-current="location"]')),'Outline anchor/close/current-position failed');
        if(width===390 && theme==='default')await capture('reader-anchor-390');
      }
      const math=await evaluate(()=>{const n=[...document.querySelectorAll('div.arithmatex')].find(n=>n.scrollWidth>n.clientWidth+10);return n?{overflow:getComputedStyle(n).overflowX}:null;});
      if(width<650)assert(math?.overflow==='auto','Wide math must scroll inside article');
    }
    if(route==='/papers/lit/'){
      const table=await evaluate(()=>{const t=document.querySelector('article table'),p=t?.closest('.md-typeset__scrollwrap');return {count:document.querySelectorAll('article table').length,focusable:p?.tabIndex===0,overflow:p?getComputedStyle(p).overflowX:null};});
      assert(table.count>0 && table.focusable && table.overflow==='auto','Accessible table scroll failed');
    }
  }
  await load('/translations/robotic-manipulation/spatial-algebra/',390);
  const cs146sArticles=['overview/',...[1,2,3,4,5].flatMap(week=>['assignment','writeup'].map(kind=>'week'+week+'/'+kind+'/'))];
  for(const article of cs146sArticles)for(const width of [320,959,960,1440])for(const theme of ['default','slate']) {
    await load('/translations/cs146s/'+article,width,theme);
    if(width<960) {
      const inline=await evaluate(()=>{const toc=document.querySelector('.mobile-toc'),article=document.querySelector('article'),r=toc.getBoundingClientRect(),a=article.getBoundingClientRect();return {inside:r.left>=a.left-1 && r.right<=a.right+1,links:toc.querySelectorAll('a').length};});
      assert(inline.inside && inline.links>0,'CS146S mobile outline must sit inside the article: '+JSON.stringify(inline));
      await evaluate(()=>document.querySelector('.mobile-toc summary').focus());await send('Page.bringToFront');
      await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,text:'\r'});
      await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});
      assert(await evaluate(()=>document.querySelector('.mobile-toc').open),'CS146S mobile outline must open by keyboard');
      const hash=await evaluate(()=>{const link=document.querySelector('.mobile-toc a');const hash=link.hash;link.click();return hash;});await delay(180);
      assert(await evaluate(hash=>location.hash===hash && !document.querySelector('.mobile-toc').open && document.activeElement.id===decodeURIComponent(hash.slice(1)),[hash]),'CS146S mobile outline navigation failed');
    } else {
      const hash=await evaluate(()=>{const link=document.querySelector('.md-sidebar--secondary a[href^="#"]');const hash=link.hash;link.click();return hash;});
      assert(await evaluate(hash=>location.hash===hash && !!document.getElementById(decodeURIComponent(hash.slice(1))),[hash]),'CS146S left outline navigation failed');
    }
    if(['overview/','week1/assignment/'].includes(article) && [320,1440].includes(width) && theme==='default')await capture('cs146s-'+article.replaceAll('/','-')+width);
  }
  await load('/translations/robotic-manipulation/introduction/',390);
  await evaluate(()=>document.querySelector('.plate-sequence img').scrollIntoView());
  await evaluate(()=>Promise.all([...document.querySelectorAll('.plate-sequence img')].map(img=>img.decode())));
  await load('/journal/',390);await load('/papers/library/',390);
  assert(!await evaluate(()=>!!document.querySelector('article table')),'PDF library must be a list');
  const papers=(await (await fetch(root+'/content-index.json')).json()).filter(p=>p.section==='papers');
  for(const paper of papers){
    await load('/'+paper.url,1440);
    assert(await evaluate(method=>document.querySelector('.md-sidebar--secondary [data-md-component="toc"] a')?.innerText.trim()===method,[paper.method]),'Paper method name missing: '+paper.url);
    assert(await evaluate(()=>{const toc=document.querySelector('.md-sidebar--secondary'),links=new Set([...toc.querySelectorAll('a')].map(a=>a.hash));return getComputedStyle(toc).display!=='none' && [...document.querySelectorAll('article h1,article h2,article h3,article h4')].slice(1).every(h=>links.has('#'+encodeURIComponent(h.id)) || links.has('#'+h.id));}),'Incomplete left outline: '+paper.url);
  }
  }
  await load('/',1440);
  // Exercise the upstream switch itself as well as rendering both schemes.
  await evaluate(()=>document.querySelector('label[for="__palette_1"]').click());await delay(100);
  assert(await evaluate(()=>document.body.dataset.mdColorScheme==='slate'),'Theme control failed to select dark mode');
  await evaluate(()=>document.querySelector('label[for="__palette_0"]').click());await delay(100);
  assert(await evaluate(()=>document.body.dataset.mdColorScheme==='default'),'Theme control failed to select light mode');
  await send('Page.bringToFront');
  for(const theme of ['default','slate']) {
    await evaluate(theme=>document.querySelector('input[data-md-color-scheme="'+theme+'"]').click(),[theme]);await delay(100);
    const button=await evaluate(()=>{const n=document.querySelector('.hero-actions .md-button--primary'),r=n.getBoundingClientRect();window.__buttonFrames=new Promise(resolve=>{const frames=[],start=performance.now();const sample=()=>{const s=getComputedStyle(n);frames.push({background:s.backgroundColor,image:s.backgroundImage,color:s.color});performance.now()-start<350?requestAnimationFrame(sample):resolve(frames);};sample();});return {x:r.left+r.width/2,y:r.top+r.height/2};});
    await send('Input.dispatchMouseEvent',{type:'mouseMoved',...button});
    const frames=await evaluate(()=>window.__buttonFrames);
    assert(frames.length>3 && frames.every(f=>f.background!=='rgba(0, 0, 0, 0)' && f.image==='none' && f.color!==f.background),'Primary button must keep an opaque, contrasting surface during hover');
    await evaluate(()=>document.querySelector('.hero-actions .md-button--primary').focus());
    assert(await evaluate(()=>parseFloat(getComputedStyle(document.activeElement).outlineWidth)>=2),'Primary action keyboard focus is missing');
    await send('Input.dispatchMouseEvent',{type:'mouseMoved',x:10,y:80});
  }
  await send('Emulation.setEmulatedMedia',{media:'screen',features:[{name:'prefers-reduced-motion',value:'reduce'}]});
  await send('Page.navigate',{url:root+'/'});await delay(700);
  assert(await evaluate(()=>!document.querySelector('canvas') && [...document.querySelectorAll('article *')].every(n=>getComputedStyle(n).animationName==='none')),'Reduced motion must keep reading paths still');
  await evaluate(()=>{document.getElementById('__search').click();const q=document.querySelector('[data-md-component=search-query]');q.focus();q.value='空间代数';q.dispatchEvent(new KeyboardEvent('keyup',{bubbles:true,key:'数'}));});
  await evaluate(()=>new Promise((resolve,reject)=>{const start=Date.now();const poll=()=>{if(document.querySelector('[data-md-component=search-result]').innerText.includes('附录 A'))return resolve();if(Date.now()-start>8000)return reject(Error('Search failed'));setTimeout(poll,150);};poll();}));
  await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
  await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
  for(const [old,next] of [['attention/python','engineering/tensor-inplace'],['attention/Linear_Algebra','engineering/tensor-operations'],['meeting/image-to-arkts','engineering/image-to-arkts']]){
    await load('/'+next+'/',1440);
    const anchor=await evaluate(()=>document.querySelector('article h2').id);
    await send('Page.navigate',{url:root+'/'+old+'/?bookmark=ui#'+anchor});await delay(800);
    assert(await evaluate((route,anchor)=>location.pathname==='/'+route+'/' && location.search==='?bookmark=ui' && decodeURIComponent(location.hash)==='#'+anchor,[next,anchor]),'Legacy bookmark redirect failed: '+old);
  }
  await evaluate(()=>document.querySelector('.header-home-link').focus());
  await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,text:'\r'});
  await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});
  await delay(600);
  assert(await evaluate(()=>location.pathname==='/' && !!document.querySelector('.research-paths')),'Header keyboard link failed');
  await send('Emulation.setScriptExecutionDisabled',{value:true});await send('Page.navigate',{url:root+'/'});await delay(600);
  assert(await evaluate(()=>document.querySelectorAll('.research-paths > a').length===4 && document.querySelectorAll('.topic-card').length===3 && document.querySelector('.research-paths > a').getBoundingClientRect().height>=44 && document.querySelectorAll('.noscript-nav a').length===7),'No-script fallback failed');
  await send('Emulation.setDeviceMetricsOverride',{width:390,height:900,deviceScaleFactor:1,mobile:true});
  await send('Page.navigate',{url:root+'/translations/robotic-manipulation/introduction/'});await delay(650);
  assert(await evaluate(()=>getComputedStyle(document.querySelector('.mobile-toc')).display!=='none' && !document.querySelector('.mobile-toc').open),'Native no-script outline failed');
  await send('Emulation.setScriptExecutionDisabled',{value:false});
  await load('/engineering/tensor-inplace/',390);
  await send('Browser.grantPermissions',{origin:root,permissions:['clipboardReadWrite','clipboardSanitizedWrite']});
  await evaluate(()=>document.querySelector('.md-clipboard').scrollIntoView({block:'center'}));
  const copy=await evaluate(()=>{const b=document.querySelector('.md-clipboard'),r=b.getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2};});
  await send('Input.dispatchMouseEvent',{type:'mousePressed',...copy,button:'left',clickCount:1});
  await send('Input.dispatchMouseEvent',{type:'mouseReleased',...copy,button:'left',clickCount:1});await delay(200);
  const clipboard=await evaluate(async()=>{const b=document.querySelector('.md-clipboard'),source=document.querySelector(b.dataset.clipboardTarget);return {actual:(await navigator.clipboard.readText()).trim(),expected:source.innerText.trim(),target:b.dataset.clipboardTarget};});
  assert(clipboard.actual.replace(/\r\n/g,'\n')===clipboard.expected.replace(/\r\n/g,'\n'),'Code copy clipboard contents differ: '+JSON.stringify(clipboard));
  await load('/',390);
  await evaluate(()=>document.querySelector('label[for="__drawer"] .header-control-focus').focus());
  await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,text:'\r'});
  await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});await delay(300);
  const drawerState=await evaluate(()=>({checked:document.getElementById('__drawer').checked,hasPaper:[...document.querySelectorAll('.md-sidebar--primary a')].some(a=>a.innerText.includes('论文阅读')),active:document.activeElement.outerHTML.slice(0,300)}));
  assert(drawerState.checked && drawerState.hasPaper,'Mobile navigation did not open: '+JSON.stringify(drawerState));
  await capture('mobile-navigation-390');
  await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
  await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
  assert(await evaluate(()=>!document.getElementById('__drawer').checked),'Mobile navigation did not close');
  await evaluate(()=>document.querySelector('label[for="__palette_1"] .header-control-focus').focus());
  await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,text:'\r'});
  await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});await delay(100);
  assert(await evaluate(()=>document.body.dataset.mdColorScheme==='slate' && document.activeElement.closest('label').htmlFor==='__palette_0'),'Keyboard theme control failed');
  assert(!errors.length,'Browser exceptions: '+errors.join(','));
  await rm(path.join(artifacts,'failure.txt'),{force:true});
  await writeFile(path.join(artifacts,'report.json'),JSON.stringify({checks,homeTransfer,keyboard:true,search:true,themeControl:true,codeCopy:true,mobileNav:true,researchPaths:true,idleStatic:true,buttonHover:true,reducedMotion:true,noScript:true,verticalTouchScroll:option('--only')!=='interaction',errors},null,2));
  console.log('PASS: real-browser layout, TOC, math, table, themes, images, search and fallbacks.');
  await send('Target.closeTarget',{targetId:target.id});
}catch(error){await writeFile(path.join(artifacts,'failure.txt'),String(error.stack));throw error;}
finally{socket?.close();server.close();browser?.kill();}
