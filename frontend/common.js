const $=s=>document.querySelector(s);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const qid=()=>new URLSearchParams(location.search).get("id");
function nav(active){document.body.insertAdjacentHTML("afterbegin",`<header><b>Face Attendance</b><nav>
<a href="index.html" class="${active=="cap"?"on":""}">Capture</a>
<a ${qid()?`href="report.html?id=${qid()}"`:'href="#" aria-disabled="true"'} class="${active=="rep"?"on":""}">Report</a>
<a ${qid()?`href="annotated.html?id=${qid()}"`:'href="#" aria-disabled="true"'} class="${active=="ann"?"on":""}">Detected faces</a></nav></header>`)}
async function api(path,opt){const r=await fetch("/api"+path,opt);if(!r.ok){let m=r.statusText;try{m=(await r.json()).detail||m}catch{}throw new Error(m)}return r.json()}
function toast(t){const d=document.createElement("div");d.className="toast";d.textContent=t;document.body.append(d);setTimeout(()=>d.remove(),2800)}
