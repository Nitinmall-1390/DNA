import { useState, useEffect, useRef, useCallback } from "react";

// ── Config ────────────────────────────────────────────────────
// Change this to your deployed backend URL when you deploy
// e.g. "https://dna-lstm-backend.onrender.com"
const API_BASE = process.env.REACT_APP_API_URL || "https://dna-production-28d1.up.railway.app";

const BC  = { A:"#00ff88", C:"#00d4ff", G:"#ffaa00", T:"#ff6b6b" };
const BBG = { A:"#00ff8812", C:"#00d4ff12", G:"#ffaa0012", T:"#ff6b6b12" };
const MF  = { fontFamily:"monospace" };
const BASES = ["A","C","G","T"];

const DEF_TRAIN = {
  sequence_col:"sequence", min_seq_len:100, window_size:100, step:3,
  max_sequences:1000, lstm_units_1:32, lstm_units_2:16, lstm_units_3:8,
  dropout_rate:0.2, batch_size:64, epochs:50, learning_rate:0.001,
  val_split:0.15, patience:15,
};

const DEF_GEN = { num_sequences:3, gen_length:100, temperature:1.0 };

// ── LetterGlitch (DNA-themed) ────────────────────────────────
function LetterGlitch({ glitchSpeed=50, smooth=true, outerVignette=true }){
  const cvRef = useRef(null);
  const animRef = useRef(null);
  const lettersRef = useRef([]);
  const gridRef = useRef({columns:0,rows:0});
  const ctxRef = useRef(null);
  const lastRef = useRef(Date.now());
  const chars = ["A","C","G","T","A","T","G","C","·","·","·","·"];
  const colors = ["#00ff88","#00d4ff","#ffaa00","#ff6b6b","#00ff8855","#00d4ff55","#ffaa0055","#ff6b6b33","#1e4030","#1a3040","#0a2020"];
  const RC = () => chars[Math.floor(Math.random()*chars.length)];
  const RK = () => colors[Math.floor(Math.random()*colors.length)];
  const h2r = h => { const r=/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(h.replace(/^#?([a-f\d])([a-f\d])([a-f\d])$/i,(_,a,b,c)=>a+a+b+b+c+c)); return r?{r:parseInt(r[1],16),g:parseInt(r[2],16),b:parseInt(r[3],16)}:null; };
  const lerp = (s,e,f) => `rgb(${Math.round(s.r+(e.r-s.r)*f)},${Math.round(s.g+(e.g-s.g)*f)},${Math.round(s.b+(e.b-s.b)*f)})`;
  const init = (cols,rows) => { gridRef.current={columns:cols,rows}; lettersRef.current=Array.from({length:cols*rows},()=>({char:RC(),color:RK(),targetColor:RK(),colorProgress:1})); };
  const draw = () => {
    const ctx=ctxRef.current; if(!ctx||!lettersRef.current.length)return;
    const {width,height}=cvRef.current.getBoundingClientRect();
    ctx.clearRect(0,0,width,height); ctx.font="16px monospace"; ctx.textBaseline="top";
    lettersRef.current.forEach((l,i)=>{ ctx.fillStyle=l.color; ctx.fillText(l.char,(i%gridRef.current.columns)*10,Math.floor(i/gridRef.current.columns)*20); });
  };
  const update = () => {
    const n=Math.max(1,Math.floor(lettersRef.current.length*0.05));
    for(let i=0;i<n;i++){
      const idx=Math.floor(Math.random()*lettersRef.current.length);
      if(!lettersRef.current[idx])continue;
      lettersRef.current[idx].char=RC(); lettersRef.current[idx].targetColor=RK();
      if(!smooth){lettersRef.current[idx].color=lettersRef.current[idx].targetColor;lettersRef.current[idx].colorProgress=1;}
      else lettersRef.current[idx].colorProgress=0;
    }
  };
  const smoothStep = () => {
    let dirty=false;
    lettersRef.current.forEach(l=>{ if(l.colorProgress<1){ l.colorProgress=Math.min(1,l.colorProgress+0.05); const s=h2r(l.color),e=h2r(l.targetColor); if(s&&e){l.color=lerp(s,e,l.colorProgress);dirty=true;} } });
    if(dirty)draw();
  };
  const resize = () => {
    const cv=cvRef.current; if(!cv)return; const par=cv.parentElement; if(!par)return;
    const dpr=window.devicePixelRatio||1,rect=par.getBoundingClientRect();
    cv.width=rect.width*dpr; cv.height=rect.height*dpr;
    cv.style.width=`${rect.width}px`; cv.style.height=`${rect.height}px`;
    if(ctxRef.current)ctxRef.current.setTransform(dpr,0,0,dpr,0,0);
    init(Math.ceil(rect.width/10),Math.ceil(rect.height/20)); draw();
  };
  const animate = () => {
    const now=Date.now();
    if(now-lastRef.current>=glitchSpeed){update();draw();lastRef.current=now;}
    if(smooth)smoothStep();
    animRef.current=requestAnimationFrame(animate);
  };
  useEffect(()=>{
    const cv=cvRef.current; if(!cv)return;
    ctxRef.current=cv.getContext("2d"); resize(); animate();
    let t; const onR=()=>{clearTimeout(t);t=setTimeout(()=>{cancelAnimationFrame(animRef.current);resize();animate();},100);};
    window.addEventListener("resize",onR);
    return()=>{cancelAnimationFrame(animRef.current);window.removeEventListener("resize",onR);};
  },[glitchSpeed,smooth]);
  return(
    <div style={{position:"absolute",inset:0,backgroundColor:"#000",overflow:"hidden"}}>
      <canvas ref={cvRef} style={{display:"block",width:"100%",height:"100%"}}/>
      {outerVignette&&<div style={{position:"absolute",inset:0,pointerEvents:"none",background:"radial-gradient(circle,rgba(0,0,0,0)60%,rgba(0,0,0,1)100%)"}}/>}
    </div>
  );
}

// ── Helix (header) ────────────────────────────────────────────
function Helix(){
  const ref=useRef(null);
  useEffect(()=>{
    const cv=ref.current; if(!cv)return; const ctx=cv.getContext("2d"); let f=0,raf;
    const draw=()=>{
      ctx.clearRect(0,0,cv.width,cv.height);
      const W=cv.width,H=cv.height,t=f*.013;
      for(let i=0;i<24;i++){
        const x=(i/23)*W,y1=H/2+Math.sin(i*.48+t)*22,y2=H/2+Math.sin(i*.48+t+Math.PI)*22,b=BASES[(i+Math.floor(t))%4];
        ctx.strokeStyle=BC[b]+"66"; ctx.lineWidth=1.4;
        ctx.beginPath(); ctx.moveTo(x,y1); ctx.lineTo(x,y2); ctx.stroke();
        ctx.fillStyle=BC[b]; ctx.font="bold 8px Courier New"; ctx.textAlign="center";
        ctx.fillText(b,x,y1-4); ctx.fillText(BASES[(i+2+Math.floor(t))%4],x,y2+11);
      }
      [[0,W*.2],[W,W*.8]].forEach(([x0,x1])=>{const g=ctx.createLinearGradient(x0,0,x1,0);g.addColorStop(0,"#080c14");g.addColorStop(1,"transparent");ctx.fillStyle=g;ctx.fillRect(0,0,W,H);});
      f++; raf=requestAnimationFrame(draw);
    };
    draw(); return()=>cancelAnimationFrame(raf);
  },[]);
  return <canvas ref={ref} width={700} height={64} style={{width:"100%",height:64,display:"block"}}/>;
}

// ── Atoms ──────────────────────────────────────────────────────
const Pill=({children,col})=><span style={{background:col+"16",color:col,border:`1px solid ${col}35`,borderRadius:20,padding:"2px 9px",fontSize:11,...MF,fontWeight:700,letterSpacing:.7}}>{children}</span>;
const SLab=({c="#00ff88",children})=><div style={{fontSize:10,color:c,...MF,letterSpacing:2,marginBottom:10,paddingBottom:7,borderBottom:`1px solid ${c}22`}}>{children}</div>;
function CField({name,value,onChange,type="number",step,min,max}){
  return(
    <div style={{marginBottom:9}}>
      <label style={{display:"block",color:"#5a6a7a",fontSize:10,marginBottom:3,...MF}}>{name}</label>
      <input type={type} value={value} step={step} min={min} max={max}
        onChange={e=>onChange(name,type==="number"?parseFloat(e.target.value):e.target.value)}
        style={{width:"100%",background:"#080c14",border:"1px solid #1e2d40",borderRadius:6,padding:"6px 9px",color:"#e2e8f0",...MF,fontSize:12,boxSizing:"border-box"}}/>
    </div>
  );
}

// ── Sequence card ──────────────────────────────────────────────
function SeqCard({seq,idx,gc}){
  const[open,setOpen]=useState(false);
  const chunks=(seq||"").match(/.{1,10}/g)||[];
  return(
    <div style={{background:"#0a0e18",border:"1px solid #1e2d40",borderRadius:10,marginBottom:8,overflow:"hidden"}}>
      <div onClick={()=>setOpen(o=>!o)} style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"9px 14px",cursor:"pointer"}}>
        <span style={{color:"#00ff88",...MF,fontSize:12,fontWeight:700}}>SEQ_{String(idx).padStart(3,"0")}</span>
        <div style={{display:"flex",gap:7,alignItems:"center"}}>
          <Pill col="#667788">len={seq.length}</Pill>
          <Pill col={gc>50?"#00d4ff":"#ffaa00"}>GC={gc}%</Pill>
          <span style={{color:"#334455",fontSize:11}}>{open?"▲":"▼"}</span>
        </div>
      </div>
      {open&&(
        <div style={{padding:"0 14px 12px",borderTop:"1px solid #1e2d40"}}>
          <div style={{...MF,fontSize:12,lineHeight:1.9,wordBreak:"break-all",marginTop:10}}>
            {chunks.map((ch,ci)=><span key={ci}>{ch.split("").map((c,bi)=><span key={bi} style={{color:BC[c]||"#ccc",background:BBG[c]||"transparent",padding:"0 1px",borderRadius:2}}>{c}</span>)}<span style={{color:"#1e2d40"}}> </span></span>)}
          </div>
          <button onClick={()=>navigator.clipboard.writeText(seq)} style={{marginTop:9,background:"none",border:"1px solid #1e2d40",borderRadius:6,color:"#5a6a7a",padding:"3px 12px",...MF,fontSize:11,cursor:"pointer"}}>COPY</button>
        </div>
      )}
    </div>
  );
}

// ── Training progress chart (canvas) ──────────────────────────
function LossChart({history}){
  const ref=useRef(null);
  useEffect(()=>{
    const cv=ref.current; if(!cv||!history.length)return;
    const ctx=cv.getContext("2d");
    const W=cv.width,H=cv.height,pad=36;
    ctx.clearRect(0,0,W,H);
    const losses=history.map(e=>e.loss);
    const valLosses=history.map(e=>e.val_loss);
    const allVals=[...losses,...valLosses];
    const minV=Math.min(...allVals),maxV=Math.max(...allVals);
    const scX=(i)=>pad+(i/(history.length-1||1))*(W-pad*2);
    const scY=(v)=>H-pad-(v-minV)/(maxV-minV||1)*(H-pad*2);
    // grid
    ctx.strokeStyle="#1e2d40"; ctx.lineWidth=0.5;
    for(let i=0;i<=4;i++){const y=pad+i*(H-pad*2)/4;ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(W-pad,y);ctx.stroke();}
    // train loss line
    ctx.strokeStyle="#00ff88"; ctx.lineWidth=2; ctx.beginPath();
    history.forEach((e,i)=>{i===0?ctx.moveTo(scX(i),scY(e.loss)):ctx.lineTo(scX(i),scY(e.loss));});
    ctx.stroke();
    // val loss line
    ctx.strokeStyle="#ff6b6b"; ctx.lineWidth=2; ctx.setLineDash([4,3]); ctx.beginPath();
    history.forEach((e,i)=>{i===0?ctx.moveTo(scX(i),scY(e.val_loss)):ctx.lineTo(scX(i),scY(e.val_loss));});
    ctx.stroke(); ctx.setLineDash([]);
    // dots
    history.forEach((e,i)=>{
      ctx.fillStyle="#00ff88"; ctx.beginPath(); ctx.arc(scX(i),scY(e.loss),3,0,Math.PI*2); ctx.fill();
      ctx.fillStyle="#ff6b6b"; ctx.beginPath(); ctx.arc(scX(i),scY(e.val_loss),3,0,Math.PI*2); ctx.fill();
    });
    // labels
    ctx.fillStyle="#5a6a7a"; ctx.font="10px monospace"; ctx.textAlign="right";
    ctx.fillText(maxV.toFixed(3),pad-4,pad+4); ctx.fillText(minV.toFixed(3),pad-4,H-pad+4);
    ctx.textAlign="center";
    ctx.fillText("1",pad,H-pad+14); ctx.fillText(String(history.length),W-pad,H-pad+14);
  },[history]);
  return(
    <div>
      <div style={{display:"flex",gap:16,marginBottom:8}}>
        <div style={{display:"flex",alignItems:"center",gap:6}}><div style={{width:16,height:2,background:"#00ff88"}}/><span style={{fontSize:11,color:"#5a6a7a",...MF}}>train loss</span></div>
        <div style={{display:"flex",alignItems:"center",gap:6}}><div style={{width:16,height:2,background:"#ff6b6b",borderTop:"2px dashed #ff6b6b"}}/><span style={{fontSize:11,color:"#5a6a7a",...MF}}>val loss</span></div>
      </div>
      <canvas ref={ref} width={440} height={180} style={{width:"100%",height:180,display:"block"}}/>
    </div>
  );
}

// ── Landing page ───────────────────────────────────────────────
function LandingPage({onEnter}){
  const[hov,setHov]=useState(false);
  return(
    <div style={{position:"relative",width:"100%",height:"100vh",overflow:"hidden",display:"flex",alignItems:"center",justifyContent:"center"}}>
      <LetterGlitch glitchSpeed={50} smooth={true} outerVignette={true}/>
      <div style={{position:"absolute",inset:0,background:"radial-gradient(ellipse 75% 65% at 50% 50%,rgba(0,0,0,0.75) 0%,rgba(0,0,0,0.45) 55%,transparent 100%)",pointerEvents:"none",zIndex:1}}/>
      <div style={{position:"relative",zIndex:2,textAlign:"center",padding:"40px 24px",maxWidth:620,background:"rgba(255,255,255,0.02)",backdropFilter:"blur(6px)",borderRadius:24,border:"1px solid rgba(255,255,255,0.03)"}}>
        <div style={{display:"inline-flex",alignItems:"center",gap:8,background:"#00ff8812",border:"1px solid #00ff8830",borderRadius:20,padding:"5px 16px",marginBottom:26}}>
          <span style={{width:7,height:7,borderRadius:"50%",background:"#00ff88",boxShadow:"0 0 12px #00ff88",display:"inline-block"}}/>
          <span style={{...MF,fontSize:10,color:"#00ff88",letterSpacing:3}}>LSTM NEURAL NETWORK · ACTIVE</span>
        </div>
        <style>{"@import url('https://fonts.googleapis.com/css2?family=Audiowide&family=Syncopate:wght@700&display=swap');"}</style>
        <h1 style={{fontFamily:"'Audiowide', sans-serif",fontSize:"clamp(34px,6vw,56px)",fontWeight:400,margin:"0 0 5px 0",color:"#ffffff",letterSpacing:"4px",lineHeight:1,textTransform:"uppercase"}}>
          DNA Sequence
        </h1>
        <h1 style={{fontFamily:"'Syncopate', sans-serif",fontSize:"clamp(34px,6vw,62px)",fontWeight:700,margin:"0 0 22px",background:"linear-gradient(100deg,#00ff88 0%,#00d4ff 45%,#a78bfa 100%)",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",letterSpacing:"6px",lineHeight:1.1,textTransform:"uppercase"}}>GENERATOR</h1>
        <div style={{display:"inline-flex",alignItems:"center",gap:10,background:"rgba(167,139,250,0.1)",border:"1px solid rgba(167,139,250,0.25)",borderRadius:30,padding:"7px 18px",marginBottom:22}}>
          <div style={{width:30,height:30,borderRadius:"50%",background:"linear-gradient(135deg,#a78bfa,#00d4ff)",display:"flex",alignItems:"center",justifyContent:"center",...MF,fontSize:11,fontWeight:800,color:"#fff",flexShrink:0}}>NM</div>
          <span style={{...MF,fontSize:13,color:"#c4b5fd",letterSpacing:.4}}>Developed by <strong style={{color:"#e9d5ff"}}>Nitin Mall</strong></span>
        </div>
        <p style={{fontSize:14,color:"rgba(255,255,255,0.85)",maxWidth:480,margin:"0 auto 28px",lineHeight:1.9}}>
          Upload your DNA dataset, train a real LSTM neural network on it, then generate novel sequences — powered by TensorFlow/Keras running on your own server.
        </p>
        <div style={{display:"flex",justifyContent:"center",gap:8,marginBottom:38,flexWrap:"wrap"}}>
          {[["🧬","Real LSTM Training"],["📊","Live Loss Chart"],["🔬","Your Data Only"],["⚡","Cached Model"]].map(([ic,lb])=>(
            <div key={lb} style={{background:"rgba(0,0,0,0.5)",border:"1px solid rgba(255,255,255,0.1)",borderRadius:8,padding:"6px 13px",display:"flex",alignItems:"center",gap:6}}>
              <span style={{fontSize:13}}>{ic}</span>
              <span style={{...MF,fontSize:10,color:"rgba(255,255,255,0.85)"}}>{lb}</span>
            </div>
          ))}
        </div>
        <div style={{display:"inline-block",position:"relative"}}>
          <div style={{position:"absolute",inset:-5,borderRadius:50,background:hov?"linear-gradient(135deg,#00ff88,#00d4ff,#a78bfa)":"transparent",filter:"blur(18px)",opacity:hov?.7:0,transition:"all .4s",pointerEvents:"none"}}/>
          <button onMouseEnter={()=>setHov(true)} onMouseLeave={()=>setHov(false)} onClick={onEnter}
            style={{position:"relative",background:hov?"linear-gradient(135deg,#00ff88,#00d4ff,#a78bfa)":"rgba(0,0,0,0.6)",border:`2px solid ${hov?"transparent":"#00ff88"}`,borderRadius:50,padding:"17px 56px",...MF,fontWeight:800,fontSize:15,letterSpacing:4,color:hov?"#000":"#00ff88",cursor:"pointer",transition:"all .35s cubic-bezier(.34,1.56,.64,1)",transform:hov?"scale(1.08)":"scale(1)",boxShadow:hov?"0 0 70px rgba(0,255,136,0.4)":"0 0 30px rgba(0,255,136,0.15)",outline:"none"}}>
            ◈ &nbsp;ENTER LAB&nbsp; ◈
          </button>
        </div>
        <p style={{...MF,fontSize:9,color:"rgba(255,255,255,0.4)",marginTop:22,letterSpacing:2}}>TENSORFLOW/KERAS · V2.0 · © NITIN MALL</p>
      </div>
    </div>
  );
}

// ── Main App ───────────────────────────────────────────────────
function DNAApp({onBack}){
  const[tab,setTab]=useState("train");
  const[trainCfg,setTrainCfg]=useState(DEF_TRAIN);
  const[genCfg,setGenCfg]=useState(DEF_GEN);
  const[file,setFile]=useState(null);
  const[fileInfo,setFileInfo]=useState(null);
  const[drag,setDrag]=useState(false);
  const[trainStatus,setTrainStatus]=useState("idle"); // idle|training|done|error
  const[genStatus,setGenStatus]=useState("idle");
  const[history,setHistory]=useState([]);
  const[log,setLog]=useState([]);
  const[results,setResults]=useState([]);
  const[seedInput,setSeedInput]=useState("");
  const[serverOnline,setServerOnline]=useState(null);
  const[modelReady,setModelReady]=useState(false);
  const[currentEpoch,setCurrentEpoch]=useState(0);
  const[totalEpochs,setTotalEpochs]=useState(50);
  const logRef=useRef(null);
  const esRef=useRef(null);
  const fRef=useRef(null);

  useEffect(()=>{if(logRef.current)logRef.current.scrollTop=logRef.current.scrollHeight;},[log]);

  const addLog=(msg,type="info")=>{
    const cols={info:"#5a7a8a",success:"#00ff88",warn:"#ffaa00",error:"#ff6b6b",epoch:"#00d4ff",sys:"#a78bfa"};
    setLog(l=>[...l.slice(-150),{msg,color:cols[type]||"#5a6a7a",ts:new Date().toLocaleTimeString("en-GB",{hour12:false})}]);
  };

  // Check server status on mount
  useEffect(()=>{
    fetch(`${API_BASE}/status`)
      .then(r=>r.json())
      .then(d=>{setServerOnline(true);setModelReady(d.trained);if(d.trained)addLog("Server online — trained model found in cache.","success");else addLog("Server online — no model trained yet.","info");})
      .catch(()=>{setServerOnline(false);addLog(`Cannot reach server at ${API_BASE}. Is it running?`,"error");});
  },[]);

  const updTrain=useCallback((k,v)=>setTrainCfg(c=>({...c,[k]:v})),[]);
  const updGen=useCallback((k,v)=>setGenCfg(c=>({...c,[k]:v})),[]);

  const handleFile=f=>{
    if(!f||!f.name.toLowerCase().endsWith(".csv")){addLog("Please select a .csv file.","warn");return;}
    setFile(f); setFileInfo(null);
    const fd=new FormData(); fd.append("file",f);
    fetch(`${API_BASE}/upload-csv?sequence_col=${trainCfg.sequence_col}&min_seq_len=${trainCfg.min_seq_len}`,{method:"POST",body:fd})
      .then(r=>r.json()).then(d=>{if(d.detail)throw new Error(d.detail);setFileInfo(d);addLog(`CSV parsed: ${d.rows.toLocaleString()} sequences, avg ${d.avg_len}bp.`,"success");})
      .catch(e=>addLog(`CSV error: ${e.message}`,"error"));
  };

  const startTraining=()=>{
    if(!file){addLog("Please upload a CSV file first.","warn");return;}
    setTrainStatus("training"); setHistory([]); setCurrentEpoch(0); setTotalEpochs(trainCfg.epochs); setModelReady(false);
    addLog("Sending dataset to server...","sys");

    const fd=new FormData(); fd.append("file",file); fd.append("config",JSON.stringify(trainCfg));

    // Close any previous SSE
    if(esRef.current)esRef.current.close();

    fetch(`${API_BASE}/train`,{method:"POST",body:fd})
      .then(response=>{
        if(!response.ok)return response.json().then(d=>{throw new Error(d.detail||"Training failed");});
        const reader=response.body.getReader(); const dec=new TextDecoder(); let buf="";
        const pump=()=>reader.read().then(({done,value})=>{
          if(done){return;}
          buf+=dec.decode(value,{stream:true});
          const parts=buf.split("\n\n"); buf=parts.pop();
          parts.forEach(part=>{
            const line=part.replace(/^data:\s*/,"").trim();
            if(!line)return;
            try{
              const ev=JSON.parse(line);
              if(ev.type==="log") addLog(ev.msg,"info");
              if(ev.type==="epoch"){
                setCurrentEpoch(ev.epoch);
                setHistory(h=>[...h,{epoch:ev.epoch,loss:ev.loss,val_loss:ev.val_loss,acc:ev.acc,val_acc:ev.val_acc}]);
                addLog(`Epoch ${ev.epoch}/${ev.total}  loss=${ev.loss}  val_loss=${ev.val_loss}  acc=${ev.acc}%`,"epoch");
              }
              if(ev.type==="done"){setTrainStatus("done");setModelReady(true);addLog("✓ Training complete! Model ready — go to GENERATE tab.","success");setTab("generate");}
              if(ev.type==="error"){setTrainStatus("error");addLog(`Training error: ${ev.msg}`,"error");}
            }catch{}
          });
          pump();
        }).catch(e=>{setTrainStatus("error");addLog(`Stream error: ${e.message}`,"error");});
        pump();
      })
      .catch(e=>{setTrainStatus("error");addLog(`Error: ${e.message}`,"error");});
  };

  const runGenerate=()=>{
    if(!modelReady){addLog("Train the model first.","warn");return;}
    setGenStatus("loading"); setResults([]);
    addLog("Generating sequences from trained LSTM...","sys");
    const body={...genCfg};
    if(seedInput.trim())body.seed=seedInput.trim().toUpperCase().replace(/[^ACGT]/g,"");
    fetch(`${API_BASE}/generate`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)})
      .then(r=>r.json())
      .then(d=>{
        if(d.detail)throw new Error(d.detail);
        setResults(d.sequences);
        setGenStatus("done");
        addLog(`✓ Generated ${d.sequences.length} sequences from seed: ${d.seed_used.slice(0,20)}...`,"success");
        setTab("results");
      })
      .catch(e=>{setGenStatus("idle");addLog(`Generate error: ${e.message}`,"error");});
  };

  const TABS=[{id:"train",label:"TRAIN"},{id:"generate",label:"GENERATE"},{id:"results",label:`RESULTS${results.length>0?` (${results.length})`:""}`}];
  const progress=totalEpochs>0?Math.round((currentEpoch/totalEpochs)*100):0;
  const SC={idle:"#667788",training:"#ffaa00",done:"#00ff88",error:"#ff6b6b"}[trainStatus];

  return(
    <div style={{background:"#0d1117",minHeight:"100vh",color:"#e2e8f0",fontFamily:"Segoe UI,system-ui,sans-serif"}}>
      {/* Header */}
      <div style={{background:"#080c14",borderBottom:"1px solid #1e2d40"}}>
        <div style={{maxWidth:960,margin:"0 auto",padding:"0 22px"}}>
          <div style={{paddingTop:14,display:"flex",justifyContent:"space-between",alignItems:"flex-start",flexWrap:"wrap",gap:10}}>
            <div>
              <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:3,flexWrap:"wrap"}}>
                <button onClick={onBack} style={{background:"none",border:"1px solid #1e2d40",borderRadius:6,color:"#5a6a7a",padding:"2px 10px",...MF,fontSize:10,cursor:"pointer"}}>← BACK</button>
                <div style={{width:7,height:7,borderRadius:"50%",background:serverOnline===null?"#ffaa00":serverOnline?"#00ff88":"#ff6b6b",boxShadow:`0 0 8px ${serverOnline?"#00ff88":"#ff6b6b"}`}}/>
                <span style={{...MF,fontSize:10,color:serverOnline?"#00ff88":"#ff6b6b",letterSpacing:1.5}}>
                  {serverOnline===null?"CONNECTING...":serverOnline?"SERVER ONLINE":"SERVER OFFLINE"}
                </span>
                {modelReady&&<span style={{background:"#001a0d",color:"#00ff88",border:"1px solid #00ff8830",...MF,fontSize:9,padding:"1px 8px",borderRadius:10}}>⚡ MODEL READY</span>}
              </div>
              <h1 style={{margin:"0 0 1px",fontSize:20,fontWeight:700,color:"#f0f4f8",letterSpacing:-.5}}>DNA LSTM Generator</h1>
              <p style={{margin:0,color:"#5a6a7a",fontSize:11}}>By <span style={{color:"#a78bfa",fontWeight:600}}>Nitin Mall</span><span style={{color:"#1e2d40",margin:"0 6px"}}>·</span>TensorFlow/Keras · Real Model</p>
            </div>
            <div style={{display:"flex",flexDirection:"column",alignItems:"flex-end",gap:4}}>
              <span style={{background:{idle:"#1e2d40",training:"#2a2000",done:"#001a0d",error:"#1a0000"}[trainStatus],color:SC,border:`1px solid ${SC}35`,borderRadius:20,padding:"3px 12px",fontSize:11,...MF,fontWeight:700}}>
                {{idle:"IDLE",training:`TRAINING ${progress}%`,done:"TRAINED",error:"ERROR"}[trainStatus]}
              </span>
              {trainStatus==="training"&&(
                <div style={{width:180,height:4,background:"#1e2d40",borderRadius:4,overflow:"hidden"}}>
                  <div style={{width:`${progress}%`,height:"100%",background:"linear-gradient(90deg,#00ff88,#00d4ff)",borderRadius:4,transition:"width .3s"}}/>
                </div>
              )}
            </div>
          </div>
          <Helix/>
          <div style={{display:"flex",gap:0,marginTop:2}}>
            {TABS.map(t=><button key={t.id} onClick={()=>setTab(t.id)} style={{background:"none",border:"none",borderBottom:tab===t.id?"2px solid #00ff88":"2px solid transparent",color:tab===t.id?"#00ff88":"#5a6a7a",...MF,fontSize:11,fontWeight:700,letterSpacing:1.5,padding:"7px 16px",cursor:"pointer",transition:"all .2s"}}>{t.label}</button>)}
          </div>
        </div>
      </div>

      <div style={{maxWidth:960,margin:"0 auto",padding:22}}>

        {/* ══ TRAIN ══ */}
        {tab==="train"&&(
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20}}>
            <div style={{display:"flex",flexDirection:"column",gap:14}}>
              {/* Upload */}
              <div style={{background:"#0a0e18",border:"1px solid #1e2d40",borderRadius:12,padding:16}}>
                <SLab color="#00ff88">STEP 1 — UPLOAD YOUR DNA CSV</SLab>
                <div onDragOver={e=>{e.preventDefault();setDrag(true);}} onDragLeave={()=>setDrag(false)}
                  onDrop={e=>{e.preventDefault();setDrag(false);handleFile(e.dataTransfer.files[0]);}}
                  onClick={()=>fRef.current.click()}
                  style={{border:`2px dashed ${drag?"#00ff88":"#1e2d40"}`,borderRadius:10,padding:"22px 14px",textAlign:"center",cursor:"pointer",background:drag?"#001a0d":"#080c14",transition:"all .2s"}}>
                  <input ref={fRef} type="file" accept=".csv" style={{display:"none"}} onChange={e=>handleFile(e.target.files[0])}/>
                  <div style={{fontSize:22,marginBottom:6}}>🧬</div>
                  <div style={{color:file?"#00ff88":"#8899aa",...MF,fontSize:13}}>{file?`✓ ${file.name}`:"Drop CSV or click to browse"}</div>
                  <div style={{color:"#334455",fontSize:11,...MF,marginTop:4}}>Must have a column of DNA sequences (A, C, G, T)</div>
                </div>
                {fileInfo&&(
                  <div style={{marginTop:10,background:"#080c14",borderRadius:8,padding:12,border:"1px solid #1e2d40"}}>
                    <div style={{display:"flex",gap:18,flexWrap:"wrap"}}>
                      {[["Sequences",fileInfo.rows.toLocaleString()],["Avg length",`${fileInfo.avg_len}bp`],["Columns",fileInfo.columns.join(", ")]].map(([k,v])=>(
                        <span key={k} style={{...MF,fontSize:11,color:"#5a6a7a"}}>{k}: <span style={{color:"#e2e8f0"}}>{v}</span></span>
                      ))}
                    </div>
                    {fileInfo.sample&&<div style={{...MF,fontSize:10,color:"#334455",marginTop:6}}>Sample: {fileInfo.sample[0]?.slice(0,60)}...</div>}
                  </div>
                )}
              </div>

              {/* Model config */}
              <div style={{background:"#0a0e18",border:"1px solid #1e2d40",borderRadius:12,padding:16}}>
                <SLab color="#00d4ff">STEP 2 — CONFIGURE MODEL</SLab>
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8}}>
                  {[["sequence_col","text"],["min_seq_len","number",1,500,1],["window_size","number",10,100,1],["step","number",1,20,1],["max_sequences","number",1000,200000,1000],["embedding_dim","number",2,32,2],["lstm_units_1","number",16,256,16],["lstm_units_2","number",16,128,16],["dropout_rate","number",0,0.7,0.05],["batch_size","number",32,512,32],["epochs","number",1,100,1],["learning_rate","number",0.00001,0.01,0.00001],["patience","number",2,20,1]].map(([n,t,mi,ma,st])=>(
                    <CField key={n} name={n} value={trainCfg[n]} onChange={updTrain} type={t} min={mi} max={ma} step={st}/>
                  ))}
                </div>
              </div>

              {/* Train button */}
              <button onClick={startTraining} disabled={trainStatus==="training"||!file}
                style={{background:trainStatus==="training"||!file?"#1e2d40":"linear-gradient(135deg,#00ff88,#00d4ff)",border:"none",borderRadius:10,padding:"14px 0",...MF,fontWeight:700,fontSize:14,letterSpacing:2,color:trainStatus==="training"||!file?"#5a6a7a":"#0d1117",cursor:trainStatus==="training"||!file?"not-allowed":"pointer",transition:"all .3s"}}>
                {trainStatus==="training"?`⟳  TRAINING... EPOCH ${currentEpoch}/${totalEpochs}`:"▶  START TRAINING"}
              </button>
              {trainStatus==="done"&&<button onClick={()=>setTab("generate")} style={{background:"none",border:"1px solid #00ff88",borderRadius:10,padding:"10px 0",...MF,fontWeight:700,fontSize:13,letterSpacing:1.5,color:"#00ff88",cursor:"pointer"}}>GENERATE SEQUENCES →</button>}
            </div>

            {/* Right: terminal + loss chart */}
            <div style={{display:"flex",flexDirection:"column",gap:14}}>
              {history.length>0&&(
                <div style={{background:"#0a0e18",border:"1px solid #1e2d40",borderRadius:12,padding:16}}>
                  <SLab color="#00d4ff">TRAINING LOSS</SLab>
                  <LossChart history={history}/>
                  {history.length>0&&(
                    <div style={{display:"flex",gap:20,marginTop:12,flexWrap:"wrap"}}>
                      {[["Loss",history[history.length-1].loss.toFixed(4),"#00ff88"],["Val loss",history[history.length-1].val_loss.toFixed(4),"#ff6b6b"],["Accuracy",`${history[history.length-1].acc}%`,"#00d4ff"],["Val acc",`${history[history.length-1].val_acc}%`,"#ffaa00"]].map(([k,v,c])=>(
                        <div key={k}>
                          <div style={{...MF,fontSize:10,color:"#5a6a7a",marginBottom:2}}>{k}</div>
                          <div style={{...MF,fontSize:16,fontWeight:700,color:c}}>{v}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
              <div style={{background:"#060a0e",border:"1px solid #1e2d40",borderRadius:12,padding:16,flex:1,display:"flex",flexDirection:"column"}}>
                <div style={{...MF,fontSize:10,color:"#00ff88",letterSpacing:2,marginBottom:8,borderBottom:"1px solid #1e2d40",paddingBottom:8,display:"flex",justifyContent:"space-between"}}>
                  <span>TRAINING LOG</span>
                  {log.length>0&&<span onClick={()=>setLog([])} style={{color:"#334455",cursor:"pointer"}}>CLEAR</span>}
                </div>
                <div ref={logRef} style={{flex:1,overflowY:"auto",height:history.length>0?200:480,...MF,fontSize:11,lineHeight:1.85}}>
                  {log.length===0&&<div style={{color:"#1e3040"}}>$ awaiting training...</div>}
                  {log.map((e,i)=><div key={i} style={{display:"flex",gap:10}}><span style={{color:"#1e3040",minWidth:74,flexShrink:0}}>[{e.ts}]</span><span style={{color:e.color,wordBreak:"break-word"}}>{e.msg}</span></div>)}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ══ GENERATE ══ */}
        {tab==="generate"&&(
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20}}>
            <div>
              {!modelReady&&(
                <div style={{background:"#1a1a00",border:"1px solid #ffaa0030",borderRadius:10,padding:"12px 16px",marginBottom:14}}>
                  <span style={{...MF,fontSize:12,color:"#ffaa00"}}>⚠ No trained model yet — train first on the TRAIN tab.</span>
                  <button onClick={()=>setTab("train")} style={{marginLeft:12,background:"none",border:"1px solid #ffaa00",borderRadius:6,color:"#ffaa00",padding:"3px 12px",...MF,fontSize:11,cursor:"pointer"}}>GO TO TRAIN →</button>
                </div>
              )}
              {modelReady&&<div style={{background:"#001a0d",border:"1px solid #00ff8825",borderRadius:10,padding:"10px 16px",marginBottom:14,...MF,fontSize:12,color:"#00ff88"}}>✓ Model trained and ready — using YOUR LSTM to generate.</div>}

              <div style={{background:"#0a0e18",border:"1px solid #1e2d40",borderRadius:12,padding:16,marginBottom:14}}>
                <SLab color="#00ff88">OPTIONAL SEED SEQUENCE</SLab>
                <textarea value={seedInput} rows={2} onChange={e=>setSeedInput(e.target.value.toUpperCase().replace(/[^ACGT]/g,""))} placeholder="Leave blank to use a random seed from your training data..."
                  style={{width:"100%",background:"#080c14",border:"1px solid #1e2d40",borderRadius:6,padding:9,color:"#00ff88",...MF,fontSize:12,resize:"none",boxSizing:"border-box"}}/>
                <div style={{...MF,fontSize:10,color:"#334455",marginTop:4}}>{seedInput.length}bp — if blank, a random training sequence is used</div>
              </div>

              <div style={{background:"#0a0e18",border:"1px solid #1e2d40",borderRadius:12,padding:16,marginBottom:14}}>
                <SLab color="#00ff88">GENERATION PARAMETERS</SLab>
                {[["temperature","Temperature (0.1=conservative, 2.0=creative)",0.1,3.0,0.1],["gen_length","Bases to generate",20,500,10],["num_sequences","Number of sequences",1,10,1]].map(([k,lbl,mn,mx,st])=>(
                  <div key={k} style={{marginBottom:13}}>
                    <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
                      <span style={{...MF,fontSize:11,color:"#5a6a7a"}}>{lbl}</span>
                      <span style={{...MF,fontSize:12,color:"#00ff88",fontWeight:700}}>{genCfg[k]}</span>
                    </div>
                    <input type="range" min={mn} max={mx} step={st} value={genCfg[k]} onChange={e=>updGen(k,parseFloat(e.target.value))} style={{width:"100%",accentColor:"#00ff88"}}/>
                  </div>
                ))}
              </div>

              <button onClick={runGenerate} disabled={!modelReady||genStatus==="loading"}
                style={{width:"100%",background:!modelReady||genStatus==="loading"?"#1e2d40":"linear-gradient(135deg,#00ff88,#00d4ff)",border:"none",borderRadius:10,padding:"13px 0",...MF,fontWeight:700,fontSize:14,letterSpacing:2,color:!modelReady||genStatus==="loading"?"#5a6a7a":"#0d1117",cursor:!modelReady||genStatus==="loading"?"not-allowed":"pointer"}}>
                {genStatus==="loading"?"⟳  GENERATING...":"▶  GENERATE FROM MY MODEL"}
              </button>
            </div>

            {/* Log panel */}
            <div style={{background:"#060a0e",border:"1px solid #1e2d40",borderRadius:12,padding:16,display:"flex",flexDirection:"column"}}>
              <SLab color="#00ff88">OUTPUT LOG</SLab>
              <div ref={logRef} style={{flex:1,overflowY:"auto",height:460,...MF,fontSize:11,lineHeight:1.85}}>
                {log.length===0&&<div style={{color:"#1e3040"}}>$ waiting...</div>}
                {log.map((e,i)=><div key={i} style={{display:"flex",gap:10}}><span style={{color:"#1e3040",minWidth:74,flexShrink:0}}>[{e.ts}]</span><span style={{color:e.color,wordBreak:"break-word"}}>{e.msg}</span></div>)}
              </div>
            </div>
          </div>
        )}

        {/* ══ RESULTS ══ */}
        {tab==="results"&&results.length===0&&(
          <div style={{textAlign:"center",padding:"60px 0",color:"#5a6a7a"}}>
            <div style={{fontSize:44,marginBottom:12}}>🧬</div>
            <div style={{...MF,fontSize:13,marginBottom:16}}>No sequences yet. Train and generate first.</div>
            <button onClick={()=>setTab("train")} style={{background:"none",border:"1px solid #00ff88",borderRadius:8,color:"#00ff88",padding:"8px 22px",...MF,fontSize:12,cursor:"pointer"}}>START TRAINING →</button>
          </div>
        )}
        {tab==="results"&&results.length>0&&(
          <div style={{display:"grid",gridTemplateColumns:"2fr 1fr",gap:20}}>
            <div>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:12}}>
                <span style={{...MF,fontSize:10,color:"#00ff88",letterSpacing:2}}>GENERATED BY YOUR LSTM ({results.length})</span>
                <div style={{display:"flex",gap:8}}>
                  <button onClick={()=>navigator.clipboard.writeText(results.map(r=>r.sequence).join("\n"))} style={{background:"none",border:"1px solid #1e2d40",borderRadius:6,color:"#5a6a7a",padding:"4px 12px",...MF,fontSize:11,cursor:"pointer"}}>COPY ALL</button>
                  <button onClick={()=>{const b=new Blob([results.map(r=>r.sequence).join("\n")],{type:"text/plain"});const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="dna_lstm_nitin_mall.txt";a.click();}} style={{background:"none",border:"1px solid #00ff88",borderRadius:6,color:"#00ff88",padding:"4px 12px",...MF,fontSize:11,cursor:"pointer"}}>DOWNLOAD</button>
                </div>
              </div>
              {results.map(r=><SeqCard key={r.id} seq={r.sequence} idx={r.id} gc={r.gc_percent}/>)}
            </div>
            <div style={{display:"flex",flexDirection:"column",gap:14}}>
              <div style={{background:"#0a0e18",border:"1px solid #1e2d40",borderRadius:10,padding:16}}>
                <SLab color="#00ff88">SUMMARY</SLab>
                {[["Sequences",results.length],["Avg length",`${Math.round(results.reduce((a,r)=>a+r.length,0)/results.length)}bp`],["Avg GC%",`${(results.reduce((a,r)=>a+r.gc_percent,0)/results.length).toFixed(1)}%`],["Temperature",genCfg.temperature],["Model",`LSTM(${trainCfg.lstm_units_1},${trainCfg.lstm_units_2})`]].map(([k,v])=>(
                  <div key={k} style={{display:"flex",justifyContent:"space-between",...MF,fontSize:12,marginBottom:6,paddingBottom:6,borderBottom:"1px solid #1e2d4030"}}>
                    <span style={{color:"#5a6a7a"}}>{k}</span><span style={{color:"#e2e8f0"}}>{v}</span>
                  </div>
                ))}
              </div>
              <div style={{background:"#0a0e18",border:"1px solid #1e2d40",borderRadius:10,padding:16}}>
                <SLab color="#00ff88">GC% PER SEQUENCE</SLab>
                {results.map(r=>(
                  <div key={r.id} style={{display:"flex",alignItems:"center",gap:8,marginBottom:7}}>
                    <span style={{...MF,fontSize:10,color:"#5a6a7a",minWidth:50}}>SEQ_{String(r.id).padStart(3,"0")}</span>
                    <div style={{flex:1,background:"#1a2030",borderRadius:4,height:8}}>
                      <div style={{width:`${Math.min(r.gc_percent,100)}%`,height:"100%",background:r.gc_percent>50?"#00d4ff":"#ffaa00",borderRadius:4,transition:"width .8s"}}/>
                    </div>
                    <span style={{...MF,fontSize:10,color:r.gc_percent>50?"#00d4ff":"#ffaa00",minWidth:36}}>{r.gc_percent}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        <div style={{marginTop:30,paddingTop:14,borderTop:"1px solid #1a2030",display:"flex",justifyContent:"space-between",alignItems:"center"}}>
          <span style={{...MF,fontSize:11,color:"#5a6a7a"}}>Developed by <span style={{color:"#a78bfa",fontWeight:700}}>Nitin Mall</span> · DNA-LSTM v2.0</span>
          <span style={{...MF,fontSize:10,color:"#1e2d40"}}>TensorFlow/Keras · FastAPI · Real Model</span>
        </div>
      </div>
    </div>
  );
}

export default function App(){
  const[page,setPage]=useState("landing");
  if(page==="landing")return <LandingPage onEnter={()=>setPage("app")}/>;
  return <DNAApp onBack={()=>setPage("landing")}/>;
}
