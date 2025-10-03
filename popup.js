
// ===== Utility: CSV parser =====
function parseCSV(text) {
  const rows = [];
  let i = 0, field = '', row = [], inQuotes = false;
  while (i < text.length) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i+1] === '"') { field += '"'; i++; }
        else inQuotes = false;
      } else field += c;
    } else {
      if (c === '"') inQuotes = true;
      else if (c === ',') { row.push(field); field=''; }
      else if (c === '\n') { row.push(field); rows.push(row); row=[]; field=''; }
      else if (c === '\r') {} // ignore
      else field += c;
    }
    i++;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  const headers = rows[0] || [];
  const outRows = rows.slice(1).filter(r => r.length && r.some(x => x !== ''));
  const objects = outRows.map(r => {
    const o = {};
    headers.forEach((h, idx) => o[h] = r[idx] ?? '');
    return o;
  });
  return { columns: headers, rows: objects };
}

// ===== Utility: numeric detection & vectorization =====
function isNumeric(x) {
  if (x === null || x === undefined) return false;
  const n = Number(x);
  return Number.isFinite(n);
}
function toNumberSafe(x) { const n = Number(x); return Number.isFinite(n) ? n : NaN; }

function trainTestSplit(X, y, testPct=0.2) {
  const n = X.length;
  const idx = Array.from({length:n}, (_,i)=>i).sort(()=>Math.random()-0.5);
  const testN = Math.max(1, Math.floor(n*testPct));
  const testIdx = new Set(idx.slice(0, testN));
  const Xtr=[], ytr=[], Xte=[], yte=[];
  for (let i=0; i<n; i++) {
    if (testIdx.has(i)) { Xte.push(X[i]); yte.push(y[i]); }
    else { Xtr.push(X[i]); ytr.push(y[i]); }
  }
  return { Xtr, ytr, Xte, yte };
}

function standardize(X) {
  const d = X[0].length;
  const mu = Array(d).fill(0);
  const sd = Array(d).fill(0);
  X.forEach(row => row.forEach((v,j)=> mu[j]+=v));
  for (let j=0;j<d;j++) mu[j]/=X.length;
  X.forEach(row => row.forEach((v,j)=> sd[j]+=(v-mu[j])**2));
  for (let j=0;j<d;j++) sd[j]=Math.sqrt(sd[j]/X.length)||1;
  const Xs = X.map(r => r.map((v,j)=>(v-mu[j])/sd[j]));
  return { Xs, mu, sd };
}

function applyStandardize(X, mu, sd) {
  return X.map(r => r.map((v,j)=>(v-mu[j])/sd[j]));
}

// ===== Linear Regression (batch GD) =====
function linearRegressionTrain(X, y, epochs=500, lr=0.05, l2=0.0) {
  const n = X.length, d = X[0].length;
  const w = Array(d).fill(0);
  let b = 0;
  for (let ep=0; ep<epochs; ep++) {
    let db=0; const dw=Array(d).fill(0);
    for (let i=0;i<n;i++) {
      const pred = b + X[i].reduce((s,v,j)=> s + w[j]*v, 0);
      const err = pred - y[i];
      db += err;
      for (let j=0;j<d;j++) dw[j] += err * X[i][j];
    }
    db/=n; for (let j=0;j<d;j++) dw[j] = dw[j]/n + l2*w[j];
    b -= lr*db; for (let j=0;j<d;j++) w[j] -= lr*dw[j];
  }
  return { w, b };
}
function linearPredict(X, model) {
  return X.map(r => model.b + r.reduce((s,v,j)=> s + model.w[j]*v, 0));
}
function regMetrics(yTrue, yPred) {
  const n = yTrue.length;
  const mse = yTrue.reduce((s,y,i)=>s+(yPred[i]-y)**2,0)/n;
  const mae = yTrue.reduce((s,y,i)=>s+Math.abs(yPred[i]-y),0)/n;
  const ybar = yTrue.reduce((s,y)=>s+y,0)/n;
  const ssTot = yTrue.reduce((s,y)=>s+(y-ybar)**2,0);
  const ssRes = yTrue.reduce((s,y,i)=>s+(yPred[i]-y)**2,0);
  const r2 = 1 - (ssRes/ssTot);
  return { mse, mae, r2 };
}

// ===== Logistic Regression (binary, batch GD) =====
function sigmoid(z){ return 1/(1+Math.exp(-z)); }
function logisticTrain(X, y, epochs=700, lr=0.05, l2=0.0) {
  const n = X.length, d = X[0].length;
  const w = Array(d).fill(0);
  let b = 0;
  for (let ep=0; ep<epochs; ep++) {
    let db=0; const dw=Array(d).fill(0);
    for (let i=0;i<n;i++) {
      const z = b + X[i].reduce((s,v,j)=> s + w[j]*v, 0);
      const p = sigmoid(z);
      const err = p - y[i]; // y in {0,1}
      db += err;
      for (let j=0;j<d;j++) dw[j] += err * X[i][j];
    }
    db/=n; for (let j=0;j<d;j++) dw[j] = dw[j]/n + l2*w[j];
    b -= lr*db; for (let j=0;j<d;j++) w[j] -= lr*dw[j];
  }
  return { w, b };
}
function logisticPredictProba(X, model) {
  return X.map(r => sigmoid(model.b + r.reduce((s,v,j)=> s + model.w[j]*v, 0)));
}
function logisticPredict(X, model, thr=0.5) {
  return logisticPredictProba(X, model).map(p => p>=thr?1:0);
}
function classMetrics(yTrue, yPred) {
  const n = yTrue.length;
  let tp=0, fp=0, tn=0, fn=0;
  for (let i=0;i<n;i++) {
    if (yTrue[i]===1 && yPred[i]===1) tp++;
    else if (yTrue[i]===0 && yPred[i]===1) fp++;
    else if (yTrue[i]===0 && yPred[i]===0) tn++;
    else if (yTrue[i]===1 && yPred[i]===0) fn++;
  }
  const acc=(tp+tn)/n;
  const prec= tp+fp ? tp/(tp+fp) : 0;
  const rec= tp+fn ? tp/(tp+fn) : 0;
  const f1= (prec+rec) ? 2*prec*rec/(prec+rec) : 0;
  return { acc, precision:prec, recall:rec, f1, confusion:{tp,fp,tn,fn} };
}

// ===== k-means =====
function kmeans(X, k=3, maxIter=100) {
  const n=X.length, d=X[0].length;
  const centroids = [];
  const used = new Set();
  while (centroids.length<k) {
    const idx = Math.floor(Math.random()*n);
    if (!used.has(idx)) { used.add(idx); centroids.push([...X[idx]]); }
  }
  let labels = Array(n).fill(0);
  for (let iter=0; iter<maxIter; iter++) {
    // assign
    let changed=false;
    for (let i=0;i<n;i++) {
      let best=0, bestDist=Infinity;
      for (let c=0;c<k;c++) {
        let dist=0;
        for (let j=0;j<d;j++) { const diff=X[i][j]-centroids[c][j]; dist+=diff*diff; }
        if (dist<bestDist) { bestDist=dist; best=c; }
      }
      if (labels[i]!==best){ labels[i]=best; changed=true; }
    }
    // update
    const sums=Array.from({length:k}, ()=>Array(d).fill(0));
    const counts=Array(k).fill(0);
    for (let i=0;i<n;i++){ counts[labels[i]]++; for (let j=0;j<d;j++) sums[labels[i]][j]+=X[i][j]; }
    for (let c=0;c<k;c++) {
      if (counts[c]===0) continue;
      for (let j=0;j<d;j++) centroids[c][j]=sums[c][j]/counts[c];
    }
    if (!changed) break;
  }
  // compute inertia (SSE)
  let sse=0;
  for (let i=0;i<n;i++){
    const c=labels[i];
    for (let j=0;j<d;j++){ const diff=X[i][j]-centroids[c][j]; sse+=diff*diff; }
  }
  return { labels, centroids, sse };
}

// ===== Permutation Importance =====
function permutationImportance(modelType, model, X, y, baseMetric, targetIsBinary=true) {
  const d = X[0].length;
  const importances = [];
  for (let j=0;j<d;j++) {
    // copy and shuffle feature j
    const Xperm = X.map(r => [...r]);
    const col = Xperm.map(r => r[j]);
    for (let i=col.length-1;i>0;i--) { const k=Math.floor(Math.random()*(i+1)); [col[i], col[k]]=[col[k], col[i]]; }
    Xperm.forEach((r,i)=> r[j]=col[i]);
    let score=0;
    if (modelType==='regression') {
      const yhat = linearPredict(Xperm, model);
      const m = regMetrics(y, yhat);
      score = -m.mse; // higher is better, so negative MSE
    } else if (modelType==='classification') {
      const yhat = logisticPredict(Xperm, model);
      const m = classMetrics(y, yhat);
      score = m.acc;
    }
    importances.push({ j, scoreDrop: baseMetric - score });
  }
  importances.sort((a,b)=> b.scoreDrop - a.scoreDrop);
  return importances;
}

// ===== OneR Rules (simple, interpretable baseline) =====
function oneR(X, y, featureNames) {
  // Discretize numeric features into 5 bins, choose best single-feature rules
  const d = X[0].length;
  const rules = [];
  for (let j=0;j<d;j++) {
    const vals = X.map(r=>r[j]);
    const min = Math.min(...vals), max=Math.max(...vals);
    const bins = 5;
    const edges = Array.from({length:bins+1}, (_,i)=> min + (i*(max-min)/bins));
    // tally majority class per bin
    const binLabels = Array(bins).fill(null).map(()=>({counts:new Map(), label:null}));
    for (let i=0;i<vals.length;i++) {
      let b= bins-1;
      for (let e=0;e<bins;e++){ if (vals[i] <= edges[e+1]) { b=e; break; } }
      const lab = y[i];
      const m = binLabels[b].counts;
      m.set(lab, (m.get(lab)||0)+1);
    }
    for (let b=0;b<bins;b++) {
      let bestLab=null, bestCnt=-1;
      for (const [lab,cnt] of binLabels[b].counts.entries()) {
        if (cnt>bestCnt){ bestCnt=cnt; bestLab=lab; }
      }
      binLabels[b].label = bestLab!==null? bestLab : 0;
    }
    // estimate accuracy of these rules
    let correct=0;
    for (let i=0;i<vals.length;i++) {
      let b= bins-1;
      for (let e=0;e<bins;e++){ if (vals[i] <= edges[e+1]) { b=e; break; } }
      if (binLabels[b].label === y[i]) correct++;
    }
    const acc = correct/vals.length;
    rules.push({ feature: featureNames[j], edges, labels: binLabels.map(b=>b.label), acc });
  }
  rules.sort((a,b)=> b.acc - a.acc);
  return rules[0]; // best single-feature rules
}

// ===== UI State =====
const state = {
  raw: null,
  columns: [],
  rows: [],
  numericCols: [],
  featureCols: [],
  task: "classification",
  target: null,
  model: null,
  scaler: null,
  enc: null,
  lastTrainSplit: null,
  lastModelType: null,
  rules: null
};

// ===== DOM helpers =====
function $(q){ return document.querySelector(q); }
function el(tag, attrs={}, children=[]) {
  const e = document.createElement(tag);
  for (const [k,v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else e.setAttribute(k,v);
  }
  children.forEach(c => e.appendChild(c));
  return e;
}
function renderTable(columns, rows, mount) {
  mount.innerHTML = "";
  if (!columns.length || !rows.length) { mount.textContent = "No data"; return; }
  const table = el("table");
  const thead = el("thead");
  const trh = el("tr");
  columns.forEach(c => trh.appendChild(el("th", { text: c })));
  thead.appendChild(trh); table.appendChild(thead);
  const tbody = el("tbody");
  rows.slice(0,200).forEach(r => {
    const tr = el("tr");
    columns.forEach(c => tr.appendChild(el("td", { text: String(r[c] ?? "") })));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  mount.appendChild(table);
}

// ===== Data loading =====
document.getElementById("btn-scrape").addEventListener("click", async () => {
  $("#scrape-status").textContent = "Scanning page for tables...";
  try {
    const [tab] = await chrome.tabs.query({active:true, currentWindow:true});
    const res = await chrome.tabs.sendMessage(tab.id, { type: "SCRAPE_TABLES" });
    const ds = (res && res.datasets) ? res.datasets : [];
    if (!ds.length) { $("#scrape-status").textContent = "No tables found."; return; }
    // Pick the largest by rows
    ds.sort((a,b)=> (b.rows?.length||0) - (a.rows?.length||0));
    const best = ds[0];
    ingestDataset(best.columns, best.rows);
    $("#scrape-status").textContent = `Imported ${best.rows.length} rows from ${best.name}`;
  } catch (e) {
    $("#scrape-status").textContent = "Error: " + e.message;
  }
});

document.getElementById("csv-file").addEventListener("change", async (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  const txt = await file.text();
  const parsed = parseCSV(txt);
  ingestDataset(parsed.columns, parsed.rows);
});

function ingestDataset(columns, rows) {
  state.raw = { columns, rows };
  state.columns = columns;
  state.rows = rows;
  renderTable(columns, rows, $("#table-container"));
  $("#dataset-meta").textContent = `${rows.length} rows × ${columns.length} columns`;
  // fill target and numeric features
  const colSelect = $("#target");
  colSelect.innerHTML = "";
  columns.forEach(c => colSelect.appendChild(el("option", { value:c, text:c })));
  colSelect.value = columns[columns.length-1];
  state.target = colSelect.value;

  // numeric detection
  const numeric = [];
  columns.forEach(c => {
    const vals = rows.slice(0,50).map(r=>r[c]);
    const numCount = vals.filter(v => isNumeric(v)).length;
    if (numCount >= Math.floor(vals.length*0.7)) numeric.push(c);
  });
  state.numericCols = numeric;
  // features exclude target (if numeric)
  state.featureCols = numeric.filter(c => c!==state.target);
  renderFeatureChips();
}

function renderFeatureChips() {
  const mount = $("#feature-list");
  mount.innerHTML = "";
  state.featureCols.forEach(c => mount.appendChild(el("span", { text: c })));
}

// Task/Target UI
$("#task").addEventListener("change", (e)=> {
  state.task = e.target.value;
  const isCluster = state.task === "clustering";
  $("#target-wrap").style.display = isCluster ? "none" : "block";
  $("#kmeans-wrap").style.display = isCluster ? "block" : "none";
});
$("#target").addEventListener("change", (e) => {
  state.target = e.target.value;
  // recompute feature list
  state.featureCols = state.numericCols.filter(c => c!==state.target);
  renderFeatureChips();
});

// ===== Training =====
$("#btn-train").addEventListener("click", () => {
  const task = state.task;
  if (!state.rows.length) { $("#train-status").textContent = "Load data first."; return; }
  const features = state.featureCols;
  if (task !== "clustering" && (!state.target || !features.length)) {
    $("#train-status").textContent = "Select a target and ensure numeric features exist.";
    return;
  }

  // Build matrices
  const X = state.rows.map(r => features.map(c => toNumberSafe(r[c]))).filter(row => row.every(x=>Number.isFinite(x)));
  if (!X.length) { $("#train-status").textContent = "No valid numeric rows after cleaning."; return; }

  let resultText = "";
  const splitPct = Number($("#split").value)/100;
  const { Xs, mu, sd } = standardize(X);
  state.scaler = { mu, sd, features };
  state.lastTrainSplit = splitPct;

  if (task === "regression") {
    const y = state.rows.map(r => toNumberSafe(r[state.target])).filter((_,i)=> X[i]!==undefined);
    const { Xtr, ytr, Xte, yte } = trainTestSplit(Xs, y, 1-splitPct);
    const model = linearRegressionTrain(Xtr, ytr, 800, 0.05, 1e-4);
    const yhat = linearPredict(Xte, model);
    const m = regMetrics(yte, yhat);
    resultText = `MSE: ${m.mse.toFixed(4)}, MAE: ${m.mae.toFixed(4)}, R²: ${m.r2.toFixed(4)} (test)`;
    // Feature importance by |weights|
    const imp = model.w.map((w,j)=>({ feature: features[j], importance: Math.abs(w) }))
      .sort((a,b)=> b.importance - a.importance);
    // permutation drop
    const base = -regMetrics(yte, yhat).mse;
    const perm = permutationImportance('regression', model, Xte, yte, base, false);
    renderImportance(imp, perm, features);
    // rules: OneR on y discretized (for interpretability preview)
    const yDisc = y.map(v => v>= (y.reduce((s,a)=>s+a,0)/y.length) ? 1 : 0).filter((_,i)=> X[i]!==undefined);
    state.rules = oneR(Xs, yDisc, features);
    renderRules(state.rules);
    state.model = { type:"regression", model, scaler: state.scaler };
    state.lastModelType="regression";
  } else if (task === "classification") {
    // Target must be binary; attempt to binarize if it's strings
    let yraw = state.rows.map(r => r[state.target]).filter((_,i)=> X[i]!==undefined);
    const unique = Array.from(new Set(yraw));
    let map = new Map();
    if (unique.length===2) { map.set(unique[0],0); map.set(unique[1],1); }
    else {
      // Try numeric threshold at median
      const nums = yraw.map(toNumberSafe);
      const median = nums.slice().sort((a,b)=>a-b)[Math.floor(nums.length/2)];
      yraw = nums.map(v => (v>=median?1:0));
    }
    const y = yraw.map(v => map.size? map.get(v): v);
    const { Xtr, ytr, Xte, yte } = trainTestSplit(Xs, y, 1-splitPct);
    const model = logisticTrain(Xtr, ytr, 900, 0.05, 1e-4);
    const yhat = logisticPredict(Xte, model);
    const m = classMetrics(yte, yhat);
    resultText = `Accuracy: ${m.acc.toFixed(4)} | Precision: ${m.precision.toFixed(4)} | Recall: ${m.recall.toFixed(4)} | F1: ${m.f1.toFixed(4)} (test)`;
    // importance from |weights| + permutation
    const imp = model.w.map((w,j)=>({ feature: features[j], importance: Math.abs(w) }))
      .sort((a,b)=> b.importance - a.importance);
    const base = classMetrics(yte, yhat).acc;
    const perm = permutationImportance('classification', model, Xte, yte, base, true);
    renderImportance(imp, perm, features);
    // OneR decision rules
    state.rules = oneR(Xs, y, features);
    renderRules(state.rules);
    state.model = { type:"classification", model, scaler: state.scaler, labelMap: Object.fromEntries(map) };
    state.lastModelType="classification";
  } else if (task === "clustering") {
    const k = Math.max(2, Math.min(10, Number($("#k").value)||3));
    const km = kmeans(Xs, k, 200);
    resultText = `k-means: k=${k}, SSE (inertia)=${km.sse.toFixed(4)}`;
    // feature importance heuristics: variance across centroids
    const centroidVar = state.featureCols.map((f,j)=>{
      const vals = km.centroids.map(c=>c[j]);
      const mu = vals.reduce((s,a)=>s+a,0)/vals.length;
      const v = vals.reduce((s,a)=>s+(a-mu)**2,0)/vals.length;
      return { feature:f, importance:v };
    }).sort((a,b)=> b.importance - a.importance);
    renderImportance(centroidVar, [], state.featureCols);
    // rules: assign thresholds from closest centroid per top feature
    state.rules = { feature: centroidVar[0]?.feature || "", edges: [], labels: [], acc: null, note: "Clusters summarized by centroid means of top-variance features." };
    renderRules(state.rules);
    state.model = { type:"clustering", km, scaler: state.scaler };
    state.lastModelType="clustering";
  }

  $("#train-status").textContent = "Training complete.";
  $("#metrics").textContent = resultText;
});

function renderImportance(absWeights, perm, features) {
  const mount = $("#feat-imp");
  const topAbs = absWeights.slice(0,10);
  let html = "<table><thead><tr><th>Feature</th><th>|Weight|/Variance</th></tr></thead><tbody>";
  topAbs.forEach(r => html += `<tr><td>${r.feature}</td><td>${r.importance.toFixed(6)}</td></tr>`);
  html += "</tbody></table>";
  if (perm && perm.length) {
    html += "<p class='muted'>Permutation importance (score drop):</p>";
    html += "<table><thead><tr><th>Feature</th><th>Δ Score</th></tr></thead><tbody>";
    perm.forEach(p => html += `<tr><td>${features[p.j]}</td><td>${p.scoreDrop.toFixed(6)}</td></tr>`);
    html += "</tbody></table>";
  }
  mount.innerHTML = html;
}

function renderRules(rule) {
  const mount = $("#rules");
  if (!rule) { mount.textContent = "No rules."; return; }
  if (rule.edges && rule.labels) {
    let html = `<p><b>OneR best feature:</b> ${rule.feature} (approx.)</p>`;
    html += "<ul>";
    for (let i=0;i<rule.labels.length;i++) {
      const lo = rule.edges[i].toFixed(4), hi = rule.edges[i+1].toFixed(4);
      html += `<li>if ${rule.feature} ∈ [${lo}, ${hi}] ⇒ predict ${rule.labels[i]}</li>`;
    }
    html += "</ul>";
    if (rule.acc!=null) html += `<p class='muted'>Rule accuracy (on training): ${(rule.acc*100).toFixed(2)}%</p>`;
    mount.innerHTML = html;
  } else {
    mount.textContent = rule.note || "Rules unavailable.";
  }
}

// ===== Prediction & Export =====
$("#btn-predict").addEventListener("click", () => {
  try {
    if (!state.model) { $("#predict-output").textContent = "Train a model first."; return; }
    const obj = JSON.parse($("#predict-input").value);
    const featVals = state.featureCols.map(f => toNumberSafe(obj[f]));
    if (featVals.some(v => !Number.isFinite(v))) { $("#predict-output").textContent = "Provide all features: " + state.featureCols.join(", "); return; }
    const X = [featVals];
    const Xs = applyStandardize(X, state.scaler.mu, state.scaler.sd);
    let out;
    if (state.lastModelType==="regression") {
      const yhat = linearPredict(Xs, state.model.model)[0];
      out = { prediction: yhat };
    } else if (state.lastModelType==="classification") {
      const p = logisticPredictProba(Xs, state.model.model)[0];
      out = { probability: p, predicted_class: p>=0.5?1:0 };
    } else if (state.lastModelType==="clustering") {
      // assign nearest centroid
      const km = state.model.km;
      let best=0, bestDist=Infinity;
      for (let c=0;c<km.centroids.length;c++){
        let dist=0;
        for (let j=0;j<Xs[0].length;j++){ const diff=Xs[0][j]-km.centroids[c][j]; dist+=diff*diff; }
        if (dist<bestDist){ bestDist=dist; best=c; }
      }
      out = { cluster: best, distance: bestDist };
    }
    $("#predict-output").textContent = JSON.stringify(out, null, 2);
  } catch(e) {
    $("#predict-output").textContent = "Error: " + e.message;
  }
});

$("#btn-export").addEventListener("click", () => {
  if (!state.model) { $("#export-status").textContent = "Train a model first."; return; }
  const payload = JSON.stringify({
    type: state.model.type,
    featureCols: state.featureCols,
    scaler: state.scaler,
    model: state.model.model || state.model.km
  }, null, 2);
  const blob = new Blob([payload], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "automl_model.json";
  a.click();
  setTimeout(()=> URL.revokeObjectURL(url), 2000);
  $("#export-status").textContent = "Exported automl_model.json";
});
