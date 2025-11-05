/* Robust client JS for GeneGuess (debug-friendly) */
(function(){
  // safe element helpers
  const el = (q)=> document.querySelector(q);
  const els = (q)=> Array.from(document.querySelectorAll(q));

  function safeLog(...args){ try{ console.log('[GeneGuess]', ...args); }catch(e){} }

  // Build or find chart safely
  let contribChart = null;
  function createChartIfNeeded(){
    try{
      const ctx = el('#contribChart') && el('#contribChart').getContext('2d');
      if(!ctx){ safeLog('No canvas #contribChart found'); return null; }
      if(typeof Chart === 'undefined'){
        safeLog('Chart.js not loaded; continuing without chart.');
        return null;
      }
      contribChart = new Chart(ctx, {
        type: 'bar',
        data: { labels: [], datasets: [{ label: 'Contribution', data: [], backgroundColor: [] }] },
        options: { indexAxis:'y', responsive:true, plugins:{legend:{display:false}}, scales:{x:{beginAtZero:true}} }
      });
      return contribChart;
    }catch(err){
      safeLog('Error creating chart:', err);
      return null;
    }
  }

  // show/hide helpers
  function showResultCard(){
    const r = el('#resultCard');
    if(!r) return;
    r.style.display = 'block';
    r.classList.remove('d-none');
  }
  function hideResultCard(){
    const r = el('#resultCard');
    if(!r) return;
    r.style.display = 'none';
    r.classList.add('d-none');
  }

  // render contributions list and chart
  function renderContribs(contribs){
    const labels = Object.keys(contribs || {});
    const values = labels.map(k => Number(contribs[k] || 0));
    // update list view
    const list = el('#contribList');
    if(list){
      list.innerHTML = '';
      labels.forEach(k=>{
        const v = contribs[k] || 0;
        const row = document.createElement('div');
        row.className = 'contrib-row';
        row.innerHTML = `<div class="contrib-name">${k.replace('_',' ')}</div><div class="contrib-value">${(v>=0?'+':'')}${Number(v).toFixed(4)}</div>`;
        list.appendChild(row);
      });
    }
    // update chart if exists
    if(contribChart){
      try{
        const colors = values.map(v => v>=0 ? 'rgba(178,31,45,0.85)' : 'rgba(31,143,178,0.85)');
        contribChart.data.labels = labels;
        contribChart.data.datasets[0].data = values;
        contribChart.data.datasets[0].backgroundColor = colors;
        contribChart.update();
      }catch(e){
        safeLog('Error updating chart:', e);
      }
    }
  }

  // show result object
  function showResult(data){
    if(!data){ safeLog('showResult called with no data'); return; }
    showResultCard();
    const pct = (data.probability*100).toFixed(1);
    if(el('#probPct')) el('#probPct').innerText = pct + '%';
    if(el('#riskLabel')){
      el('#riskLabel').innerText = data.probability >= 0.5 ? 'Higher risk' : 'Lower risk';
      el('#riskLabel').className = data.probability >= 0.5 ? 'label-high' : 'label-low';
    }
    renderContribs(data.contributions || {});
    window.lastResult = data;
  }

  // submit handler
  async function submitForm(e){
    e && e.preventDefault && e.preventDefault();
    safeLog('submitForm fired');
    // gather form data
    const formEl = el('#frm');
    if(!formEl){ alert('Form element not found on page'); return; }
    const form = new FormData(formEl);

    // basic client-side validation
    const age = Number(form.get('age')||0);
    if(isNaN(age) || age < 0 || age > 120){ alert('Age must be between 0 and 120'); return; }

    // send to server
    let resp;
    try{
      resp = await fetch('/predict', { method:'POST', body: form });
    }catch(err){
      safeLog('Network error while fetching /predict', err);
      alert('Network error: could not reach server. Is Flask running? See console for details.');
      return;
    }

    // parse response
    let data;
    try{
      data = await resp.json();
    }catch(err){
      const text = await resp.text().catch(()=>'<no body>');
      safeLog('Failed to parse JSON from /predict, status:', resp.status, 'body:', text);
      alert('Server returned invalid response (see console).');
      return;
    }

    if(resp.status !== 200 || data.error){
      safeLog('Server returned error:', resp.status, data);
      alert('Server error: ' + (data.error || ('status ' + resp.status)));
      return;
    }

    safeLog('Server response:', data);
    showResult(data);
  }

  // copy & download helpers
  async function copyResults(){
    if(!window.lastResult){ alert('No results yet'); return; }
    const txt = `GeneGuess result:\\nProbability: ${(window.lastResult.probability*100).toFixed(1)}%\\nInput: ${JSON.stringify(window.lastResult.sanitized_input)}`;
    try{ await navigator.clipboard.writeText(txt); alert('Copied summary to clipboard'); }catch(e){ alert('Copy failed'); safeLog(e); }
  }
  function downloadReport(){
    if(!window.lastResult){ alert('No results to download'); return; }
    const w = window.open('','_blank');
    const html = `<html><head><title>GeneGuess Report</title></head><body><h1>GeneGuess Report</h1><p><strong>Probability:</strong> ${(window.lastResult.probability*100).toFixed(1)}%</p><pre>${JSON.stringify(window.lastResult,null,2)}</pre></body></html>`;
    w.document.write(html); w.document.close(); w.print();
  }

  // modal open/close
  function openExplain(){
    if(!window.lastResult){ alert('Make a prediction first to see the explanation.'); return; }
    const modal = el('#modal'); if(!modal) return;
    modal.classList.add('show'); modal.setAttribute('aria-hidden','false');
    const body = el('#modal .box .modal-body'); if(!body) return;
    body.innerHTML = `<p><strong>Probability:</strong> ${(window.lastResult.probability*100).toFixed(1)}%</p><p><strong>Input:</strong> ${JSON.stringify(window.lastResult.sanitized_input)}</p>`;
    const contribs = window.lastResult.contributions || {};
    const container = document.createElement('div');
    Object.keys(contribs).forEach(k=>{
      const v = contribs[k];
      const row = document.createElement('div'); row.style.display='flex'; row.style.justifyContent='space-between';
      row.style.padding='4px 0'; row.innerHTML = `<div style="color:#9aa7b2">${k.replace('_',' ')}</div><div style="font-weight:700">${(v>=0?'+':'')}${Number(v).toFixed(4)}</div>`;
      container.appendChild(row);
    });
    body.appendChild(container);
  }
  function closeModal(){ const modal = el('#modal'); if(modal){ modal.classList.remove('show'); modal.setAttribute('aria-hidden','true'); } }

  // initialize on DOM ready
  document.addEventListener('DOMContentLoaded', function(){
    safeLog('DOM ready - initializing GeneGuess client');
    // create chart if possible
    createChartIfNeeded();

    const form = el('#frm');
    if(form){
      form.addEventListener('submit', submitForm);
      safeLog('Attached submit handler to #frm');
    } else {
      safeLog('Form #frm not found');
    }
    const btnExplain = el('#btnExplain'); if(btnExplain) btnExplain.addEventListener('click', openExplain);
    const btnCopy = el('#btnCopy'); if(btnCopy) btnCopy.addEventListener('click', copyResults);
    const btnDownload = el('#btnDownload'); if(btnDownload) btnDownload.addEventListener('click', downloadReport);
    const modalClose = el('#modalClose'); if(modalClose) modalClose.addEventListener('click', closeModal);
  });
})();
