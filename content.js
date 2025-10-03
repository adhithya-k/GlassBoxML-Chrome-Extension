
(function() {
  function tableToJSON(table) {
    const headers = [];
    const headerCells = table.querySelectorAll("thead th, tr:first-child th, tr:first-child td");
    if (headerCells.length) {
      headerCells.forEach(h => headers.push(h.innerText.trim()));
    }
    const rows = [];
    const trs = table.querySelectorAll("tbody tr, tr");
    trs.forEach((tr, rIdx) => {
      const cells = tr.querySelectorAll("td");
      if (cells.length === 0) return;
      const row = {};
      cells.forEach((td, i) => {
        const key = headers[i] || ("col_" + i);
        row[key] = td.innerText.trim();
      });
      rows.push(row);
    });
    return { columns: headers.length ? headers : Object.keys(rows[0] || {}), rows };
  }

  function scrapeAllTables() {
    const tables = Array.from(document.querySelectorAll("table"));
    const datasets = tables.map((t, i) => {
      try {
        return { name: "table_" + (i+1), ...tableToJSON(t) };
      } catch(e) {
        return { name: "table_" + (i+1), error: e.message, columns: [], rows: [] };
      }
    });
    return datasets;
  }

  // Listen for ping from popup to fetch tables
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg && msg.type === "SCRAPE_TABLES") {
      const data = scrapeAllTables();
      sendResponse({ datasets: data });
      return true;
    }
  });
})();
