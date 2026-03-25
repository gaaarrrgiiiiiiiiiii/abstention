let globalChartInstances = {};

async function fetchResultsData() {
    try {
        const response = await fetch('data/results.json');
        if (!response.ok) throw new Error("Could not fetch results.json");
        const data = await response.json();
        
        // 1. Update KPI Cards
        updateKPICards(data.final_results, data.experiment_metrics);
        
        // 2. Populate Table
        populateTable(data.final_results);
        
        // 3. Render Charts (from charts.js)
        renderCharts(data);
        
    } catch (err) {
        console.error("Error loading dashboard data:", err);
        alert("Failed to load dashboard data. Ensure you are serving the files via an HTTP server.");
    }
}

function updateKPICards(finalResults, experimentMetrics) {
    if (!finalResults || finalResults.length === 0) return;
    
    // Best F1
    let bestF1Model = finalResults[0];
    for (let r of finalResults) {
        if (r['F1 Score'] >= bestF1Model['F1 Score']) bestF1Model = r;
    }
    
    // Avg Coverage (only across abstaining models, meaning Coverage < 1.0)
    let totalCov = 0;
    let covCount = 0;
    for (let r of finalResults) {
        if (r['Coverage'] < 1.0) {
            totalCov += r['Coverage'];
            covCount++;
        }
    }
    const avgCov = covCount > 0 ? (totalCov / covCount) : 1.0;
    
    // Lowest Risk
    let lowestRiskModel = finalResults[0];
    for (let r of finalResults) {
        if (r['Selective Risk'] < lowestRiskModel['Selective Risk']) lowestRiskModel = r;
    }
    
    // Memory
    let maxMem = 0;
    for (let i = 1; i <= 4; i++) {
        const expData = experimentMetrics[`exp_${i}`];
        if (expData) {
            for (let row of expData) {
                if (row['process_memory_mb'] > maxMem) maxMem = row['process_memory_mb'];
            }
        }
    }

    animateValue("kpi-f1", 0, bestF1Model['F1 Score'], 1000, true);
    document.getElementById("kpi-f1-model").innerText = bestF1Model['Model Name'];
    
    animateValue("kpi-cov", 0, avgCov * 100, 1000, false, "%");
    
    animateValue("kpi-risk", 100, lowestRiskModel['Selective Risk'] * 100, 1000, false, "%");
    document.getElementById("kpi-risk-model").innerText = lowestRiskModel['Model Name'];
    
    animateValue("kpi-mem", 0, maxMem, 1000, false, " MB");
}

function populateTable(results) {
    const tbody = document.querySelector("#resultsTable tbody");
    tbody.innerHTML = '';
    
    results.forEach(row => {
        const tr = document.createElement('tr');
        
        // Model Name
        let html = `<td><strong>${row['Model Name']}</strong></td>`;
        
        // Accuracy
        html += `<td>${(row['Accuracy'] * 100).toFixed(2)}%</td>`;
        
        // Coverage
        const cov = row['Coverage'] * 100;
        const covClass = cov === 100 ? '' : 'highlight-neutral';
        html += `<td class="${covClass}">${cov.toFixed(2)}%</td>`;
        
        // Risk
        html += `<td>${(row['Selective Risk'] * 100).toFixed(4)}%</td>`;
        
        // ECE
        html += `<td>${row['ECE'].toFixed(4)}</td>`;
        
        // F1 
        const f1 = row['F1 Score'];
        let f1Class = '';
        if (f1 > 0.8) f1Class = 'highlight-good';
        else if (f1 === 0) f1Class = 'highlight-bad';
        html += `<td class="${f1Class}">${f1.toFixed(4)}</td>`;
        
        tr.innerHTML = html;
        tbody.appendChild(tr);
    });
}

function animateValue(id, start, end, duration, isFloat=false, append="") {
    const obj = document.getElementById(id);
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const current = progress * (end - start) + start;
        if (isFloat) {
            obj.innerHTML = current.toFixed(4) + append;
        } else {
            obj.innerHTML = current.toFixed(1) + append;
        }
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Call fetch on load
window.onload = fetchResultsData;
