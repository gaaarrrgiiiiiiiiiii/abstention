/**
 * Abstention Classifier — Frontend Application
 * 
 * Dynamically fetches feature names and sample data from the API.
 * Handles form submission, result rendering, and prediction history.
 */

(() => {
    'use strict';

    // ============================================================
    // Configuration
    // ============================================================
    const API_BASE = 'http://localhost:5000/api';
    
    // State
    let featureNames = [];
    let sampleData = [];
    let predictionHistory = [];
    let confidenceChart = null;

    // ============================================================
    // DOM References
    // ============================================================
    const $featureGrid   = document.getElementById('feature-grid');
    const $sampleButtons = document.getElementById('sample-buttons');
    const $predictForm   = document.getElementById('predict-form');
    const $btnPredict    = document.getElementById('btn-predict');
    const $btnClear      = document.getElementById('btn-clear');
    const $resultPanel   = document.getElementById('result-panel');
    const $historyTbody  = document.getElementById('history-tbody');
    const $historyEmpty  = document.getElementById('history-empty');
    const $statFeatures  = document.getElementById('stat-features');
    const $statStatus    = document.getElementById('stat-status');

    // ============================================================
    // Initialization
    // ============================================================
    async function init() {
        try {
            // Health check
            const health = await apiGet('/health');
            $statStatus.textContent = health.model_loaded ? 'Online' : 'Offline';

            // Fetch feature names
            const featureData = await apiGet('/features');
            featureNames = featureData.features;
            $statFeatures.textContent = featureData.count;
            renderFeatureInputs(featureNames);

            // Fetch sample data
            const sampleResponse = await apiGet('/sample-data');
            sampleData = sampleResponse.samples;
            renderSampleButtons(sampleData);

        } catch (err) {
            console.error('Initialization failed:', err);
            $statStatus.textContent = 'Offline';
            showToast('Cannot connect to API. Please ensure the Flask server is running on port 5000.');
        }

        // Event listeners
        $predictForm.addEventListener('submit', handlePredict);
        $btnClear.addEventListener('click', clearForm);

        // Smooth scroll nav
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.getElementById(link.dataset.target);
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
                document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                link.classList.add('active');
            });
        });
    }

    // ============================================================
    // API Helpers
    // ============================================================
    async function apiGet(endpoint) {
        const res = await fetch(API_BASE + endpoint);
        if (!res.ok) throw new Error(`GET ${endpoint} failed: ${res.status}`);
        return res.json();
    }

    async function apiPost(endpoint, body) {
        const res = await fetch(API_BASE + endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `POST ${endpoint} failed: ${res.status}`);
        return data;
    }

    // ============================================================
    // Render Feature Inputs (Dynamic)
    // ============================================================
    function renderFeatureInputs(features) {
        $featureGrid.innerHTML = '';
        features.forEach((name, index) => {
            const field = document.createElement('div');
            field.className = 'feature-field';
            
            const label = document.createElement('label');
            label.setAttribute('for', `feature-${index}`);
            label.textContent = name;
            
            const input = document.createElement('input');
            input.type = 'number';
            input.id = `feature-${index}`;
            input.name = `feature-${index}`;
            input.step = 'any';
            input.placeholder = '0.00';
            input.setAttribute('data-index', index);
            input.required = true;

            // Visual feedback on input
            input.addEventListener('input', () => {
                if (input.value !== '') {
                    input.classList.add('has-value');
                    input.classList.remove('error');
                } else {
                    input.classList.remove('has-value');
                }
            });

            field.appendChild(label);
            field.appendChild(input);
            $featureGrid.appendChild(field);
        });
    }

    // ============================================================
    // Render Sample Buttons (Dynamic)
    // ============================================================
    function renderSampleButtons(samples) {
        $sampleButtons.innerHTML = '';
        samples.forEach((sample, idx) => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'btn btn-sample';
            btn.textContent = sample.name;
            btn.title = sample.description;
            btn.addEventListener('click', () => fillSample(sample.features));
            $sampleButtons.appendChild(btn);
        });
    }

    // ============================================================
    // Fill Sample Data
    // ============================================================
    function fillSample(values) {
        values.forEach((val, idx) => {
            const input = document.getElementById(`feature-${idx}`);
            if (input) {
                input.value = val;
                input.classList.add('has-value');
                input.classList.remove('error');
            }
        });

        // Scroll to the predict button area
        $btnPredict.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    // ============================================================
    // Clear Form
    // ============================================================
    function clearForm() {
        featureNames.forEach((_, idx) => {
            const input = document.getElementById(`feature-${idx}`);
            if (input) {
                input.value = '';
                input.classList.remove('has-value', 'error');
            }
        });
        $resultPanel.style.display = 'none';
    }

    // ============================================================
    // Handle Prediction
    // ============================================================
    async function handlePredict(e) {
        e.preventDefault();

        // Collect feature values
        const features = [];
        let hasError = false;

        featureNames.forEach((_, idx) => {
            const input = document.getElementById(`feature-${idx}`);
            const val = parseFloat(input.value);
            if (isNaN(val)) {
                input.classList.add('error');
                hasError = true;
            } else {
                input.classList.remove('error');
                features.push(val);
            }
        });

        if (hasError) {
            showToast('Please fill in all feature fields with valid numeric values.');
            return;
        }

        // Show loading state
        $btnPredict.classList.add('loading');
        $btnPredict.disabled = true;

        try {
            const result = await apiPost('/predict', { features });
            renderResult(result);
            addToHistory(result);
        } catch (err) {
            showToast('Prediction failed: ' + err.message);
        } finally {
            $btnPredict.classList.remove('loading');
            $btnPredict.disabled = false;
        }
    }

    // ============================================================
    // Render Prediction Result
    // ============================================================
    function renderResult(result) {
        $resultPanel.style.display = 'block';
        $resultPanel.style.animation = 'none';
        // Trigger reflow for re-animation
        void $resultPanel.offsetHeight;
        $resultPanel.style.animation = 'slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards';

        // Icon
        const $icon = document.getElementById('result-icon');
        $icon.className = 'result-icon'; // Reset

        const svgIcons = {
            legit: `<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 16l5 5 9-10"/></svg>`,
            fraud: `<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10 10l12 12M22 10l-12 12"/></svg>`,
            abstain: `<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="16" cy="16" r="10"/><path d="M16 12v5M16 20v0.5"/></svg>`
        };

        if (result.prediction_code === 0) {
            $icon.classList.add('icon-legit');
            $icon.innerHTML = svgIcons.legit;
        } else if (result.prediction_code === 1) {
            $icon.classList.add('icon-fraud');
            $icon.innerHTML = svgIcons.fraud;
        } else {
            $icon.classList.add('icon-abstain');
            $icon.innerHTML = svgIcons.abstain;
        }

        // Label
        document.getElementById('result-label').textContent = result.prediction;
        document.getElementById('result-decision').textContent = result.should_decide
            ? 'The model has decided to commit to this prediction.'
            : 'The model chose to abstain. Human review recommended.';

        // Confidence bars
        const legitPct  = (result.confidence.legitimate * 100).toFixed(2);
        const fraudPct  = (result.confidence.fraud * 100).toFixed(2);
        const abstainPct = (result.confidence.abstain * 100).toFixed(2);

        // We use setTimeout to allow CSS transition to work
        setTimeout(() => {
            document.getElementById('conf-legit').style.width = legitPct + '%';
            document.getElementById('conf-fraud').style.width = fraudPct + '%';
            document.getElementById('conf-abstain').style.width = abstainPct + '%';
        }, 50);

        document.getElementById('conf-legit-val').textContent = legitPct + '%';
        document.getElementById('conf-fraud-val').textContent = fraudPct + '%';
        document.getElementById('conf-abstain-val').textContent = abstainPct + '%';

        // Recommendation
        document.getElementById('recommendation-text').textContent = result.recommendation;

        // Doughnut Chart
        renderConfidenceChart(result.confidence);

        // Scroll to results
        $resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ============================================================
    // Confidence Doughnut Chart
    // ============================================================
    function renderConfidenceChart(confidence) {
        const ctx = document.getElementById('confidenceChart').getContext('2d');

        if (confidenceChart) {
            confidenceChart.destroy();
        }

        confidenceChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Legitimate', 'Fraud', 'Abstain'],
                datasets: [{
                    data: [
                        (confidence.legitimate * 100).toFixed(2),
                        (confidence.fraud * 100).toFixed(2),
                        (confidence.abstain * 100).toFixed(2)
                    ],
                    backgroundColor: [
                        '#1B7A4A',
                        '#B3261E',
                        '#B8860B'
                    ],
                    borderColor: '#FFFFFF',
                    borderWidth: 3,
                    hoverBorderWidth: 0,
                    hoverOffset: 8
                }]
            },
            options: {
                responsive: false,
                cutout: '62%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            font: {
                                family: "'Times New Roman', Times, Georgia, serif",
                                size: 13,
                                weight: '600'
                            },
                            color: '#4A5568',
                            padding: 16,
                            usePointStyle: true,
                            pointStyleWidth: 10
                        }
                    },
                    tooltip: {
                        backgroundColor: '#1F2733',
                        titleFont: {
                            family: "'Times New Roman', Times, Georgia, serif",
                            size: 13
                        },
                        bodyFont: {
                            family: "'Times New Roman', Times, Georgia, serif",
                            size: 13
                        },
                        padding: 12,
                        cornerRadius: 8,
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.parsed + '%';
                            }
                        }
                    }
                },
                animation: {
                    animateRotate: true,
                    duration: 800
                }
            }
        });
    }

    // ============================================================
    // Prediction History
    // ============================================================
    function addToHistory(result) {
        predictionHistory.push({
            ...result,
            timestamp: new Date()
        });

        renderHistory();
    }

    function renderHistory() {
        if (predictionHistory.length === 0) {
            $historyEmpty.style.display = 'block';
            $historyTbody.parentElement.style.display = 'none';
            return;
        }

        $historyEmpty.style.display = 'none';
        $historyTbody.parentElement.style.display = 'table';
        $historyTbody.innerHTML = '';

        predictionHistory.forEach((entry, idx) => {
            const tr = document.createElement('tr');
            tr.className = 'fade-in';

            const predClass = entry.prediction_code === 0 ? 'badge-legit'
                            : entry.prediction_code === 1 ? 'badge-fraud'
                            : 'badge-abstain';

            const decisionBadge = entry.should_decide
                ? '<span class="badge badge-decide">Decide</span>'
                : '<span class="badge badge-defer">Defer</span>';

            const timeStr = entry.timestamp.toLocaleTimeString();

            tr.innerHTML = `
                <td>${idx + 1}</td>
                <td><span class="${predClass}">${entry.prediction}</span></td>
                <td>${(entry.confidence.legitimate * 100).toFixed(2)}%</td>
                <td>${(entry.confidence.fraud * 100).toFixed(2)}%</td>
                <td>${(entry.confidence.abstain * 100).toFixed(2)}%</td>
                <td>${decisionBadge}</td>
                <td>${timeStr}</td>
            `;

            $historyTbody.appendChild(tr);
        });
    }

    // ============================================================
    // Toast Notification
    // ============================================================
    function showToast(message, duration = 4000) {
        // Remove existing toast
        const existing = document.querySelector('.error-toast');
        if (existing) existing.remove();

        const toast = document.createElement('div');
        toast.className = 'error-toast';
        toast.textContent = message;
        document.body.appendChild(toast);

        // Trigger show
        requestAnimationFrame(() => {
            toast.classList.add('visible');
        });

        // Auto-hide
        setTimeout(() => {
            toast.classList.remove('visible');
            setTimeout(() => toast.remove(), 400);
        }, duration);
    }

    // ============================================================
    // Start
    // ============================================================
    document.addEventListener('DOMContentLoaded', init);

})();
