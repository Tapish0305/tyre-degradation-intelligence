"""
Standalone Flask Web Dashboard for F1 Tyre Degradation Intelligence.
Includes interactive Plotly.js charts, checkpoint tables, and REST API.
"""

from flask import Flask, render_template_string, jsonify, request
import numpy as np
import pandas as pd
import json

from src import dashboard_data

app = Flask(__name__)

# Global cache for fitted dashboard data
_GLOBAL_DATA = None

def get_data():
    global _GLOBAL_DATA
    if _GLOBAL_DATA is None:
        _GLOBAL_DATA = dashboard_data.get_or_fit_dashboard_data()
    return _GLOBAL_DATA

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>F1 Tyre Degradation Intelligence Dashboard</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Plotly.js CDN -->
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        f1red: '#E10600',
                        darkbg: '#0F172A',
                        cardbg: '#1E293B',
                        bordercolor: '#334155'
                    }
                }
            }
        }
    </script>
    <style>
        .active-tab {
            border-bottom: 3px solid #E10600;
            color: #F8FAFC;
            font-weight: 700;
        }
    </style>
</head>
<body class="bg-darkbg text-slate-100 min-h-screen font-sans antialiased relative">
    
    <!-- Full-screen Loading Overlay -->
    <div id="loadingOverlay" class="absolute inset-0 bg-darkbg/90 z-[9999] flex flex-col items-center justify-center transition-opacity duration-500">
        <svg class="animate-spin -ml-1 mr-3 h-12 w-12 text-f1red mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        <h2 class="text-xl font-bold text-slate-100 mb-2">🏎️ Compiling Models & Simulating Data...</h2>
        <p class="text-slate-400 max-w-md text-center">First-time load will take ~60 seconds to execute the Bayesian State-Space Models via NumPyro.</p>
    </div>

    <!-- Navbar -->
    <header class="bg-cardbg border-b border-bordercolor sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex justify-between items-center">
            <div class="flex items-center space-x-3">
                <span class="text-2xl font-black text-f1red tracking-wider">F1</span>
                <div class="h-6 w-px bg-bordercolor"></div>
                <div>
                    <h1 class="text-lg font-bold tracking-tight">TYRE DEGRADATION INTELLIGENCE</h1>
                    <p class="text-xs text-slate-400">Bayesian State-Space Isolation: Fuel, Traffic & Track Evolution</p>
                </div>
            </div>
            <div class="flex items-center space-x-3">
                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-900 text-green-300">
                    ● Model Engine Ready (NumPyro/JAX)
                </span>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        
        <!-- Controls Bar -->
        <div class="bg-cardbg rounded-xl p-4 border border-bordercolor shadow-lg grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
                <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Grand Prix Event</label>
                <select id="gpSelect" class="w-full bg-slate-900 border border-bordercolor rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-f1red">
                    <option value="monza">2024 Italian Grand Prix (Monza)</option>
                    <option value="austria">2024 Austrian Grand Prix (Spielberg)</option>
                </select>
            </div>
            <div>
                <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Driver</label>
                <select id="driverSelect" onchange="updateDashboard()" class="w-full bg-slate-900 border border-bordercolor rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-f1red">
                    <option value="HAM">Lewis Hamilton (HAM)</option>
                    <option value="LEC">Charles Leclerc (LEC)</option>
                    <option value="VER">Max Verstappen (VER)</option>
                    <option value="NOR">Lando Norris (NOR)</option>
                </select>
            </div>
            <div>
                <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Pit Loss Delta: <span id="pitDeltaVal">22.0</span>s</label>
                <input type="range" id="pitLossSlider" min="15" max="30" step="0.5" value="22" oninput="document.getElementById('pitDeltaVal').innerText = this.value; updateDashboard();" class="w-full accent-f1red cursor-pointer">
            </div>
            <div>
                <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Checkpoint Step: <span id="cpStepVal">5</span> Laps</label>
                <input type="range" id="cpStepSlider" min="2" max="10" step="1" value="5" oninput="document.getElementById('cpStepVal').innerText = this.value; updateDashboard();" class="w-full accent-f1red cursor-pointer">
            </div>
        </div>

        <!-- KPI Summary Cards Row -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div class="bg-cardbg rounded-xl p-4 border-l-4 border-f1red border-bordercolor shadow">
                <div class="text-xs text-slate-400 uppercase font-semibold">Active Wear Rate (ν)</div>
                <div id="kpiWear" class="text-2xl font-black text-slate-100 mt-1">+0.049 s/lap</div>
                <div id="kpiWearSub" class="text-xs text-slate-500 mt-1">95% CI: [0.004, 0.133] • HARD</div>
            </div>
            <div class="bg-cardbg rounded-xl p-4 border-l-4 border-blue-500 border-bordercolor shadow">
                <div class="text-xs text-slate-400 uppercase font-semibold">Optimal Pit Window</div>
                <div id="kpiPit" class="text-2xl font-black text-slate-100 mt-1">Lap 18 – 23</div>
                <div id="kpiPitSub" class="text-xs text-slate-500 mt-1">Crossover Lap: 20 • Pit Delta: 22.0s</div>
            </div>
            <div class="bg-cardbg rounded-xl p-4 border-l-4 border-green-500 border-bordercolor shadow">
                <div class="text-xs text-slate-400 uppercase font-semibold">Prediction Trust Index</div>
                <div id="kpiTrust" class="text-2xl font-black text-green-400 mt-1">94.2%</div>
                <div id="kpiTrustSub" class="text-xs text-slate-500 mt-1">HIGH CONFIDENCE • 91.8% In-Band</div>
            </div>
            <div class="bg-cardbg rounded-xl p-4 border-l-4 border-purple-500 border-bordercolor shadow">
                <div class="text-xs text-slate-400 uppercase font-semibold">Out-of-Session RMSPE</div>
                <div id="kpiRMSPE" class="text-2xl font-black text-slate-100 mt-1">3.05 s</div>
                <div id="kpiRMSPESub" class="text-xs text-slate-500 mt-1">FP2 -> Race Validation • CRPS: 2.20s</div>
            </div>
        </div>

        <!-- Tabs Navigation -->
        <div class="border-b border-bordercolor flex space-x-4 text-sm font-semibold text-slate-400 overflow-x-auto whitespace-nowrap">
            <button onclick="switchTab('tab1')" id="btn-tab1" class="py-3 px-1 active-tab focus:outline-none">📉 Q1: Rate of Loss & Pit</button>
            <button onclick="switchTab('tab2')" id="btn-tab2" class="py-3 px-1 hover:text-slate-200 focus:outline-none">⏱️ Q2: Checkpoints</button>
            <button onclick="switchTab('tab3')" id="btn-tab3" class="py-3 px-1 hover:text-slate-200 focus:outline-none">🛡️ Q3: Trustworthiness</button>
            <button onclick="switchTab('tab4')" id="btn-tab4" class="py-3 px-1 hover:text-slate-200 focus:outline-none">🏎️ Multi-Driver & Compounds</button>
            <button onclick="switchTab('tab5')" id="btn-tab5" class="py-3 px-1 hover:text-slate-200 focus:outline-none">🌦️ Weather</button>
            <button onclick="switchTab('tab6')" id="btn-tab6" class="py-3 px-1 hover:text-slate-200 focus:outline-none">🧭 Driver Style</button>
            <button onclick="switchTab('tab7')" id="btn-tab7" class="py-3 px-1 hover:text-slate-200 focus:outline-none">🔍 Data Exclusion Audit</button>
        </div>

        <!-- TAB 1: QUESTION 1 -->
        <div id="tab1-content" class="space-y-6">
            <div class="bg-slate-800/60 border-l-4 border-blue-500 rounded-lg p-4 text-sm text-slate-300">
                <span class="font-bold text-white">Mentor Core Question 1:</span> How quickly is the driver losing tyre performance, lap over lap? At what point would the driver reasonably call for a pit stop?
            </div>
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div class="lg:col-span-2 bg-cardbg rounded-xl p-4 border border-bordercolor">
                    <h3 class="text-sm font-bold text-slate-200 uppercase mb-3">Clean Latent Tyre Pace α vs Observed Lap Times</h3>
                    <div id="plotPace" style="height: 380px;"></div>
                </div>
                <div class="bg-cardbg rounded-xl p-4 border border-bordercolor">
                    <h3 class="text-sm font-bold text-slate-200 uppercase mb-3">Cumulative Deficit & Pit Crossover</h3>
                    <div id="plotCross" style="height: 380px;"></div>
                </div>
            </div>
            <div id="q1SummaryBox" class="bg-cardbg rounded-xl p-4 border border-bordercolor text-sm text-slate-300">
                <!-- Dynamically populated -->
            </div>
        </div>

        <!-- TAB 2: QUESTION 2 -->
        <div id="tab2-content" class="space-y-6 hidden">
            <div class="bg-slate-800/60 border-l-4 border-blue-500 rounded-lg p-4 text-sm text-slate-300">
                <span class="font-bold text-white">Mentor Core Question 2:</span> How has tyre performance changed each lap at defined intervals/checkpoints (not just an abstract curve)?
            </div>
            <div class="bg-cardbg rounded-xl p-4 border border-bordercolor overflow-x-auto">
                <h3 class="text-sm font-bold text-slate-200 uppercase mb-3">Stint Checkpoint Degradation Table</h3>
                <table class="w-full text-left text-xs text-slate-300">
                    <thead class="bg-slate-900 text-slate-400 uppercase font-semibold">
                        <tr>
                            <th class="py-2.5 px-3">Checkpoint</th>
                            <th class="py-2.5 px-3">Observed (s)</th>
                            <th class="py-2.5 px-3">Latent Pace α (s)</th>
                            <th class="py-2.5 px-3">Wear Delta Δt</th>
                            <th class="py-2.5 px-3">Fuel Effect</th>
                            <th class="py-2.5 px-3">Traffic Loss</th>
                            <th class="py-2.5 px-3">Tyre Life %</th>
                            <th class="py-2.5 px-3">Strategist Action</th>
                        </tr>
                    </thead>
                    <tbody id="checkpointTbody" class="divide-y divide-bordercolor">
                        <!-- Dynamically populated -->
                    </tbody>
                </table>
            </div>
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div class="bg-cardbg rounded-xl p-4 border border-bordercolor">
                    <h3 class="text-sm font-bold text-slate-200 uppercase mb-3">Component Breakdown at Checkpoints</h3>
                    <div id="plotComponents" style="height: 300px;"></div>
                </div>
                <div class="bg-cardbg rounded-xl p-4 border border-bordercolor">
                    <h3 class="text-sm font-bold text-slate-200 uppercase mb-3">Remaining Tyre Life %</h3>
                    <div id="plotLife" style="height: 300px;"></div>
                </div>
            </div>
        </div>

        <!-- TAB 3: QUESTION 3 -->
        <div id="tab3-content" class="space-y-6 hidden">
            <div class="bg-slate-800/60 border-l-4 border-blue-500 rounded-lg p-4 text-sm text-slate-300">
                <span class="font-bold text-white">Mentor Core Question 3:</span> Can the prediction be trusted? Where is the validation evidence and justified confidence quantification?
            </div>
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div class="lg:col-span-2 bg-cardbg rounded-xl p-4 border border-bordercolor">
                    <h3 class="text-sm font-bold text-slate-200 uppercase mb-3">Post-Race Validation: Predicted Pace with 90%/95% Credible Envelopes</h3>
                    <div id="plotVal" style="height: 380px;"></div>
                </div>
                <div class="bg-cardbg rounded-xl p-4 border border-bordercolor space-y-4">
                    <h3 class="text-sm font-bold text-slate-200 uppercase">Trust Justification</h3>
                    <div id="trustJustText" class="text-xs text-slate-300 bg-slate-900 p-3 rounded-lg border border-bordercolor">
                        <!-- Dynamically populated -->
                    </div>
                    <h3 class="text-sm font-bold text-slate-200 uppercase pt-2">Benchmark Comparison</h3>
                    <table class="w-full text-left text-xs text-slate-300">
                        <thead class="bg-slate-900 text-slate-400">
                            <tr>
                                <th class="py-1.5 px-2">Model</th>
                                <th class="py-1.5 px-2">RMSPE</th>
                                <th class="py-1.5 px-2">CRPS</th>
                            </tr>
                        </thead>
                        <tbody id="benchmarkTbody" class="divide-y divide-bordercolor">
                            <!-- Populated dynamically -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- TAB 4: MULTI-DRIVER PATTERNS -->
        <div id="tab4-content" class="space-y-6 hidden">
            <div class="bg-slate-800/60 border-l-4 border-blue-500 rounded-lg p-4 text-sm text-slate-300">
                <span class="font-bold text-white">Pattern Analysis:</span> Multi-driver wear comparison across the cohort, compound degradation hierarchy, and session track evolution grip buildup.
            </div>
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div class="bg-cardbg rounded-xl p-4 border border-bordercolor">
                    <h3 class="text-sm font-bold text-slate-200 uppercase mb-3">Compound Wear Rates (Hard vs Medium vs Soft)</h3>
                    <div id="plotCompounds" style="height: 320px;"></div>
                </div>
                <div class="bg-cardbg rounded-xl p-4 border border-bordercolor">
                    <h3 class="text-sm font-bold text-slate-200 uppercase mb-3">Session Track Evolution Grip Buildup</h3>
                    <div id="plotTrack" style="height: 320px;"></div>
                </div>
            </div>
        </div>

        <!-- TAB 5: WEATHER EFFECTS -->
        <div id="tab5-content" class="space-y-6 hidden">
            <div class="bg-slate-800/60 border-l-4 border-blue-500 rounded-lg p-4 text-sm text-slate-300">
                <span class="font-bold text-white">Weather Modeling:</span> The hierarchical SSM fits track/air temperature, humidity, and wind-speed effects. This tab shows the raw conditions and posterior distributions.
            </div>
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div class="bg-cardbg rounded-xl p-4 border border-bordercolor">
                    <h3 class="text-sm font-bold text-slate-200 uppercase mb-3">🌡️ Track & Air Temp Trend (Practice)</h3>
                    <div id="plotWTemp" style="height: 320px;"></div>
                </div>
                <div class="bg-cardbg rounded-xl p-4 border border-bordercolor">
                    <h3 class="text-sm font-bold text-slate-200 uppercase mb-3">💧 Humidity & Wind Trend (Practice)</h3>
                    <div id="plotWHumid" style="height: 320px;"></div>
                </div>
            </div>
            <h3 class="text-sm font-bold text-slate-200 uppercase mt-4">📊 Weather Coefficient Posterior Distributions</h3>
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div class="bg-cardbg rounded-xl p-4 border border-bordercolor">
                    <div id="plotCoef1" style="height: 250px;"></div>
                </div>
                <div class="bg-cardbg rounded-xl p-4 border border-bordercolor">
                    <div id="plotCoef2" style="height: 250px;"></div>
                </div>
                <div class="bg-cardbg rounded-xl p-4 border border-bordercolor">
                    <div id="plotCoef3" style="height: 250px;"></div>
                </div>
                <div class="bg-cardbg rounded-xl p-4 border border-bordercolor">
                    <div id="plotCoef4" style="height: 250px;"></div>
                </div>
            </div>
        </div>

        <!-- TAB 6: DRIVER STYLE -->
        <div id="tab6-content" class="space-y-6 hidden">
            <div class="bg-slate-800/60 border-l-4 border-blue-500 rounded-lg p-4 text-sm text-slate-300">
                <span class="font-bold text-white">Driver Style / Aggression Indicator:</span> A jerk-based measure of how abruptly each driver changes acceleration through a lap, derived from car telemetry.
            </div>
            <div class="bg-cardbg rounded-xl p-4 border border-bordercolor">
                <h3 class="text-sm font-bold text-slate-200 uppercase mb-3">🧭 Driver Style Indices</h3>
                <div id="plotDriverStyle" style="height: 400px;"></div>
            </div>
        </div>

        <!-- TAB 7: DATA EXCLUSION AUDIT -->
        <div id="tab7-content" class="space-y-6 hidden">
            <div class="bg-slate-800/60 border-l-4 border-blue-500 rounded-lg p-4 text-sm text-slate-300">
                <span class="font-bold text-white">Data Exclusion Audit:</span> A transparent accounting of exactly which laps were removed during preprocessing and why.
            </div>
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div class="bg-cardbg rounded-xl p-4 border border-bordercolor overflow-x-auto">
                    <h3 class="text-sm font-bold text-slate-200 uppercase mb-3">Practice Session (Fitting) Exclusions</h3>
                    <table class="w-full text-left text-xs text-slate-300">
                        <thead class="bg-slate-900 text-slate-400"><tr><th class="py-1.5 px-2">Reason</th><th class="py-1.5 px-2">Laps Removed</th></tr></thead>
                        <tbody id="auditPracticeBody" class="divide-y divide-bordercolor"></tbody>
                    </table>
                </div>
                <div class="bg-cardbg rounded-xl p-4 border border-bordercolor overflow-x-auto">
                    <h3 class="text-sm font-bold text-slate-200 uppercase mb-3">Race Session (Prediction) Exclusions</h3>
                    <table class="w-full text-left text-xs text-slate-300">
                        <thead class="bg-slate-900 text-slate-400"><tr><th class="py-1.5 px-2">Reason</th><th class="py-1.5 px-2">Laps Removed</th></tr></thead>
                        <tbody id="auditRaceBody" class="divide-y divide-bordercolor"></tbody>
                    </table>
                </div>
            </div>
        </div>

    </main>

    <!-- JavaScript Application Logic -->
    <script>
        let currentTab = 'tab1';

        function switchTab(tabId) {
            ['tab1', 'tab2', 'tab3', 'tab4', 'tab5', 'tab6', 'tab7'].forEach(t => {
                document.getElementById(t + '-content').classList.add('hidden');
                document.getElementById('btn-' + t).classList.remove('active-tab');
            });
            document.getElementById(tabId + '-content').classList.remove('hidden');
            document.getElementById('btn-' + tabId).classList.add('active-tab');
            currentTab = tabId;
            updateDashboard();
        }

        async function updateDashboard() {
            const driver = document.getElementById('driverSelect').value;
            const pitLoss = parseFloat(document.getElementById('pitLossSlider').value);
            const cpStep = parseInt(document.getElementById('cpStepSlider').value);

            try {
                const res = await fetch(`/api/data?driver=${driver}&pit_loss=${pitLoss}&cp_step=${cpStep}`);
                const data = await res.json();
                
                // Hide loading overlay on success
                const overlay = document.getElementById('loadingOverlay');
                if (overlay) {
                    overlay.style.opacity = '0';
                    setTimeout(() => overlay.style.display = 'none', 500);
                }

                // Update KPIs
            document.getElementById('kpiWear').innerText = `+${data.active_nu.toFixed(3)} s/lap`;
            document.getElementById('kpiWearSub').innerText = `95% CI: [${data.active_nu_q025.toFixed(3)}, ${data.active_nu_q975.toFixed(3)}] • ${data.compound}`;
            document.getElementById('kpiPit').innerText = `Lap ${data.q1.window_start} – ${data.q1.window_end}`;
            document.getElementById('kpiPitSub').innerText = `Crossover Lap: ${data.q1.crossover_lap} • Delta: ${pitLoss.toFixed(1)}s`;
            document.getElementById('kpiTrust').innerText = `${data.q3.trust_index.toFixed(1)}%`;
            document.getElementById('kpiTrustSub').innerText = `${data.q3.trust_status} • ${data.q3.in_90_band_pct.toFixed(1)}% In-Band`;
            document.getElementById('kpiRMSPE').innerText = `${data.overall_rmspe.toFixed(2)} s`;
            document.getElementById('kpiRMSPESub').innerText = `FP2 -> Race Validation • CRPS: ${data.overall_crps.toFixed(2)}s`;

            // Plot 1: Pace & Pit Window
            const traceObs = {
                x: data.stint_laps,
                y: data.stint_times,
                mode: 'markers',
                name: 'Observed Lap',
                marker: { color: '#A0AEC0', size: 6 }
            };
            const traces = [traceObs];
            
            if (data.clean_q025 && data.clean_q025.length > 0) {
                const traceBand = {
                    x: data.stint_laps.concat([...data.stint_laps].reverse()),
                    y: data.clean_q975.concat([...data.clean_q025].reverse()),
                    fill: 'toself',
                    fillcolor: 'rgba(225, 6, 0, 0.15)',
                    line: { color: 'rgba(255,255,255,0)' },
                    name: 'Latent Pace α — 95% CI',
                    hoverinfo: 'skip'
                };
                traces.push(traceBand);
            }
            
            const tracePace = {
                x: data.stint_laps,
                y: data.clean_mean || data.latent_pace_line,
                mode: 'lines',
                name: 'Latent Tyre Pace α (Race-Fitted)',
                line: { color: '#E10600', width: 3 }
            };
            traces.push(tracePace);
            
            if (data.proj_laps && data.proj_laps.length > 0) {
                const traceProj = {
                    x: data.proj_laps,
                    y: data.proj_pace,
                    mode: 'lines',
                    name: 'Projected Wear Forecast',
                    line: { color: '#E10600', width: 2, dash: 'dot' }
                };
                traces.push(traceProj);
            }

            const layoutPace = {
                template: 'plotly_dark',
                margin: { l: 40, r: 20, t: 20, b: 40 },
                xaxis: { title: 'Lap Number' },
                yaxis: { title: 'Lap Time (s)' },
                shapes: [{
                    type: 'rect',
                    xref: 'x', yref: 'paper',
                    x0: data.stint_laps[0] + data.q1.window_start - 1,
                    x1: data.stint_laps[0] + data.q1.window_end - 1,
                    y0: 0, y1: 1,
                    fillcolor: 'rgba(49, 130, 206, 0.2)',
                    line: { width: 0 }
                }]
            };
            Plotly.newPlot('plotPace', traces, layoutPace);

            // Plot 2: Crossover
            const traceCum = {
                x: data.q1.laps,
                y: data.q1.cumulative_loss,
                mode: 'lines+markers',
                name: 'Cumulative Loss',
                line: { color: '#DD6B20', width: 2.5 }
            };
            const tracePitLine = {
                x: data.q1.laps,
                y: Array(data.q1.laps.length).fill(pitLoss),
                mode: 'lines',
                name: `Pit Delta (${pitLoss}s)`,
                line: { color: '#E53E3E', width: 2, dash: 'dash' }
            };
            const layoutCross = {
                template: 'plotly_dark',
                margin: { l: 40, r: 20, t: 20, b: 40 },
                xaxis: { title: 'Laps into Stint' },
                yaxis: { title: 'Cumulative Time Lost (s)' }
            };
            Plotly.newPlot('plotCross', [traceCum, tracePitLine], layoutCross);

            // Q1 Strategy Text
            document.getElementById('q1SummaryBox').innerHTML = `
                <div class="font-bold text-slate-100 mb-1">🏁 Strategist Pit Recommendation:</div>
                Driver <b>${driver}</b> is degrading at <b>+${data.active_nu.toFixed(3)} s/lap</b> on <b>${data.compound}</b> tyres. 
                Cumulative pace loss offsets the ${pitLoss}s pit penalty at <b>Lap ${data.q1.crossover_lap}</b>. 
                The optimal pit window is open between <b>Lap ${data.q1.window_start} and Lap ${data.q1.window_end}</b>.
            `;

            // Checkpoints Table
            let tbodyHtml = '';
            data.checkpoints.forEach(row => {
                tbodyHtml += `
                    <tr class="hover:bg-slate-800/40">
                        <td class="py-2 px-3 font-bold text-slate-200">${row['Checkpoint Lap']}</td>
                        <td class="py-2 px-3">${row['Observed Lap (s)']}</td>
                        <td class="py-2 px-3 text-red-400 font-semibold">${row['Latent Tyre Pace α (s)']}</td>
                        <td class="py-2 px-3 text-amber-400">${row['Tyre Wear Delta Δt (s)']}</td>
                        <td class="py-2 px-3 text-blue-400">${row['Fuel Effect (s)']}</td>
                        <td class="py-2 px-3 text-orange-400">${row['Traffic Penalty (s)']}</td>
                        <td class="py-2 px-3 text-green-400 font-bold">${row['Remaining Tyre Life']}</td>
                        <td class="py-2 px-3">${row['Strategy Action']}</td>
                    </tr>
                `;
            });
            document.getElementById('checkpointTbody').innerHTML = tbodyHtml;

            // Plot Components Bar
            const traceCompWear = {
                x: data.checkpoints.map(c => c['Checkpoint Lap']),
                y: data.checkpoints.map(c => parseFloat(c['Tyre Wear Delta Δt (s)'].replace('+', ''))),
                type: 'bar', name: 'Tyre Wear (s)', marker: { color: '#E10600' }
            };
            const traceCompFuel = {
                x: data.checkpoints.map(c => c['Checkpoint Lap']),
                y: data.checkpoints.map(c => parseFloat(c['Fuel Effect (s)'].replace('+', ''))),
                type: 'bar', name: 'Fuel Penalty (s)', marker: { color: '#3182CE' }
            };
            Plotly.newPlot('plotComponents', [traceCompWear, traceCompFuel], {
                barmode: 'group', template: 'plotly_dark', margin: { l: 40, r: 20, t: 20, b: 40 }
            });

            // Plot Remaining Life
            const traceLife = {
                x: data.checkpoints.map(c => c['Checkpoint Lap']),
                y: data.checkpoints.map(c => c['Remaining Life Value']),
                mode: 'lines+markers', name: 'Tyre Life %',
                line: { color: '#38A169', width: 3 }, fill: 'tozeroy'
            };
            Plotly.newPlot('plotLife', [traceLife], {
                template: 'plotly_dark', margin: { l: 40, r: 20, t: 20, b: 40 },
                yaxis: { range: [0, 105], title: 'Remaining Life %' }
            });

            // Tab 3: Post-Race Validation
            const traceValPred = {
                x: data.race_driver_laps,
                y: data.val_pred_mean,
                mode: 'lines', name: 'Practice-Predicted Race Pace',
                line: { color: '#3182CE', width: 2.5 }
            };
            const traceValObs = {
                x: data.race_driver_laps,
                y: data.race_driver_times,
                mode: 'markers', name: 'Actual Race Laps',
                marker: { color: '#FFFFFF', size: 5 }
            };
            const traceValAlpha = {
                x: data.race_driver_laps,
                y: data.val_clean_alpha,
                mode: 'lines', name: 'Clean Latent Tyre Pace α',
                line: { color: '#E10600', width: 1.5, dash: 'dash' }
            };
            Plotly.newPlot('plotVal', [traceValPred, traceValObs, traceValAlpha], {
                template: 'plotly_dark', margin: { l: 40, r: 20, t: 20, b: 40 },
                xaxis: { title: 'Race Lap Number' }, yaxis: { title: 'Lap Time (s)' }
            });

            // Trust Justification
            document.getElementById('trustJustText').innerText = data.q3.justification;

            // Benchmarks Table
            let benchHtml = '';
            data.benchmarks.forEach(b => {
                benchHtml += `
                    <tr>
                        <td class="py-1.5 px-2 font-medium">${b['Model']}</td>
                        <td class="py-1.5 px-2 text-amber-400 font-bold">${b['RMSPE (s)']}</td>
                        <td class="py-1.5 px-2 text-blue-400">${b['CRPS (s)']}</td>
                    </tr>
                `;
            });
            document.getElementById('benchmarkTbody').innerHTML = benchHtml;

            // Tab 4: Compounds & Track Evolution
            const traceHard = { x: data.samples_hard, type: 'histogram', name: 'Hard (C1/C2)', marker: { color: '#A0AEC0' }, opacity: 0.6 };
            const traceMed = { x: data.samples_med, type: 'histogram', name: 'Medium (C3)', marker: { color: '#ECC94B' }, opacity: 0.6 };
            const traceSoft = { x: data.samples_soft, type: 'histogram', name: 'Soft (C4/C5)', marker: { color: '#E53E3E' }, opacity: 0.6 };
            Plotly.newPlot('plotCompounds', [traceHard, traceMed, traceSoft], {
                barmode: 'overlay', template: 'plotly_dark', margin: { l: 40, r: 20, t: 20, b: 40 },
                xaxis: { title: 'Degradation Rate ν (s/lap)' }
            });

            const traceTrack = {
                x: Array.from({length: 60}, (_, i) => i + 1),
                y: Array.from({length: 60}, (_, i) => (data.rho_mean * (i + 1) / 60.0)),
                mode: 'lines', name: 'Track Grip Buildup',
                line: { color: '#38A169', width: 3 }
            };
            Plotly.newPlot('plotTrack', [traceTrack], {
                template: 'plotly_dark', margin: { l: 40, r: 20, t: 20, b: 40 },
                xaxis: { title: 'Session Minutes' }, yaxis: { title: 'Track Delta (s)' }
            });

            // Tab 5: Weather Effects
            const traceWTemp1 = { x: data.practice_laps, y: data.practice_track_temp, mode: 'lines+markers', name: 'Track Temp', line: { color: '#E10600' } };
            const traceWTemp2 = { x: data.practice_laps, y: data.practice_air_temp, mode: 'lines+markers', name: 'Air Temp', line: { color: '#3182CE' } };
            Plotly.newPlot('plotWTemp', [traceWTemp1, traceWTemp2], { template: 'plotly_dark', margin: { l: 40, r: 20, t: 20, b: 40 } });

            const traceWHum1 = { x: data.practice_laps, y: data.practice_humidity, mode: 'lines+markers', name: 'Humidity', line: { color: '#38A169' } };
            const traceWHum2 = { x: data.practice_laps, y: data.practice_wind_speed, mode: 'lines+markers', name: 'Wind', line: { color: '#ECC94B' } };
            Plotly.newPlot('plotWHumid', [traceWHum1, traceWHum2], { template: 'plotly_dark', margin: { l: 40, r: 20, t: 20, b: 40 } });

            const coefMap = [
                { id: 'plotCoef1', samples: data.kappa_temp, name: 'Track Temp Mod (κ)', color: '#E10600' },
                { id: 'plotCoef2', samples: data.beta_air, name: 'Air Temp Mod (β_air)', color: '#3182CE' },
                { id: 'plotCoef3', samples: data.beta_humid, name: 'Humidity Mod (β_humid)', color: '#38A169' },
                { id: 'plotCoef4', samples: data.beta_wind, name: 'Wind Mod (β_wind)', color: '#ECC94B' }
            ];
            coefMap.forEach(c => {
                if(c.samples && c.samples.length > 0) {
                    const traceC = { x: c.samples, type: 'histogram', marker: { color: c.color }, name: c.name };
                    Plotly.newPlot(c.id, [traceC], { template: 'plotly_dark', margin: { l: 30, r: 20, t: 40, b: 30 }, title: c.name });
                } else {
                    document.getElementById(c.id).innerHTML = `<div class='text-slate-400 p-4'>${c.name} data not available.</div>`;
                }
            });

            // Tab 6: Driver Style
            if(data.style_profile && data.style_profile.length > 0) {
                const traceDSI1 = { x: data.style_profile.map(d=>d.Driver), y: data.style_profile.map(d=>d.Longitudinal_DSI), type: 'bar', name: 'Long DSI', marker: { color: '#3182CE' } };
                const traceDSI2 = { x: data.style_profile.map(d=>d.Driver), y: data.style_profile.map(d=>d.Lateral_DSI), type: 'bar', name: 'Lat DSI', marker: { color: '#38A169' } };
                const traceDSI3 = { x: data.style_profile.map(d=>d.Driver), y: data.style_profile.map(d=>d.DSI_combined), mode: 'lines+markers', name: 'Combined', marker: { color: '#E10600', size: 10 }, line: { color: '#E10600', dash: 'dot', width: 2 } };
                Plotly.newPlot('plotDriverStyle', [traceDSI1, traceDSI2, traceDSI3], { barmode: 'group', template: 'plotly_dark', margin: { l: 40, r: 20, t: 20, b: 40 } });
            } else {
                document.getElementById('plotDriverStyle').innerHTML = "<div class='text-slate-400 p-4'>Driver style data not available for this session.</div>";
            }

            // Tab 7: Audit
            const renderAudit = (auditData, elementId) => {
                let html = '';
                if(auditData && auditData.length > 0) {
                    auditData.forEach(row => {
                        html += `<tr><td class="py-1.5 px-2">${row.Reason}</td><td class="py-1.5 px-2 font-bold text-f1red">${row.Laps_Removed}</td></tr>`;
                    });
                } else {
                    html = '<tr><td colspan="2" class="py-1.5 px-2 text-slate-400">No data.</td></tr>';
                }
                document.getElementById(elementId).innerHTML = html;
            };
            renderAudit(data.exclusion_practice, 'auditPracticeBody');
            renderAudit(data.exclusion_race, 'auditRaceBody');
        } catch (err) {
                console.error("Failed to load dashboard data:", err);
                const overlay = document.getElementById('loadingOverlay');
                if (overlay) {
                    overlay.innerHTML = `<h2 class="text-xl font-bold text-f1red mb-2">❌ Error loading data</h2><p class="text-slate-400 max-w-md text-center">${err.message}</p>`;
                    overlay.style.opacity = '1';
                    overlay.style.display = 'flex';
                }
            }
        }

        window.onload = updateDashboard;
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/data')
def get_api_data():
    dash_data = get_data()
    r_dataset = dash_data['r_dataset']
    val_results = dash_data['val_results']
    fit_p3 = dash_data['fit_p3']
    benchmark_df = dash_data['benchmark_df']
    df_race = r_dataset['dataframe']
    driver_names = r_dataset['driver_names']
    
    p_dataset = dash_data['p_dataset']
    df_practice = p_dataset['dataframe']
    style_profile = dash_data.get('style_profile')
    exclusion_practice = dash_data.get('exclusion_summary_practice')
    exclusion_race = dash_data.get('exclusion_summary_race')

    driver = request.args.get('driver', 'HAM')
    pit_loss = float(request.args.get('pit_loss', 22.0))
    cp_step = int(request.args.get('cp_step', 5))
    
    driver_race_df = df_race[df_race['Driver'] == driver].sort_values(by='LapNumber').reset_index(drop=True)
    stint_df = driver_race_df[driver_race_df['Stint'] == 1].copy()
    compound = stint_df['Compound'].iloc[0] if len(stint_df) > 0 else 'MEDIUM'
    
    driver_idx = driver_names.index(driver)
    compound_to_id = {'HARD': 0, 'MEDIUM': 1, 'SOFT': 2}
    comp_id = compound_to_id.get(compound, 1)
    
    nu_samples = fit_p3['samples']['nu_dc'][:, driver_idx, comp_id]
    active_nu = float(np.mean(nu_samples))
    active_nu_q025 = float(np.percentile(nu_samples, 2.5))
    active_nu_q975 = float(np.percentile(nu_samples, 97.5))
    
    fresh_pace_est = float(np.median(stint_df['LapTime_s']) - 0.033 * np.median(stint_df['fuel_kg']))
    
    q1 = dashboard_data.compute_question1_pit_analytics(
        driver=driver,
        compound=compound,
        stint_length=max(len(stint_df), 25),
        degradation_rate=active_nu,
        fresh_tyre_pace=fresh_pace_est,
        pit_loss_seconds=pit_loss
    )
    
    q2_df = dashboard_data.compute_question2_checkpoints(stint_df, active_nu, fresh_pace_est, cp_step)
    q3 = dashboard_data.compute_question3_trust_metrics(val_results, r_dataset)
    
    driver_mask = (df_race['Driver'] == driver)
    indices = np.where(driver_mask)[0]
    
    fit_hier_race = dash_data.get('fit_hier_race')
    
    if fit_hier_race is not None and 'alpha' in fit_hier_race['samples']:
        stint_indices = np.where(driver_mask & (df_race['Stint'] == 1))[0]
        alpha_samples = fit_hier_race['samples']['alpha'][:, stint_indices]
        clean_mean = np.mean(alpha_samples, axis=0).tolist()
        clean_q025 = np.percentile(alpha_samples, 2.5, axis=0).tolist()
        clean_q975 = np.percentile(alpha_samples, 97.5, axis=0).tolist()
        
        race_nu_samples = fit_hier_race['samples']['nu_dc'][:, driver_idx, comp_id]
        race_nu_mean = float(np.mean(race_nu_samples))
        
        if len(stint_df) > 0:
            last_stint_lap = int(stint_df['LapNumber'].max())
            proj_laps = list(range(last_stint_lap, last_stint_lap + 15))
            proj_pace = [clean_mean[-1] + (lap - proj_laps[0] + 1) * race_nu_mean for lap in proj_laps]
        else:
            proj_laps = []
            proj_pace = []
    else:
        clean_mean = (fresh_pace_est + np.arange(len(stint_df)) * active_nu).tolist()
        clean_q025 = []
        clean_q975 = []
        proj_laps = []
        proj_pace = []

    return jsonify({
        'driver': driver,
        'compound': compound,
        'active_nu': active_nu,
        'active_nu_q025': active_nu_q025,
        'active_nu_q975': active_nu_q975,
        'stint_laps': stint_df['LapNumber'].tolist(),
        'stint_times': stint_df['LapTime_s'].tolist(),
        'clean_mean': clean_mean,
        'clean_q025': clean_q025,
        'clean_q975': clean_q975,
        'proj_laps': proj_laps,
        'proj_pace': proj_pace,
        'latent_pace_line': (fresh_pace_est + np.arange(len(stint_df)) * active_nu).tolist(),
        'q1': {
            'laps': q1['laps'].tolist(),
            'cumulative_loss': q1['cumulative_loss'].tolist(),
            'crossover_lap': q1['crossover_lap'],
            'window_start': q1['window_start'],
            'window_end': q1['window_end']
        },
        'checkpoints': q2_df.to_dict(orient='records'),
        'q3': {
            'trust_index': q3['trust_index'],
            'trust_status': q3['trust_status'],
            'in_90_band_pct': q3['in_90_band_pct'],
            'justification': q3['justification']
        },
        'overall_rmspe': val_results['overall_rmspe'],
        'overall_crps': val_results['overall_crps'],
        'benchmarks': benchmark_df.to_dict(orient='records'),
        'race_driver_laps': df_race[driver_mask]['LapNumber'].tolist(),
        'race_driver_times': r_dataset['lap_time'][indices].tolist(),
        'val_pred_mean': val_results['pred_mean'][indices].tolist(),
        'val_clean_alpha': val_results['clean_alpha_mean'][indices].tolist(),
        'samples_hard': fit_p3['samples']['mu_nu_hard'].tolist(),
        'samples_med': fit_p3['samples']['mu_nu_med'].tolist(),
        'samples_soft': fit_p3['samples']['mu_nu_soft'].tolist(),
        'rho_mean': float(np.mean(fit_p3['samples']['rho'])),
        'practice_laps': df_practice['LapNumber'].tolist(),
        'practice_track_temp': df_practice['track_temp'].tolist(),
        'practice_air_temp': df_practice['air_temp'].tolist(),
        'practice_humidity': df_practice['humidity'].tolist(),
        'practice_wind_speed': df_practice['wind_speed'].tolist(),
        'kappa_temp': fit_p3['samples'].get('kappa_temp', np.array([])).tolist(),
        'beta_air': fit_p3['samples'].get('beta_air', np.array([])).tolist(),
        'beta_humid': fit_p3['samples'].get('beta_humid', np.array([])).tolist(),
        'beta_wind': fit_p3['samples'].get('beta_wind', np.array([])).tolist(),
        'style_profile': style_profile.to_dict(orient='records') if style_profile is not None else [],
        'exclusion_practice': exclusion_practice.to_dict(orient='records') if exclusion_practice is not None else [],
        'exclusion_race': exclusion_race.to_dict(orient='records') if exclusion_race is not None else []
    })

if __name__ == '__main__':
    print("Starting Tyre Degradation Intelligence Flask App...")
    print("The first request will take ~1-2 minutes to compile and run JAX/NumPyro models.")
    # Must use threaded=False to prevent JAX/NumPyro Thread-Local State assertion errors!
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
