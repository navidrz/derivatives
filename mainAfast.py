# main.py

import logging
import pandas as pd
from pathlib import Path
import sys
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json
import plotly.express as px
import plotly.graph_objects as go
import multiprocessing

from DataLoaderV import APIDataLoader
from OptionAnalyzerA import OptionAnalyzer

def setup_logger():
    logger = logging.getLogger("MainLogger")
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler('main_analysis.log', encoding='utf-8')
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    return logger

logger = setup_logger()

def export_dataframe(df: pd.DataFrame, excel_path: str, csv_path: str, sheet_name: str = 'Sheet1'):
    try:
        df.to_excel(excel_path, index=False, sheet_name=sheet_name)
        logger.info(f"Exported to Excel at '{excel_path}'.")
    except Exception as e:
        logger.error(f"Excel export failed: {e}")
    try:
        df.to_csv(csv_path, index=False)
        logger.info(f"Exported to CSV at '{csv_path}'.")
    except Exception as e:
        logger.error(f"CSV export failed: {e}")

def create_dashboard_html(metrics_df, recommendations_df, scenario_df, cluster_assignments, dr_coords_df, cleaned_data, output_dir: Path):
    # Merge all data
    merged_df = pd.merge(metrics_df, recommendations_df, on='OptionName', how='left')
    full_df = pd.merge(scenario_df, merged_df, on='OptionName', how='left')
    full_df = pd.merge(full_df, cluster_assignments, on='OptionName', how='left')
    full_df = pd.merge(full_df, dr_coords_df, on='OptionName', how='left')

    recommendation_categories = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']
    if 'Recommendation' in full_df.columns:
        full_df['Recommendation'] = pd.Categorical(full_df['Recommendation'], categories=recommendation_categories, ordered=True)

    # Dimensionality Reduction Scatter
    dr_data = full_df.dropna(subset=['DRX','DRY'])
    if not dr_data.empty:
        fig_dr = px.scatter(
            dr_data,
            x='DRX', y='DRY',
            color='MarketScenario',
            symbol='UserMarketView',
            hover_data=['OptionName','PoP_Long','Sharpe_Long','CVaR_Long','Payout_Long'],
            title='Dimensionality Reduction Scatter: Metrics in 2D'
        )
        fig_dr_json = fig_dr.to_json()
    else:
        fig_dr_json = "{}"

    # Radar Chart
    radar_metrics = ['Sharpe_Long','PoP_Long','VaR_Long','CVaR_Long','Payout_Long']
    radar_agg = full_df.groupby(['MarketScenario','UserMarketView'], as_index=False)[radar_metrics].mean().dropna()
    fig_radar = go.Figure()
    if not radar_agg.empty:
        default_sc = radar_agg['MarketScenario'].iloc[0]
        default_uv = radar_agg['UserMarketView'].iloc[0]
        sub = radar_agg[(radar_agg['MarketScenario']==default_sc)&(radar_agg['UserMarketView']==default_uv)]
        if not sub.empty:
            vals = sub[radar_metrics].values.flatten()
            fig_radar.add_trace(go.Scatterpolar(r=vals, theta=radar_metrics, fill='toself'))
    fig_radar.update_layout(title="Radar Chart: Avg Metrics (Dynamic)", polar=dict(radialaxis=dict(visible=True)))
    fig_radar_json = fig_radar.to_json()

    # Bubble Chart Over Time (Days)
    scenario_days = full_df.copy()
    if 'days' not in scenario_days.columns and 'days' in cleaned_data.columns:
        scenario_days = pd.merge(scenario_days, cleaned_data[['option_name','days']], left_on='OptionName', right_on='option_name', how='left')
    if 'days' in scenario_days.columns and not scenario_days['days'].isna().all():
        sub_bubble = scenario_days.dropna(subset=['days','Sharpe_Long','PoP_Long'])
        if not sub_bubble.empty:
            fig_bubble = px.scatter(
                sub_bubble, x='Sharpe_Long', y='PoP_Long',
                size='Payout_Long',
                color='MarketScenario',
                animation_frame='days',
                hover_data=['OptionName','UserMarketView','CVaR_Long'],
                title='Animated Bubble Chart of Sharpe vs PoP by Days'
            )
            fig_bubble_json = fig_bubble.to_json()
        else:
            fig_bubble_json = "{}"
    else:
        fig_bubble_json = "{}"

    # Sankey Diagram
    if 'InferredMarketSentiment' in full_df.columns and 'Recommendation' in full_df.columns:
        sankey_data = full_df.groupby(['MarketScenario','UserMarketView','InferredMarketSentiment','Recommendation'], as_index=False)['OptionName'].count()
        if not sankey_data.empty:
            scenarios_list = sankey_data['MarketScenario'].unique().tolist()
            uviews_list = sankey_data['UserMarketView'].unique().tolist()
            sentiments = sankey_data['InferredMarketSentiment'].unique().tolist()
            recs = recommendation_categories

            nodes = scenarios_list + uviews_list + sentiments + recs
            node_indices = {node:i for i,node in enumerate(nodes)}
            source, target, value = [],[],[]

            for sc in scenarios_list:
                for uv in uviews_list:
                    flow = sankey_data[(sankey_data['MarketScenario']==sc)&(sankey_data['UserMarketView']==uv)]['OptionName'].sum()
                    if flow>0:
                        source.append(node_indices[sc])
                        target.append(node_indices[uv])
                        value.append(flow)
            for uv in uviews_list:
                for sent in sentiments:
                    flow = sankey_data[(sankey_data['UserMarketView']==uv)&(sankey_data['InferredMarketSentiment']==sent)]['OptionName'].sum()
                    if flow>0:
                        source.append(node_indices[uv])
                        target.append(node_indices[sent])
                        value.append(flow)
            for sent in sentiments:
                for r in recs:
                    flow = sankey_data[(sankey_data['InferredMarketSentiment']==sent)&(sankey_data['Recommendation']==r)]['OptionName'].sum()
                    if flow>0:
                        source.append(node_indices[sent])
                        target.append(node_indices[r])
                        value.append(flow)

            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(label=nodes),
                link=dict(source=source,target=target,value=value)
            )])
            fig_sankey.update_layout(title="Sankey: Scenario->UserView->Sentiment->Recommendation")
            fig_sankey_json = fig_sankey.to_json()
        else:
            fig_sankey_json = "{}"
    else:
        fig_sankey_json = "{}"

    # Parallel Coordinates
    pc_cols = ['PoP_Long','VaR_Long','CVaR_Long','Sharpe_Long','Payout_Long']
    pc_data = full_df.dropna(subset=pc_cols)
    if not pc_data.empty:
        fig_pc = px.parallel_coordinates(
            pc_data,
            dimensions=pc_cols,
            color='PoP_Long',
            color_continuous_scale=px.colors.sequential.Viridis,
            title='Parallel Coordinates: Metrics with scenario/view embedded'
        )
        fig_pc_json = fig_pc.to_json()
    else:
        fig_pc_json = "{}"

    scenarios = sorted(full_df['MarketScenario'].dropna().unique().tolist())
    uviews = sorted(full_df['UserMarketView'].dropna().unique().tolist())

    scenarios_js = json.dumps(scenarios, ensure_ascii=False)
    uviews_js = json.dumps(uviews, ensure_ascii=False)

    html_content = f'''<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>آپشن ایران بورس</title>
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.9.1/font/bootstrap-icons.css">
<link rel="manifest" href="manifest.json">

<style>
body {{
    font-family: sans-serif;
}}
.sidebar {{
    width: 250px;
}}
.main-content {{
    flex: 1;
}}
</style>

<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<!-- DataTables -->
<link rel="stylesheet" href="https://cdn.datatables.net/1.11.3/css/jquery.dataTables.min.css">
<script src="https://cdn.datatables.net/1.11.3/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<script>
if ('serviceWorker' in navigator) {{
    navigator.serviceWorker.register('service-worker.js');
}}
</script>
</head>
<body class="bg-gray-100 text-gray-900">

<nav class="bg-white shadow p-4 flex items-center justify-between">
    <h1 class="text-2xl font-bold">آپشن ایران بورس</h1>
    <div class="flex space-x-4 items-center">
      <div>
        <label for="scenarioSelect" class="mr-2">سناریو:</label>
        <select id="scenarioSelect" class="border p-1 rounded">
        </select>
      </div>
      <div>
        <label for="userViewSelect" class="mr-2">دیدگاه بازار:</label>
        <select id="userViewSelect" class="border p-1 rounded">
        </select>
      </div>
    </div>
</nav>

<div class="flex">
    <aside class="sidebar bg-white shadow h-screen p-4 overflow-auto">
        <h2 class="font-bold mb-4">داده‌ها و فیلترها</h2>
        <p>جداول داده‌های ترکیبی:</p>
        <ul class="list-disc list-inside mb-4">
            <li><a href="#metricsTableSection">جدول متریک‌ها</a></li>
            <li><a href="#recommendationsTableSection">جدول پیشنهادات</a></li>
            <li><a href="#scenarioTableSection">جدول سناریو</a></li>
        </ul>
        <p>نمودارها:</p>
        <ul class="list-disc list-inside">
            <li><a href="#drSection">نمودار کاهش بُعد</a></li>
            <li><a href="#radarSection">رادار متریک‌ها</a></li>
            <li><a href="#bubbleSection">نمودار حبابی متحرک</a></li>
            <li><a href="#sankeySection">سانکی سناریو</a></li>
            <li><a href="#parallelSection">مختصات موازی</a></li>
        </ul>
    </aside>

    <main class="main-content p-4 overflow-auto">
        <section id="metricsTableSection" class="mb-8">
            <h2 class="text-xl font-bold mb-2">جدول متریک‌ها</h2>
            <table id="metricsTable" class="display w-full"></table>
        </section>

        <section id="recommendationsTableSection" class="mb-8">
            <h2 class="text-xl font-bold mb-2">جدول پیشنهادات</h2>
            <table id="recommendationsTable" class="display w-full"></table>
        </section>

        <section id="scenarioTableSection" class="mb-8">
            <h2 class="text-xl font-bold mb-2">جدول سناریو</h2>
            <table id="scenarioTable" class="display w-full"></table>
        </section>

        <section id="drSection" class="mb-8">
            <h2 class="text-xl font-bold mb-2">نمودار کاهش بُعد</h2>
            <div id="drPlot" style="width:100%;height:600px;"></div>
        </section>

        <section id="radarSection" class="mb-8">
            <h2 class="text-xl font-bold mb-2">رادار متریک‌ها</h2>
            <div id="radarPlot" style="width:100%;height:600px;"></div>
        </section>

        <section id="bubbleSection" class="mb-8">
            <h2 class="text-xl font-bold mb-2">نمودار حبابی متحرک</h2>
            <div id="bubblePlot" style="width:100%;height:600px;"></div>
        </section>

        <section id="sankeySection" class="mb-8">
            <h2 class="text-xl font-bold mb-2">سانکی سناریو</h2>
            <div id="sankeyPlot" style="width:100%;height:600px;"></div>
        </section>

        <section id="parallelSection" class="mb-8">
            <h2 class="text-xl font-bold mb-2">مختصات موازی متریک‌ها</h2>
            <div id="pcPlot" style="width:100%;height:600px;"></div>
        </section>
    </main>
</div>

<script>
var scenarios = {scenarios_js};
var userViews = {uviews_js};

var scenarioSelect = document.getElementById('scenarioSelect');
var userViewSelect = document.getElementById('userViewSelect');

scenarios.forEach(sc => {{
    var opt = document.createElement('option');
    opt.value = sc;
    opt.textContent = sc;
    scenarioSelect.appendChild(opt);
}});
userViews.forEach(uv => {{
    var opt = document.createElement('option');
    opt.value = uv;
    opt.textContent = uv;
    userViewSelect.appendChild(opt);
}});

function loadTable(tableId, csvPath) {{
    $.ajax({{
        url: csvPath,
        dataType: 'text'
    }}).done(function(data) {{
        var lines = data.trim().split('\\n');
        var headers = lines[0].split(',');
        var rows = lines.slice(1).map(line => line.split(','));
        var table = $('#' + tableId).DataTable({{
            data: rows,
            columns: headers.map(h => {{return {{title: h}}}}),
            responsive: true
        }});

        $('#scenarioSelect').on('change', function() {{
            var val = this.value ? '^'+this.value+'$' : '';
            var scColIndex = headers.indexOf('MarketScenario');
            if (scColIndex >= 0) {{
                table.column(scColIndex).search(val, true, false).draw();
            }}
        }});

        $('#userViewSelect').on('change', function() {{
            var val = this.value ? '^'+this.value+'$' : '';
            var uvColIndex = headers.indexOf('UserMarketView');
            if (uvColIndex >= 0) {{
                table.column(uvColIndex).search(val, true, false).draw();
            }}
        }});
    }});
}}

loadTable('metricsTable', 'combined_metrics_data.csv');
loadTable('recommendationsTable', 'combined_recommendations.csv');
loadTable('scenarioTable', 'combined_scenario_analysis.csv');

var figDR = {fig_dr_json};
if(figDR.data) {{
    Plotly.newPlot('drPlot', figDR.data, figDR.layout);
}}

var figRadar = {fig_radar_json};
if(figRadar.data) {{
    Plotly.newPlot('radarPlot', figRadar.data, figRadar.layout);
}}

var figBubble = {fig_bubble_json};
if(figBubble.data) {{
    Plotly.newPlot('bubblePlot', figBubble.data, figBubble.layout);
}}

var figSankey = {fig_sankey_json};
if(figSankey.data) {{
    Plotly.newPlot('sankeyPlot', figSankey.data, figSankey.layout);
}}

var figPC = {fig_pc_json};
if(figPC.data) {{
    Plotly.newPlot('pcPlot', figPC.data, figPC.layout);
}}
</script>
</body>
</html>
'''

    with open(output_dir / 'index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    manifest_content = '''{
  "name": "آپشن ایران بورس",
  "short_name": "OptionIRB",
  "start_url": "index.html",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#ffffff",
  "icons": []
}'''
    with open(output_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        f.write(manifest_content)

    sw_content = '''self.addEventListener('install', function(e) {
  e.waitUntil(
    caches.open('option-iran-bourse-v1').then(function(cache) {
      return cache.addAll([
        'index.html',
        'manifest.json',
        'combined_metrics_data.csv',
        'combined_recommendations.csv',
        'combined_scenario_analysis.csv'
      ]);
    })
  );
});

self.addEventListener('fetch', function(e) {
  e.respondWith(
    caches.match(e.request).then(function(response) {
      return response || fetch(e.request);
    })
  );
});
'''
    with open(output_dir / 'service-worker.js', 'w', encoding='utf-8') as f:
        f.write(sw_content)

def run_scenario_view(args):
    m_scenario, u_view, cleaned_data, historical_data, scenarios_for_analysis = args
    logger.info(f"Processing MarketScenario={m_scenario}, UserMarketView={u_view}")
    option_analyzer = OptionAnalyzer(
        cleaned_data=cleaned_data.copy(),
        historical_data=historical_data,
        risk_free_rate=0.31,
        market_scenario=m_scenario
    )

    num_simulations = 100000
    option_analyzer.monte_carlo_simulation(num_simulations=num_simulations)
    option_analyzer.calculate_pop()
    option_analyzer.calculate_breakeven()
    option_analyzer.calculate_sharpe_ratio()
    option_analyzer.calculate_var_cvar()
    option_analyzer.calculate_payout_ratio()
    option_analyzer.get_recommendations(user_market_view=u_view)

    metrics_df = option_analyzer.compile_metrics_data()
    metrics_df['MarketScenario'] = m_scenario
    metrics_df['UserMarketView'] = u_view

    recommendations_df = option_analyzer.recommendations_df
    recommendations_df['MarketScenario'] = m_scenario
    recommendations_df['UserMarketView'] = u_view

    option_analyzer.perform_scenario_analysis(scenarios=scenarios_for_analysis, num_simulations=15000)
    scenario_metrics = option_analyzer.scenario_analysis_results
    scenario_data = []
    for scenario, options_data in scenario_metrics.items():
        for option, metrics in options_data.items():
            row = {'Scenario': scenario, 'OptionName': option}
            row.update(metrics)
            scenario_data.append(row)
    scenario_df = pd.DataFrame(scenario_data)
    scenario_df['MarketScenario'] = m_scenario
    scenario_df['UserMarketView'] = u_view

    return metrics_df, recommendations_df, scenario_df

def main():
    logger.info("Starting multi-scenario, multi-user-view Option Analysis Workflow (no nested multiprocessing).")

    csv_file_path = Path('C:\\Users\\Administrator\\Downloads\\Trader extention\\TSETMC_sample_data.csv')
    if not csv_file_path.exists():
        logger.error(f"CSV file not found at path: {csv_file_path}")
        return

    market_scenarios = ['Normal Market','Bull Market','Bear Market','Volatile Market','Market Crash','Market Rally']
    user_market_views = ['bullish','bearish','neutral']

    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Initializing APIDataLoader.")
    api_data_loader = APIDataLoader(file_path=csv_file_path)
    api_data_loader.fetch_and_clean_data()
    cleaned_data = api_data_loader.get_cleaned_data()
    historical_data = api_data_loader.get_historical_data()

    scenarios_for_analysis = {
        '2008 Crisis': {'sigma_v_factor': 2.0},
        'High Inflation': {'r_shift': 0.02}
    }

    tasks = []
    for m_scenario in market_scenarios:
        for u_view in user_market_views:
            tasks.append((m_scenario, u_view, cleaned_data, historical_data, scenarios_for_analysis))

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(run_scenario_view, tasks)

    all_metrics = []
    all_recommendations = []
    all_scenario_results = []
    for (metrics_df, recommendations_df, scenario_df) in results:
        all_metrics.append(metrics_df)
        all_recommendations.append(recommendations_df)
        all_scenario_results.append(scenario_df)

    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_recommendations = pd.concat(all_recommendations, ignore_index=True)
    combined_scenario_results = pd.concat(all_scenario_results, ignore_index=True)

    # After computing combined_metrics and numeric_cols:
    numeric_cols = ['Sharpe_Long','PoP_Long','VaR_Long','CVaR_Long','Payout_Long','Breakeven_Long_Pct']
    cluster_data = combined_metrics.dropna(subset=numeric_cols).copy()  # Added .copy() here
    if not cluster_data.empty:
        X = cluster_data[numeric_cols].values
        km = KMeans(n_clusters=5, random_state=42).fit(X)
        cluster_data.loc[:, 'Cluster'] = km.labels_  # Use loc explicitly
        pca = PCA(n_components=2, random_state=42)
        dr_coords = pca.fit_transform(X)
        cluster_data.loc[:, 'DRX'] = dr_coords[:,0]
        cluster_data.loc[:, 'DRY'] = dr_coords[:,1]

        cluster_assignments = cluster_data[['OptionName','Cluster']]
        dr_coords_df = cluster_data[['OptionName','DRX','DRY']]
    else:
        cluster_assignments = pd.DataFrame(columns=['OptionName','Cluster'])
        dr_coords_df = pd.DataFrame(columns=['OptionName','DRX','DRY'])

    export_dataframe(
        df=combined_metrics,
        excel_path=str(output_dir / 'combined_metrics_data.xlsx'),
        csv_path=str(output_dir / 'combined_metrics_data.csv'),
        sheet_name='Metrics'
    )
    export_dataframe(
        df=combined_recommendations,
        excel_path=str(output_dir / 'combined_recommendations.xlsx'),
        csv_path=str(output_dir / 'combined_recommendations.csv'),
        sheet_name='Recommendations'
    )
    export_dataframe(
        df=combined_scenario_results,
        excel_path=str(output_dir / 'combined_scenario_analysis.xlsx'),
        csv_path=str(output_dir / 'combined_scenario_analysis.csv'),
        sheet_name='ScenarioAnalysis'
    )
    export_dataframe(
        df=cluster_assignments,
        excel_path=str(output_dir / 'cluster_assignments.xlsx'),
        csv_path=str(output_dir / 'cluster_assignments.csv'),
        sheet_name='Clusters'
    )
    export_dataframe(
        df=dr_coords_df,
        excel_path=str(output_dir / 'dr_coords.xlsx'),
        csv_path=str(output_dir / 'dr_coords.csv'),
        sheet_name='DimRed'
    )

    create_dashboard_html(
        metrics_df=combined_metrics,
        recommendations_df=combined_recommendations,
        scenario_df=combined_scenario_results,
        cluster_assignments=cluster_assignments,
        dr_coords_df=dr_coords_df,
        cleaned_data=cleaned_data,
        output_dir=output_dir
    )

    logger.info("Dashboard generation completed successfully with scenario/user-view filters for DataTables and single-level parallelization.")

if __name__ == "__main__":
    main()
