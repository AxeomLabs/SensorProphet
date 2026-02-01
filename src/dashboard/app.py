"""
Predictive Maintenance Dashboard - Advanced Features
Real-time CSV processing with anomaly detection, playback controls, and analytics.
"""
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import base64
import io

from src.config import DASHBOARD_HOST, DASHBOARD_PORT, DASHBOARD_DEBUG
from src.utils import setup_logging
from src.alerts.email_notifier import email_notifier, send_anomaly_email

logger = setup_logging("dashboard")

# Theme
T = {
    "bg": "#0d1117", "card": "#161b22", "header": "#21262d", "border": "#30363d",
    "accent": "#f78166", "text": "#c9d1d9", "muted": "#8b949e", "dim": "#6e7681",
    "success": "#3fb950", "warning": "#d29922", "danger": "#f85149", "info": "#58a6ff",
}

# Global state
STATE = {
    "csv": None, "idx": 0, "paused": False, "speed": 1,
    "history": {"rms": [], "kurtosis": [], "mean": [], "health": [], "zscore": [], "anomaly": []},
    "alerts": [], "threshold": 2.5, "stats": {},
    "email_enabled": False, "email_sender": "", "email_password": "",
    "email_recipient": "snox.kali@usa.com", "last_email_idx": -100,
}


def create_app(data_store: Optional[Dict] = None) -> dash.Dash:
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], 
                    title="Predictive Maintenance", suppress_callback_exceptions=True)
    
    app.index_string = f'''<!DOCTYPE html><html><head>
        {{%metas%}}<title>{{%title%}}</title>{{%favicon%}}{{%css%}}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono&display=swap" rel="stylesheet">
        <style>
            body {{ font-family: 'Inter', sans-serif; background: {T["bg"]}; color: {T["text"]}; margin: 0; font-size: 12px; }}
            .mono {{ font-family: 'JetBrains Mono', monospace; }}
            .card {{ background: {T["card"]}; border: 1px solid {T["border"]}; border-radius: 6px; }}
            .card-header {{ background: {T["header"]}; border-bottom: 1px solid {T["border"]}; padding: 8px 12px; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: {T["muted"]}; }}
            .btn {{ padding: 6px 12px; border: none; border-radius: 4px; cursor: pointer; font-size: 11px; transition: all 0.2s; }}
            .btn-primary {{ background: {T["accent"]}; color: white; }}
            .btn-secondary {{ background: {T["header"]}; color: {T["text"]}; border: 1px solid {T["border"]}; }}
            .btn:hover {{ opacity: 0.8; transform: translateY(-1px); }}
            .pulse {{ animation: pulse 1s infinite; }} @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.4; }} }}
            .slider {{ width: 100%; }}
            input[type=range] {{ -webkit-appearance: none; background: {T["header"]}; height: 6px; border-radius: 3px; }}
            input[type=range]::-webkit-slider-thumb {{ -webkit-appearance: none; width: 14px; height: 14px; background: {T["accent"]}; border-radius: 50%; cursor: pointer; }}
        </style>
    </head><body>{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body></html>'''
    
    app.layout = create_layout()
    register_callbacks(app)
    return app


def create_layout() -> html.Div:
    return html.Div([
        # Header
        html.Div([
            html.Div([
                html.Span("⚙ PREDICTIVE MAINTENANCE", style={"fontWeight": "600", "letterSpacing": "1px"}),
                html.Span(" v2.0", style={"color": T["dim"], "fontSize": "10px", "marginLeft": "8px"}),
            ]),
            html.Div([
                html.Span("●", id="status-dot", style={"color": T["muted"], "marginRight": "6px"}),
                html.Span(id="system-status", children="IDLE"),
                html.Span(" | ", style={"color": T["border"], "margin": "0 10px"}),
                html.Span(id="time-display", style={"color": T["dim"]}),
            ], style={"fontSize": "10px"}),
        ], style={"display": "flex", "justifyContent": "space-between", "padding": "10px 20px", 
                  "background": T["card"], "borderBottom": f"1px solid {T['border']}"}),
        
        dcc.Interval(id="tick", interval=500, n_intervals=0),
        
        # Main
        html.Div([
            # Left Sidebar
            html.Div([
                # Upload
                html.Div([
                    html.Div("DATA SOURCE", className="card-header"),
                    html.Div([
                        dcc.Upload(id="csv-upload", children=html.Div([
                            "📁 ", html.Span("Browse", style={"color": T["accent"]})
                        ]), style={"border": f"1px dashed {T['border']}", "padding": "8px", "textAlign": "center", 
                                   "background": T["header"], "borderRadius": "4px", "cursor": "pointer"}),
                        html.Div(id="file-info", style={"marginTop": "6px", "fontSize": "10px", "color": T["dim"]}),
                    ], style={"padding": "10px"}),
                ], className="card", style={"marginBottom": "10px"}),
                
                # Playback Controls
                html.Div([
                    html.Div("PLAYBACK CONTROLS", className="card-header"),
                    html.Div([
                        html.Div([
                            html.Button("⏮", id="btn-reset", className="btn btn-secondary", style={"marginRight": "4px"}),
                            html.Button("⏸", id="btn-pause", className="btn btn-primary", style={"marginRight": "4px"}),
                            html.Button("⏭", id="btn-skip", className="btn btn-secondary"),
                        ], style={"display": "flex", "marginBottom": "10px"}),
                        html.Div([
                            html.Span("Speed: ", style={"color": T["dim"], "fontSize": "10px"}),
                            html.Span(id="speed-display", children="1x", className="mono", style={"color": T["accent"]}),
                        ], style={"marginBottom": "4px"}),
                        dcc.Slider(id="speed-slider", min=0.5, max=5, step=0.5, value=1, 
                                   marks={0.5: "0.5x", 1: "1x", 2: "2x", 5: "5x"}),
                    ], style={"padding": "10px"}),
                ], className="card", style={"marginBottom": "10px"}),
                
                # Progress
                html.Div([
                    html.Div("PROGRESS", className="card-header"),
                    html.Div([
                        html.Div([
                            html.Span(id="current-sample", children="0", className="mono", style={"color": T["accent"], "fontSize": "16px"}),
                            html.Span(" / ", style={"color": T["dim"]}),
                            html.Span(id="total-samples", children="0", className="mono"),
                        ]),
                        html.Div(id="progress-bar", style={"height": "4px", "background": T["header"], "borderRadius": "2px", "overflow": "hidden", "marginTop": "6px"}),
                    ], style={"padding": "10px"}),
                ], className="card", style={"marginBottom": "10px"}),
                
                # Threshold Config
                html.Div([
                    html.Div("ANOMALY THRESHOLD", className="card-header"),
                    html.Div([
                        html.Div([
                            html.Span("Z-Score: ", style={"color": T["dim"], "fontSize": "10px"}),
                            html.Span(id="threshold-display", children="2.5", className="mono", style={"color": T["accent"]}),
                        ], style={"marginBottom": "4px"}),
                        dcc.Slider(id="threshold-slider", min=1, max=5, step=0.5, value=2.5,
                                   marks={1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}),
                    ], style={"padding": "10px"}),
                ], className="card", style={"marginBottom": "10px"}),
                
                # Email Config
                html.Div([
                    html.Div("📧 EMAIL ALERTS", className="card-header"),
                    html.Div([
                        dcc.Checklist(id="email-toggle", options=[{"label": " Enable", "value": "on"}], value=[],
                                      style={"fontSize": "11px", "marginBottom": "8px"}),
                        dcc.Input(id="email-sender", type="email", placeholder="Your email",
                                  style={"width": "100%", "padding": "6px", "marginBottom": "4px", "fontSize": "10px",
                                         "background": T["header"], "border": f"1px solid {T['border']}",
                                         "borderRadius": "4px", "color": T["text"]}),
                        dcc.Input(id="email-password", type="password", placeholder="App password",
                                  style={"width": "100%", "padding": "6px", "marginBottom": "4px", "fontSize": "10px",
                                         "background": T["header"], "border": f"1px solid {T['border']}",
                                         "borderRadius": "4px", "color": T["text"]}),
                        html.Div(id="email-status", style={"fontSize": "9px", "color": T["dim"]}),
                    ], style={"padding": "10px"}),
                ], className="card", style={"marginBottom": "10px"}),
                
                # Statistics
                html.Div([
                    html.Div("STATISTICS", className="card-header"),
                    html.Div(id="stats-panel", style={"padding": "10px"}),
                ], className="card", style={"marginBottom": "10px"}),
                
                # Early Warning
                html.Div([
                    html.Div([
                        html.Span("⚠ EARLY WARNING"),
                        html.Span(id="early-warning-badge", className="mono", 
                                  style={"marginLeft": "8px", "background": T["warning"], "padding": "2px 8px", 
                                         "borderRadius": "10px", "fontSize": "9px", "fontWeight": "600"}),
                    ], className="card-header", style={"background": T["warning"] + "20"}),
                    html.Div(id="early-warning-panel", style={"padding": "8px"}),
                ], id="early-warning-card", className="card", style={"marginBottom": "10px", "border": f"2px solid {T['warning']}"}),
                
                # Alerts - Enhanced visibility
                html.Div([
                    html.Div([
                        html.Span("🚨 ALERTS"),
                        html.Span(id="alert-count", className="mono", 
                                  style={"marginLeft": "8px", "background": T["danger"], "padding": "3px 10px", 
                                         "borderRadius": "12px", "fontSize": "11px", "fontWeight": "700"}),
                    ], className="card-header", style={"background": T["danger"] + "30", "padding": "10px 12px"}),
                    html.Div(id="alerts-panel", style={"padding": "8px", "maxHeight": "150px", "overflowY": "auto"}),
                ], id="alerts-card", className="card", style={"border": f"2px solid {T['danger']}"}),
            ], style={"width": "220px", "padding": "10px", "flexShrink": "0", "overflowY": "auto"}),
            
            # Main Panel
            html.Div([
                # Top Status Row
                html.Div([
                    # Anomaly Status
                    html.Div([
                        html.Div("STATUS", className="card-header"),
                        html.Div(id="anomaly-display", style={"padding": "15px", "textAlign": "center"}),
                    ], className="card", style={"flex": "1"}),
                    
                    # Failure Probability
                    html.Div([
                        html.Div("FAILURE PROBABILITY", className="card-header"),
                        html.Div(id="failure-prob", style={"padding": "15px", "textAlign": "center"}),
                    ], className="card", style={"flex": "1", "marginLeft": "10px"}),
                    
                    # Remaining Useful Life
                    html.Div([
                        html.Div("REMAINING USEFUL LIFE", className="card-header"),
                        html.Div(id="rul-display", style={"padding": "15px", "textAlign": "center"}),
                    ], className="card", style={"flex": "1", "marginLeft": "10px"}),
                    
                    # Health Score
                    html.Div([
                        html.Div("HEALTH SCORE", className="card-header"),
                        html.Div(id="health-display", style={"padding": "15px", "textAlign": "center"}),
                    ], className="card", style={"flex": "1", "marginLeft": "10px"}),
                ], style={"display": "flex", "marginBottom": "10px"}),
                
                # Live Values Row
                html.Div([
                    html.Div("LIVE SENSOR READINGS", className="card-header"),
                    html.Div([
                        sensor_box("RMS", "val-rms"),
                        sensor_box("KURTOSIS", "val-kurtosis"),
                        sensor_box("MEAN", "val-mean"),
                        sensor_box("Z-SCORE", "val-zscore"),
                        sensor_box("BASELINE μ", "val-baseline"),
                        sensor_box("BASELINE σ", "val-std"),
                    ], style={"display": "flex", "gap": "20px", "padding": "12px", "justifyContent": "center"}),
                ], className="card", style={"marginBottom": "10px"}),
                
                # Charts Row 1
                html.Div([
                    html.Div([
                        html.Div("HEALTH GAUGE", className="card-header"),
                        dcc.Graph(id="gauge-chart", config={"displayModeBar": False}, style={"height": "160px"}),
                    ], className="card", style={"flex": "1"}),
                    
                    html.Div([
                        html.Div("SENSOR TRENDS (LIVE)", className="card-header"),
                        dcc.Graph(id="trends-chart", config={"displayModeBar": False}, style={"height": "160px"}),
                    ], className="card", style={"flex": "2", "marginLeft": "10px"}),
                ], style={"display": "flex", "marginBottom": "10px"}),
                
                # Charts Row 2
                html.Div([
                    html.Div([
                        html.Div("ANOMALY TIMELINE", className="card-header"),
                        dcc.Graph(id="anomaly-chart", config={"displayModeBar": False}, style={"height": "130px"}),
                    ], className="card", style={"flex": "1"}),
                    
                    html.Div([
                        html.Div("HEALTH PREDICTION (48H)", className="card-header"),
                        dcc.Graph(id="prediction-chart", config={"displayModeBar": False}, style={"height": "130px"}),
                    ], className="card", style={"flex": "1", "marginLeft": "10px"}),
                ], style={"display": "flex", "marginBottom": "10px"}),
                
                # Distribution Chart
                html.Div([
                    html.Div("DATA DISTRIBUTION", className="card-header"),
                    dcc.Graph(id="distribution-chart", config={"displayModeBar": False}, style={"height": "120px"}),
                ], className="card"),
                
            ], style={"flex": "1", "padding": "10px", "overflowY": "auto"}),
        ], style={"display": "flex", "height": "calc(100vh - 40px)"}),
    ])


def sensor_box(label, vid):
    return html.Div([
        html.Div(label, style={"fontSize": "9px", "color": T["dim"], "letterSpacing": "0.5px"}),
        html.Div(id=vid, children="--", className="mono", style={"fontSize": "16px", "color": T["accent"], "marginTop": "2px"}),
    ], style={"textAlign": "center"})


def register_callbacks(app: dash.Dash):
    
    @app.callback(Output("file-info", "children"), Input("csv-upload", "contents"), 
                  State("csv-upload", "filename"), prevent_initial_call=True)
    def upload_csv(contents, filename):
        if not contents: return ""
        try:
            _, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            STATE["csv"] = df
            STATE["idx"] = 0
            STATE["history"] = {"rms": [], "kurtosis": [], "mean": [], "health": [], "zscore": [], "anomaly": []}
            STATE["alerts"] = []
            # Pre-calculate statistics
            if "rms" in df.columns:
                STATE["stats"] = {
                    "rms_mean": float(df["rms"].mean()),
                    "rms_std": float(df["rms"].std()),
                    "rms_min": float(df["rms"].min()),
                    "rms_max": float(df["rms"].max()),
                    "baseline_mean": float(df["rms"].head(100).mean()),
                    "baseline_std": float(df["rms"].head(100).std()),
                }
            return f"✓ {filename} ({len(df)} rows)"
        except Exception as e:
            return f"✗ {str(e)[:25]}"
    
    @app.callback(Output("btn-pause", "children"), Input("btn-pause", "n_clicks"), prevent_initial_call=True)
    def toggle_pause(n):
        STATE["paused"] = not STATE["paused"]
        return "▶" if STATE["paused"] else "⏸"
    
    @app.callback(Output("speed-display", "children"), Input("speed-slider", "value"))
    def update_speed(v):
        STATE["speed"] = v
        return f"{v}x"
    
    @app.callback(Output("threshold-display", "children"), Input("threshold-slider", "value"))
    def update_threshold(v):
        STATE["threshold"] = v
        return str(v)
    
    @app.callback(Output("current-sample", "children", allow_duplicate=True), 
                  Input("btn-reset", "n_clicks"), prevent_initial_call=True)
    def reset_playback(n):
        STATE["idx"] = 0
        STATE["history"] = {"rms": [], "kurtosis": [], "mean": [], "health": [], "zscore": [], "anomaly": []}
        STATE["alerts"] = []
        return "0"
    
    @app.callback(Output("current-sample", "children", allow_duplicate=True),
                  Input("btn-skip", "n_clicks"), prevent_initial_call=True)
    def skip_forward(n):
        if STATE["csv"] is not None:
            STATE["idx"] = min(STATE["idx"] + 50, len(STATE["csv"]) - 1)
        return str(STATE["idx"])
    
    @app.callback(
        Output("email-status", "children"),
        [Input("email-toggle", "value"), Input("email-sender", "value"), Input("email-password", "value")],
        prevent_initial_call=True
    )
    def update_email_config(toggle, sender, password):
        STATE["email_enabled"] = "on" in (toggle or [])
        STATE["email_sender"] = sender or ""
        STATE["email_password"] = password or ""
        if STATE["email_enabled"] and sender and password:
            email_notifier.configure(sender, password, STATE["email_recipient"])
            return f"✓ Will send to {STATE['email_recipient']}"
        elif STATE["email_enabled"]:
            return "⚠ Enter credentials"
        return ""
    
    @app.callback(
        [
            Output("time-display", "children"), Output("system-status", "children"), Output("status-dot", "style"),
            Output("current-sample", "children"), Output("total-samples", "children"), Output("progress-bar", "children"),
            Output("anomaly-display", "children"), Output("failure-prob", "children"), Output("rul-display", "children"),
            Output("health-display", "children"), Output("val-rms", "children"), Output("val-kurtosis", "children"),
            Output("val-mean", "children"), Output("val-zscore", "children"), Output("val-baseline", "children"),
            Output("val-std", "children"), Output("stats-panel", "children"), Output("alerts-panel", "children"),
            Output("alert-count", "children"), Output("early-warning-panel", "children"), Output("early-warning-badge", "children"),
            Output("gauge-chart", "figure"), Output("trends-chart", "figure"),
            Output("anomaly-chart", "figure"), Output("prediction-chart", "figure"), Output("distribution-chart", "figure"),
        ],
        Input("tick", "n_intervals"),
    )
    def process_tick(n):
        time_str = datetime.now().strftime("%H:%M:%S")
        csv = STATE["csv"]
        
        if csv is None:
            empty = empty_fig()
            ew_panel = html.Div("No data", style={"color": T["dim"], "textAlign": "center", "fontSize": "10px"})
            return (time_str, "IDLE", dot_style(T["muted"]), "0", "0", None,
                    status_el("NO DATA", T["dim"]), prob_el(0), rul_el("--"), health_el(0),
                    "--", "--", "--", "--", "--", "--", stats_empty(), alerts_empty(), "0",
                    ew_panel, "--", empty, empty, empty, empty, empty)
        
        if STATE["paused"]:
            idx = STATE["idx"]
        else:
            idx = STATE["idx"]
            STATE["idx"] = (idx + int(STATE["speed"])) % len(csv)
        
        total = len(csv)
        row = csv.iloc[idx]
        
        # Extract values
        rms = float(row["rms"]) if "rms" in row.index else 0.0
        kurtosis = float(row["kurtosis"]) if "kurtosis" in row.index else 0.0
        mean = float(row["mean"]) if "mean" in row.index else 0.0
        
        stats = STATE["stats"]
        bl_mean = stats.get("baseline_mean", 0.08)
        bl_std = stats.get("baseline_std", 0.01)
        
        z_score = abs(rms - bl_mean) / max(bl_std, 1e-9)
        is_anomaly = z_score > STATE["threshold"]
        health = max(0, min(100, 100 - (z_score * 12)))
        
        # Failure probability (based on z-score)
        fail_prob = min(100, max(0, (z_score - 1) * 25))
        
        # RUL estimation (samples until predicted failure)
        if health > 30:
            rul = int((health - 30) / (z_score * 0.5 + 0.1) * 10)
        else:
            rul = 0
        
        # Store history
        for key, val in [("rms", rms), ("kurtosis", kurtosis), ("mean", mean), 
                         ("health", health), ("zscore", z_score), ("anomaly", 1 if is_anomaly else 0)]:
            STATE["history"][key].append(val)
            if len(STATE["history"][key]) > 300:
                STATE["history"][key] = STATE["history"][key][-300:]
        
        # Add alert
        if is_anomaly:
            if not STATE["alerts"] or STATE["alerts"][0].get("idx") != idx:
                STATE["alerts"].insert(0, {
                    "idx": idx, "msg": f"#{idx}: Z={z_score:.2f}, H={health:.0f}%",
                    "sev": "critical" if health < 30 else "warning"
                })
                STATE["alerts"] = STATE["alerts"][:20]
                
                # Send email alert if enabled (rate limit: 50 samples between emails)
                if STATE["email_enabled"] and email_notifier.is_configured():
                    if idx - STATE["last_email_idx"] >= 50:
                        success = send_anomaly_email(
                            sample=idx, rms=rms, kurtosis=kurtosis,
                            mean=mean, z_score=z_score, health=health
                        )
                        if success:
                            STATE["last_email_idx"] = idx
                            logger.info(f"Email sent for anomaly at sample {idx}")
        
        # Build outputs
        progress = html.Div(style={"width": f"{(idx/total)*100}%", "height": "100%", "background": T["accent"]})
        
        # Status
        if is_anomaly:
            status = "ANOMALY"
            dot = dot_style(T["danger"])
            anom_el = status_el("⚠ ANOMALY", T["danger"], pulse=True)
        else:
            status = "NORMAL"
            dot = dot_style(T["success"])
            anom_el = status_el("✓ NORMAL", T["success"])
        
        # Stats panel
        stats_panel = html.Div([
            stat_row("Samples", str(idx), T["accent"]),
            stat_row("Anomalies", str(sum(STATE["history"]["anomaly"])), T["danger"]),
            stat_row("Avg Health", f"{np.mean(STATE['history']['health']) if STATE['history']['health'] else 0:.0f}%", T["success"]),
            stat_row("Max Z-Score", f"{max(STATE['history']['zscore']) if STATE['history']['zscore'] else 0:.2f}", T["warning"]),
        ])
        
        # Alerts panel - Enhanced visibility
        alerts_els = []
        for a in STATE["alerts"][:10]:
            c = T["danger"] if a["sev"] == "critical" else T["warning"]
            icon = "🔴" if a["sev"] == "critical" else "🟠"
            alerts_els.append(html.Div([
                html.Span(icon, style={"marginRight": "8px", "fontSize": "12px"}),
                html.Span(a["msg"], style={"fontSize": "11px", "fontWeight": "500", "color": c})
            ], style={"padding": "6px 4px", "borderBottom": f"1px solid {T['border']}", "background": c + "15"}))
        if not alerts_els:
            alerts_els = html.Div("✓ No alerts", style={"color": T["success"], "textAlign": "center", "padding": "15px", "fontSize": "12px"})
        
        # Early warning panel
        # Simulate early warning based on current trend
        ew_hours = None
        ew_confidence = 0
        if len(STATE["history"]["zscore"]) > 10:
            recent_zscores = STATE["history"]["zscore"][-10:]
            trend = (recent_zscores[-1] - recent_zscores[0]) / 10 if len(recent_zscores) > 1 else 0
            if trend > 0.05:  # Degrading trend
                ew_hours = max(2, min(48, int((STATE["threshold"] - z_score) / (trend + 0.01))))
                ew_confidence = min(95, int(50 + trend * 200))
        
        if ew_hours and ew_hours <= 48:
            severity_color = T["danger"] if ew_hours <= 6 else T["warning"] if ew_hours <= 24 else T["info"]
            ew_panel = html.Div([
                html.Div([
                    html.Span("⚠", style={"fontSize": "20px", "marginRight": "8px"}),
                    html.Span(f"~{ew_hours}h", className="mono", style={"fontSize": "20px", "fontWeight": "700", "color": severity_color}),
                ], style={"textAlign": "center", "marginBottom": "6px"}),
                html.Div(f"Confidence: {ew_confidence}%", style={"fontSize": "10px", "color": T["dim"], "textAlign": "center"}),
                html.Div("Degradation detected", style={"fontSize": "9px", "color": severity_color, "textAlign": "center", "marginTop": "4px"}),
            ])
            ew_badge = f"~{ew_hours}h"
        else:
            ew_panel = html.Div([
                html.Div("✓", style={"fontSize": "18px", "color": T["success"], "textAlign": "center"}),
                html.Div("No imminent issues", style={"fontSize": "10px", "color": T["dim"], "textAlign": "center"}),
            ])
            ew_badge = "OK"
        
        # Charts
        gauge = gauge_chart(health)
        trends = trends_chart(STATE["history"])
        anom_chart = anomaly_timeline(STATE["history"])
        pred = prediction_chart(health)
        dist = distribution_chart(STATE["history"])
        
        return (
            time_str, status, dot, str(idx), str(total), progress,
            anom_el, prob_el(fail_prob), rul_el(rul), health_el(health),
            f"{rms:.5f}", f"{kurtosis:.4f}", f"{mean:.6f}", f"{z_score:.2f}",
            f"{bl_mean:.5f}", f"{bl_std:.5f}", stats_panel, alerts_els, str(len(STATE["alerts"])),
            ew_panel, ew_badge, gauge, trends, anom_chart, pred, dist
        )


def dot_style(color): return {"color": color, "marginRight": "6px"}

def status_el(text, color, pulse=False):
    cls = "pulse" if pulse else ""
    return html.Span(text, className=cls, style={"fontSize": "18px", "fontWeight": "600", "color": color})

def health_el(h):
    c = T["success"] if h >= 70 else T["warning"] if h >= 40 else T["danger"]
    return html.Div([
        html.Span(f"{h:.0f}", className="mono", style={"fontSize": "28px", "fontWeight": "600", "color": c}),
        html.Span("%", style={"fontSize": "12px", "color": T["dim"]}),
    ])

def prob_el(p):
    c = T["success"] if p < 30 else T["warning"] if p < 60 else T["danger"]
    return html.Div([
        html.Span(f"{p:.0f}", className="mono", style={"fontSize": "28px", "fontWeight": "600", "color": c}),
        html.Span("%", style={"fontSize": "12px", "color": T["dim"]}),
    ])

def rul_el(r):
    if r == "--": return html.Span("--", style={"fontSize": "28px", "color": T["dim"]})
    c = T["success"] if r > 100 else T["warning"] if r > 30 else T["danger"]
    return html.Div([
        html.Span(str(r), className="mono", style={"fontSize": "28px", "fontWeight": "600", "color": c}),
        html.Span(" samples", style={"fontSize": "10px", "color": T["dim"]}),
    ])

def stat_row(label, val, color):
    return html.Div([
        html.Span(label, style={"color": T["dim"], "fontSize": "10px"}),
        html.Span(val, className="mono", style={"color": color, "fontWeight": "500"}),
    ], style={"display": "flex", "justifyContent": "space-between", "padding": "4px 0"})

def stats_empty():
    return html.Div("No data", style={"color": T["dim"], "textAlign": "center", "padding": "10px"})

def alerts_empty():
    return html.Div("Upload CSV", style={"color": T["dim"], "textAlign": "center", "padding": "10px"})

def empty_fig():
    fig = go.Figure()
    fig.update_layout(paper_bgcolor=T["card"], plot_bgcolor=T["card"], margin=dict(l=10, r=10, t=10, b=10))
    return fig

def gauge_chart(health):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=health,
        number={"suffix": "%", "font": {"size": 22, "color": T["text"]}},
        gauge={"axis": {"range": [0, 100], "tickcolor": T["dim"], "tickwidth": 1},
               "bar": {"color": T["accent"], "thickness": 0.25},
               "bgcolor": T["header"],
               "steps": [{"range": [0, 30], "color": "rgba(248,81,73,0.2)"},
                        {"range": [30, 70], "color": "rgba(210,153,34,0.2)"},
                        {"range": [70, 100], "color": "rgba(63,185,80,0.2)"}]}
    ))
    fig.update_layout(paper_bgcolor=T["card"], margin=dict(l=15, r=15, t=20, b=5), font=dict(color=T["text"], size=9))
    return fig

def trends_chart(history):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        row_heights=[0.6, 0.4])
    
    fig.add_trace(go.Scatter(y=history["rms"], mode="lines", name="RMS",
                            line=dict(color=T["accent"], width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(y=history["health"], mode="lines", name="Health",
                            line=dict(color=T["success"], width=1.5)), row=2, col=1)
    
    fig.update_layout(
        paper_bgcolor=T["card"], plot_bgcolor=T["card"],
        margin=dict(l=35, r=10, t=5, b=20), showlegend=False,
        font=dict(color=T["dim"], size=8)
    )
    fig.update_xaxes(gridcolor=T["border"], showticklabels=False)
    fig.update_yaxes(gridcolor=T["border"], title_font_size=9)
    fig.update_yaxes(title="RMS", row=1, col=1)
    fig.update_yaxes(title="Health", row=2, col=1)
    return fig

def anomaly_timeline(history):
    fig = go.Figure()
    
    anomalies = history["anomaly"]
    colors = [T["danger"] if a else T["success"] for a in anomalies]
    
    fig.add_trace(go.Bar(y=[1]*len(anomalies), marker_color=colors, showlegend=False))
    
    fig.update_layout(
        paper_bgcolor=T["card"], plot_bgcolor=T["card"],
        margin=dict(l=10, r=10, t=5, b=20),
        xaxis=dict(title="Sample", title_font_size=9, gridcolor=T["border"]),
        yaxis=dict(visible=False),
        font=dict(color=T["dim"], size=8), bargap=0
    )
    return fig

def prediction_chart(current_health):
    hours = list(range(48))
    np.random.seed(42)
    predicted = np.clip(current_health - np.cumsum(np.random.uniform(0.3, 0.7, 48)), 0, 100)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=predicted, mode="lines", line=dict(color=T["accent"], width=2)))
    fig.add_hline(y=30, line_dash="dot", line_color=T["danger"], line_width=1)
    
    fig.update_layout(
        paper_bgcolor=T["card"], plot_bgcolor=T["card"],
        margin=dict(l=30, r=10, t=5, b=25),
        xaxis=dict(title="Hours", title_font_size=9, gridcolor=T["border"]),
        yaxis=dict(title="Health %", title_font_size=9, gridcolor=T["border"], range=[0, 100]),
        font=dict(color=T["dim"], size=8), showlegend=False
    )
    return fig

def distribution_chart(history):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("RMS", "Kurtosis", "Z-Score"))
    
    if history["rms"]:
        fig.add_trace(go.Histogram(x=history["rms"], marker_color=T["accent"], opacity=0.7), row=1, col=1)
        fig.add_trace(go.Histogram(x=history["kurtosis"], marker_color=T["info"], opacity=0.7), row=1, col=2)
        fig.add_trace(go.Histogram(x=history["zscore"], marker_color=T["warning"], opacity=0.7), row=1, col=3)
    
    fig.update_layout(
        paper_bgcolor=T["card"], plot_bgcolor=T["card"],
        margin=dict(l=30, r=10, t=25, b=20),
        font=dict(color=T["dim"], size=8), showlegend=False
    )
    fig.update_xaxes(gridcolor=T["border"])
    fig.update_yaxes(gridcolor=T["border"])
    return fig


def run_dashboard(data_store: Optional[Dict] = None):
    app = create_app(data_store)
    logger.info(f"Dashboard: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=DASHBOARD_DEBUG)


if __name__ == "__main__":
    run_dashboard()
