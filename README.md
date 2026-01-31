# Predictive Maintenance System

An advanced predictive maintenance system for industrial equipment with real-time anomaly detection, health monitoring, and email alerts.

![Dashboard](docs/dashboard.png)

## Features

- 🔍 **Anomaly Detection** - Z-score and Isolation Forest algorithms
- 📊 **Real-time Dashboard** - Professional industrial-grade UI
- 📈 **Health Monitoring** - Live equipment health scoring
- 🔮 **Failure Prediction** - Remaining useful life estimation
- 📧 **Email Alerts** - Automated notifications with data attachments
- 📁 **CSV Upload** - Analyze any sensor data file
- ⏯️ **Playback Controls** - Pause, speed control, skip
- 📉 **Data Visualization** - Trends, distributions, timelines

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python -m src.pipeline --mode dashboard
```

Then open **http://localhost:8050** and upload your CSV file.

## CSV Format

Your CSV should have these columns:
- `timestamp` - DateTime
- `rms` - RMS vibration value
- `kurtosis` - Kurtosis value
- `mean` - Mean value

## Email Alerts Setup

1. Enable email alerts in the dashboard sidebar
2. Enter your Gmail address
3. Create a [Gmail App Password](https://myaccount.google.com/apppasswords)
4. Enter the app password

## Project Structure

```
src/
├── pipeline.py          # Main orchestration
├── config.py            # Configuration
├── data/
│   ├── loader.py        # Data loading
│   └── preprocessing.py # Data preprocessing
├── models/
│   ├── anomaly_detector.py
│   ├── health_scorer.py
│   └── forecaster.py
├── alerts/
│   ├── alert_engine.py
│   └── email_notifier.py
└── dashboard/
    └── app.py           # Dash dashboard
```

## Requirements

- Python 3.9+
- pandas, numpy, scipy
- scikit-learn
- plotly, dash
- dash-bootstrap-components

## License

MIT
