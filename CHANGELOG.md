# Changelog

## [Unreleased]
### Added
- Real-time feature drift monitoring with streaming data
- PagerDuty integration for critical alerts
- Model A/B comparison dashboard
- Automated retraining triggers when drift exceeds thresholds
- REST API for alert configuration management
- Grafana dashboard JSON export
- Time-series model support: ARIMA, Prophet drift detection

### Changed
- Improved Isolation Forest scoring with SHAP explanations
- PSI calculation now uses adaptive binning
- Alert cooldown configurable per-model

### Fixed
- Fixed memory leak in continuous monitoring loop
- Fixed Chi-squared test on zero-frequency bins

## [0.1.0] - 2026-06-01
### Added
- KS-test, Chi-squared, PSI, Wasserstein drift detection
- Isolation Forest anomaly detection
- Slack alerting integration
- FastAPI REST API