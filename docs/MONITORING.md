# Monitoring & Observability

Track reliability, quotas, and performance to keep the Pharmaceutical RAG deployment healthy.

## Metrics Sources

- `scripts/performance_monitor.py` – collects latency, throughput, token usage.
- `src/monitoring/credit_tracker.py` – NVIDIA Build credit consumption.
- `src/monitoring/endpoint_health_monitor.py` – endpoint status + retry counts.

## Dashboards

1. **CLI Snapshot**
   ```bash
   python scripts/performance_monitor.py --summary
   ```
2. **Continuous Watcher**
   ```bash
   python scripts/performance_monitor.py --interval 60 --export metrics.json
   ```
3. **Quota Tracking**
   ```bash
   python -m src.monitoring.credit_tracker
   ```

## Alerts

- Configure PagerDuty/Webhook targets in `config/monitoring.yaml` (create if missing).
- Use `ENABLE_MONITORING_ALERTS=true` to enable notifier hooks in production.

## Troubleshooting Sequence

1. Run `python scripts/nvidia_build_api_test.py` for connectivity.
2. Check `logs/monitoring/*.log` for retries or throttling.
3. Escalate using the [Support](SUPPORT.md) contacts if outages persist >15 minutes.
