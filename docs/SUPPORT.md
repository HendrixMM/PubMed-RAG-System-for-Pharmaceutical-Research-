# Support & Escalation

Follow this runbook when you need assistance operating the Pharmaceutical RAG Template.

## Self-Service

1. Review [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md) for known issues.
2. Check `logs/` for stack traces or API throttling errors.
3. Re-run `python scripts/validate_env.py` to confirm configuration.

## Contact Channels

- **Documentation / Developer Experience:** `#pharma-rag-docs` Slack.
- **Operations:** `#pharma-rag-ops` Slack (24/7 paging window).
- **Email:** pharma-rag@nvidia.com for partner escalations.

## Issue Templates

Open GitHub issues with the following labels:

- `bug`: functional regressions, runtime errors.
- `docs`: inconsistencies in API or benchmarking guides.
- `enhancement`: feature requests.

## Response Targets

| Severity | Target         | Actions                                                   |
| -------- | -------------- | --------------------------------------------------------- |
| Sev 1    | 15 min         | Page ops, stop traffic, share status updates every 30 min |
| Sev 2    | 2 hrs          | Reproduce, roll back, document fix                        |
| Sev 3    | 1 business day | Plan enhancement or documentation update                  |

Escalate to the Operations lead if SLAs are at risk.
