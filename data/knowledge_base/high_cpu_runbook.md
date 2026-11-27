# High CPU utilisation runbook

## Symptoms
CPU utilisation rises sharply, often with higher application response time and error counts. Latency may rise slightly because requests queue behind busy workers. CPU that stays high while the request rate is flat or falling suggests load that is not explained by user traffic.

## Investigation steps
- Identify the top CPU-consuming processes on the affected server (for example with top or pidstat).
- Compare CPU utilisation with the request rate: is the extra CPU explained by more traffic?
- Check for runaway batch jobs, cron tasks, backups or malware scans started around the same time.
- Review recent deployments, configuration changes and feature flags on the server.
- Check for CPU throttling or noisy neighbours if the server is virtualised.

## Possible causes
Traffic surge, inefficient code path after a release, runaway background job, crypto-mining malware, resource contention on a shared host.
