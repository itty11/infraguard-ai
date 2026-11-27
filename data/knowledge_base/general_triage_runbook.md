# General infrastructure triage runbook

## Symptoms
Several metrics deviate from normal at the same time, such as CPU, memory, latency and packet loss. Grouped anomalies describe symptoms; they do not by themselves prove a root cause.

## Investigation steps
- Establish the exact start time of the anomaly and list changes made shortly before it (deployments, configuration, maintenance).
- Check whether other servers on the same network segment or application tier are also affected.
- Examine workload: did request rate or traffic volume change?
- Review server resource consumption first, then network congestion and interface errors.
- Collect logs from the affected period before restarting anything, so evidence is preserved.
- Escalate with a timeline of symptoms, affected metrics and actions taken.

## Possible causes
Workload change, release regression, shared dependency failure, network issue, hardware fault.
