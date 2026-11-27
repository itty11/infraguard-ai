# Memory growth and memory leak runbook

## Symptoms
Memory utilisation climbs steadily over tens of minutes or hours instead of following the daily workload pattern. Response time and error counts rise late in the leak, and the server may restart when memory is exhausted.

## Investigation steps
- Plot memory per process over time and find the process whose resident memory keeps growing.
- Check whether the growth started after a deployment or configuration change.
- Look for restart events, out-of-memory kills (dmesg, journalctl) and swap usage.
- Capture a heap profile or memory dump of the suspect process before restarting it.
- Check connection pools, caches and queues for unbounded growth.

## Possible causes
Application memory leak, unbounded cache, connection or file-descriptor leak, large queue backlog.
