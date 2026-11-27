# Sudden latency surge runbook

## Symptoms
Latency jumps abruptly while CPU and memory stay normal. Response time rises with latency. Packet loss may or may not be present.

## Investigation steps
- Check DNS resolution time and upstream dependency latency (database, API, storage).
- Measure latency from several clients to separate server-side from path-related causes.
- Look for a route change, VPN or load balancer failover around the start time.
- Check whether a single dependency such as the database shows slow queries.
- Review recent network or infrastructure maintenance windows.

## Possible causes
Routing change, slow upstream dependency, DNS issue, load balancer problem, noisy neighbour on shared network.
