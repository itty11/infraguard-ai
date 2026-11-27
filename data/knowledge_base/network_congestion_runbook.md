# Network congestion and latency runbook

## Symptoms
Network latency increases significantly together with packet loss and more connection failures. Throughput on the interfaces may rise. Application response time degrades because requests wait on the network.

## Investigation steps
- Check interface utilisation, errors and drops on the server and on the upstream switch or router.
- Identify the top talkers: which hosts or services produce the extra traffic?
- Run ping and traceroute (or mtr) to find where latency and loss begin along the path.
- Review recent network changes, link failovers and routing updates.
- Check for retransmissions and TCP resets on the server.

## Possible causes
Link saturation, a misbehaving client or backup job flooding the link, faulty cable or interface, upstream provider issue, routing change.
