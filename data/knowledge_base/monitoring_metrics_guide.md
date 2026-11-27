# Monitoring metrics guide

## What the metrics mean
CPU utilisation is the share of processor time in use. Memory utilisation is the share of RAM in use. Disk usage is occupied capacity; disk I/O is read and write throughput. Network in and out are traffic rates. Latency is network round-trip time. Packet loss is the share of packets that never arrive. Response time is how long the application takes to answer a request. Error count and restart events come from application and system logs.

## Why fixed thresholds miss problems
A fixed alert such as CPU above 90 percent misses abnormal behaviour that stays below the threshold, and ignores that normal load changes through the day. Anomaly detection compares each measurement with what is normal for that server at that time and with the other metrics at the same moment.
