# Packet loss and connection failures runbook

## Symptoms
Packet loss rises above its normal range with more failed connections and slightly higher latency. Applications show timeouts and retries.

## Investigation steps
- Check interface error and drop counters on the server and the connected switch port.
- Test the path with mtr to see which hop drops packets.
- Inspect cable, optics and duplex or speed mismatches on the affected link.
- Check firewall and load balancer health, connection tables and rate limits.
- Compare other servers on the same switch: if they are affected too, suspect shared network equipment.

## Possible causes
Bad cable or transceiver, duplex mismatch, overloaded firewall or load balancer, congested uplink, wireless interference.
