# Disk usage and disk I/O saturation runbook

## Symptoms
Disk usage climbs steadily and disk I/O is unusually high. Response time degrades and application errors may appear when the disk fills up or writes are throttled.

## Investigation steps
- Find the largest directories and files (du, ncdu) and the fastest-growing ones.
- Check log rotation, core dumps, temporary files and database write-ahead logs.
- Use iostat or iotop to identify the processes generating most of the I/O.
- Check for backup, indexing or reporting jobs scheduled at the same time.
- Verify free inodes as well as free space.

## Possible causes
Log growth without rotation, runaway write job, large backup, database checkpoint or vacuum activity, failing disk.
