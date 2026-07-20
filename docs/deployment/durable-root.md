# Vulcan durable-root operations

The production image and Helm chart use one canonical durable root: `/var/lib/vulcan`.
It is owned by UID/GID `1001`, mounted from a single ReadWriteOnce volume/PVC, and kept writable while the root filesystem remains read-only.

Authoritative persistent owners live below typed subdirectories: `audit/`, `alignment/`, `domains/`, `memory/`, `learning/outbox/`, `csiu/`, `approval/`, and `improvement/`.
Model scratch and caches are ephemeral (`/tmp/vulcan-cache`) and may be discarded on restart.

Backup and restore assumptions:

- Back up the complete `/var/lib/vulcan` volume as one consistency unit while the single writer is stopped or quiesced.
- Restore the full directory tree with UID/GID `1001:1001` and mode `0700` before starting the pod/container.
- Capacity planning starts at the Helm PVC size (`50Gi` by default); increase it before audit or memory growth reaches the storage-class expansion threshold.
- Disaster recovery assumes one active writer. Do not run multiple replicas with the SQLite durable backend; promote exactly one restored volume.

Rollback: stop the new workload, restore the previous `/var/lib/vulcan` snapshot, and redeploy the previous image/chart revision. Ephemeral caches do not need restoration.
