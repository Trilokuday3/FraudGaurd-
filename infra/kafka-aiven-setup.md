# Kafka on Aiven — one-time setup

The live streaming pipeline (`.github/workflows/live-streaming.yml`) uses an
Aiven for Kafka service as its broker. Aiven authenticates clients with TLS
certificates (mTLS), so there are no usernames or passwords, and no VM to
run.

## 1. Service and topic

- Service: Aiven for Kafka (free plan). In the Aiven console, open the
  service's **Overview** page: the **Service URI** is `<host>:<port>` and is
  the value of `KAFKA_BOOTSTRAP_SERVERS`.
- Topic: `fraudguard.transactions`. Aiven does not create topics on first
  write, so it must exist (Topics tab). It already does for the current
  service.
- Confirm on the plan page what the free tier's topic, partition and
  retention limits are. The pipeline needs one topic and a small backlog,
  but retention decides how much would be re-scored if the checkpoint cache
  were ever lost.

## 2. Client credentials

On the service's **Overview** page, under **Connection information**, pick
**Client certificate** and download the three files:

| File | GitHub secret |
|---|---|
| `ca.pem` | `KAFKA_CA_CERT` |
| `service.cert` (access certificate) | `KAFKA_SERVICE_CERT` |
| `service.key` (access key) | `KAFKA_SERVICE_KEY` |

Each secret's value is the **entire file contents**, including the
`-----BEGIN ...-----` / `-----END ...-----` lines.

**Never commit these files.** `service.key` is a private key that grants
write access to the broker. They are listed in `.gitignore`, but keep them
outside the repo directory once the secrets are set. If a key is ever
committed, pushed, pasted into a chat or otherwise exposed, revoke it: in
the Aiven console, open the service's users, and rotate or delete the
`avnadmin` certificate, then re-download and update the secrets.

## 3. GitHub Actions secrets

Repository → Settings → Secrets and variables → Actions → New repository
secret. Add all five, since the workflow skips itself until every one is
set:

- `DEPLOYED_DECISION_DB_URL` (already set for `mlops-monitor.yml`)
- `KAFKA_BOOTSTRAP_SERVERS`
- `KAFKA_CA_CERT`
- `KAFKA_SERVICE_CERT`
- `KAFKA_SERVICE_KEY`

## 4. Verify the connection (optional, from your machine)

With `kafka-python` installed and the three files in the current directory:

```python
from kafka import KafkaConsumer
c = KafkaConsumer(
    bootstrap_servers="<host>:<port>",
    security_protocol="SSL",
    ssl_cafile="ca.pem", ssl_certfile="service.cert", ssl_keyfile="service.key",
)
print(sorted(c.topics()))   # must include 'fraudguard.transactions'
```

## 5. First run

Run the workflow manually and follow the acceptance checklist in
`infra/deploy.md` ("Live streaming pipeline").

## How the checkpoint works

Spark's checkpoint directory (the offsets it has already consumed) is stored
in the GitHub Actions cache: each successful run saves it under a new key and
the next run restores the newest one. A failed run does not save, so it can't
overwrite good offsets. The cache is only a bookmark; if it is evicted the
consumer restarts from the topic's earliest retained offset and re-scores
whatever is still there.
