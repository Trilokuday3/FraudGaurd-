# infra/kafka-vm-setup.md

Manual setup for the VM hosting the live streaming pipeline's Kafka
broker. Like infra/deploy.md's Neon/Render/Vercel steps, this is a
one-time manual process only you can do -- create the account, verify
current free-tier terms yourself (they change; this doc doesn't pin a
specific provider's exact numbers).

> Status: the compose file and this runbook were written without access to
> Docker or a VM and have not been executed. Treat the verification
> checklist at the end as the real test, and expect to fix small issues.

## 1. Provision the VM

Any perpetual (not trial) free compute tier works. At minimum: 1 GB RAM,
a public IP, Docker installable. Note the VM's public IP/hostname --
this becomes `VM_PUBLIC_HOST` below and `KAFKA_BOOTSTRAP_SERVERS`
(`<host>:9092`) in the GitHub Actions secrets (Task 7). Also set the
`VM_HOST` GitHub Actions secret to the same host you use for SSH (Task 7
rsyncs as `deploy@<VM_HOST>`).

## 2. Install Docker

Follow your VM OS's standard Docker install instructions (including the
Docker Compose plugin). Copy `streaming/docker-compose.kafka.yml` to the
VM (or clone the repo there).

## 3. Choose SASL credentials

Pick a username/password for the Kafka broker's PLAIN SASL mechanism.
Use a strong, random password: port 9092 is public. Keep the username
simple (letters, digits, underscore) and the password alphanumeric (no
quotes, backslash or `$`), because both are embedded in a JAAS string and
a properties file without escaping. These become
`KAFKA_SASL_USERNAME` / `KAFKA_SASL_PASSWORD` -- set them as GitHub
Actions **secrets** (Task 7), never committed.

## 4. Start the broker

```
export VM_PUBLIC_HOST=<your VM's public IP or hostname>
export KAFKA_SASL_USERNAME=<chosen username>
export KAFKA_SASL_PASSWORD=<chosen password>
docker compose -f streaming/docker-compose.kafka.yml up -d
```

Note: every `docker compose ... exec` (or `ps`, `logs`, etc.) re-interpolates
the compose file, so `VM_PUBLIC_HOST`, `KAFKA_SASL_USERNAME` and
`KAFKA_SASL_PASSWORD` must still be exported in that shell (in a new SSH
session, export them again). Missing values make compose fail fast.

## 5. Firewall

Open port 9092 to the internet (GitHub Actions runners need to reach
it) -- but only 9092 for Kafka (plus SSH, port 22, for step 7). Do not
open the controller port (9093). Check both the VM's OS firewall and the
provider's network/security-group rules.

## 6. Create the topic

The Kafka CLI scripts in the `apache/kafka` image live in
`/opt/kafka/bin/`. First write a small client config inside the container
(the variables expand on the VM shell, so they must be exported as in
step 4):

```
docker compose -f streaming/docker-compose.kafka.yml exec kafka sh -c 'cat > /tmp/client.properties <<EOF
security.protocol=SASL_PLAINTEXT
sasl.mechanism=PLAIN
sasl.jaas.config=org.apache.kafka.common.security.plain.PlainLoginModule required username="'"$KAFKA_SASL_USERNAME"'" password="'"$KAFKA_SASL_PASSWORD"'";
EOF'
```

Create the topic:

```
docker compose -f streaming/docker-compose.kafka.yml exec kafka \
  /opt/kafka/bin/kafka-topics.sh --create --topic fraudguard.transactions \
  --bootstrap-server localhost:9092 \
  --command-config /tmp/client.properties
```

Confirm SASL auth works by listing topics with the same config (should
show `fraudguard.transactions`):

```
docker compose -f streaming/docker-compose.kafka.yml exec kafka \
  /opt/kafka/bin/kafka-topics.sh --list \
  --bootstrap-server localhost:9092 \
  --command-config /tmp/client.properties
```

Negative check -- connecting without credentials must be rejected (expect
an authentication/timeout error, not a topic list):

```
docker compose -f streaming/docker-compose.kafka.yml exec kafka \
  /opt/kafka/bin/kafka-topics.sh --list \
  --bootstrap-server localhost:9092
```

This may hang while retrying; Ctrl-C after a few seconds. The point is
that it must not return a topic list.

## 7. Create the Spark checkpoint directory and SSH access

Task 7's workflow rsyncs the checkpoint as a dedicated `deploy` user, so
create that user (no sudo/admin rights needed) and give it the checkpoint
directory. Run on the VM as an admin user:

```
sudo adduser --disabled-password --gecos "" deploy
sudo mkdir -p /opt/fraudguard/spark-checkpoint
sudo chown -R deploy:deploy /opt/fraudguard/spark-checkpoint
```

(`adduser` is the Debian/Ubuntu spelling; use `useradd -m deploy` on
distros without it.)

Generate a dedicated SSH keypair for GitHub Actions (do not reuse your
personal key). Do this on your own machine, not the VM:

```
ssh-keygen -t ed25519 -f fraudguard-streaming-deploy-key -N ""
```

Install the **public** key for the `deploy` user on the VM (paste the
contents of `fraudguard-streaming-deploy-key.pub` in place of `<PUBKEY>`):

```
sudo -u deploy mkdir -p /home/deploy/.ssh
echo "<PUBKEY>" | sudo -u deploy tee -a /home/deploy/.ssh/authorized_keys
sudo chmod 700 /home/deploy/.ssh
sudo chmod 600 /home/deploy/.ssh/authorized_keys
sudo chown -R deploy:deploy /home/deploy/.ssh
```

Add the **private** key's contents as a GitHub Actions secret
(`VM_SSH_PRIVATE_KEY`, Task 7), and set the `VM_HOST` secret to the same
host you SSH to. Never commit the private key.

## Verification checklist

- [ ] `docker compose -f streaming/docker-compose.kafka.yml ps` shows the broker running
- [ ] From your own machine: `nc -zv <VM_PUBLIC_HOST> 9092` succeeds (port reachable)
- [ ] Listing topics with the SASL client config succeeds and shows `fraudguard.transactions`
- [ ] A connection *without* the SASL credentials is rejected (confirms auth is actually enforced, not just configured)
- [ ] `ssh -i fraudguard-streaming-deploy-key deploy@<VM_PUBLIC_HOST> "ls /opt/fraudguard/spark-checkpoint"` succeeds

## Troubleshooting

If the broker refuses authentication or fails to start, run
`docker compose -f streaming/docker-compose.kafka.yml logs kafka` and check
the generated server.properties (inside the container, under
`/opt/kafka/config/` or the path the logs mention) for the JAAS key first.
It must read `listener.name.sasl_plaintext.plain.sasl.jaas.config` (with an
underscore in `sasl_plaintext`); the compose variable uses three underscores
(`SASL___PLAINTEXT`) to produce that.

## Known limitations

- **No encryption in transit.** The listener is SASL_PLAINTEXT, so
  credentials and message data cross the network unencrypted. Adding TLS
  termination (SASL_SSL or a TLS proxy) is a follow-up.
- **Kafka port 9092 is public.** Anyone can attempt to connect, so use a
  strong, random SASL password and rotate it if it leaks.
- **Single point of failure.** The broker runs on one personally-owned VM
  with no replication or failover (accepted per the spec).
- **Unverified setup.** The compose file and this runbook were written
  without being run; see the status note at the top.
