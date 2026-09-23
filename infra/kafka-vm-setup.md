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

## 4. Generate the JAAS file and start the broker

The broker reads its SASL credentials from a static JAAS file,
`streaming/kafka_server_jaas.conf`, which the compose file mounts
read-only into the container. The compose file itself no longer
interpolates the credentials -- they are only used here, to generate that
file on the VM. The file holds the password in plain text: it is listed in
`.gitignore`, must be `chmod 600`, and must **never** be committed.

Generate it from the repo checkout's root (the compose file resolves the
mount relative to `streaming/`). Create it **before** the first
`docker compose up`: if the mounted file is missing, Docker creates an
empty directory in its place and the broker fails to start.

```
export VM_PUBLIC_HOST=<your VM's public IP or hostname>
export KAFKA_SASL_USERNAME=<chosen username>
export KAFKA_SASL_PASSWORD=<chosen password>

umask 077
cat > streaming/kafka_server_jaas.conf <<EOF
KafkaServer {
  org.apache.kafka.common.security.plain.PlainLoginModule required
  username="$KAFKA_SASL_USERNAME"
  password="$KAFKA_SASL_PASSWORD"
  user_$KAFKA_SASL_USERNAME="$KAFKA_SASL_PASSWORD";
};
EOF
chmod 600 streaming/kafka_server_jaas.conf

docker compose -f streaming/docker-compose.kafka.yml up -d
```

`username`/`password` are what the broker uses for its own inter-broker
connections; the `user_<name>="<password>"` entry is the account clients
(the producer and the Spark consumer) authenticate as. Both use the same
credentials here.

The container reads the file as its own (non-root) user. If the broker
logs a permission error reading it, keep `chmod 600` and instead
`chown` the file to the container user's UID (check with
`docker compose -f streaming/docker-compose.kafka.yml run --rm kafka id -u`).

Note: every `docker compose ... exec` (or `ps`, `logs`, etc.) re-interpolates
the compose file, so `VM_PUBLIC_HOST` must still be exported in that shell
(in a new SSH session, export it again); a missing value makes compose fail
fast. `KAFKA_SASL_USERNAME` / `KAFKA_SASL_PASSWORD` must also be exported
for step 6's client config.

**Advertised address.** The broker advertises `VM_PUBLIC_HOST:9092`, and
uses that same advertised listener for its own inter-broker connections.
So both the broker itself and the in-container CLI calls in step 6 (which
bootstrap via `localhost:9092`, then follow the advertised address) must be
able to reach the VM's **public** address from inside the VM. If the
provider's network doesn't let a VM reach its own public IP (no hairpin
NAT), or a firewall blocks it, those connections hang or time out -- allow
it, or resolve `VM_PUBLIC_HOST` to the VM's own address inside the
container (e.g. an `extra_hosts` entry).

## 5. Firewall

Open port 9092 to the internet (GitHub Actions runners need to reach
it) -- but only 9092 for Kafka (plus SSH, port 22, for step 7). Do not
open the controller port (9093). Check both the VM's OS firewall and the
provider's network/security-group rules.

## 6. Create the topic

The Kafka CLI scripts in the `apache/kafka` image live in
`/opt/kafka/bin/`. First write a small client config inside the container
(the variables expand on the VM shell, so `KAFKA_SASL_USERNAME` /
`KAFKA_SASL_PASSWORD` must be exported as in step 4):

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
  --config retention.ms=86400000 \
  --command-config /tmp/client.properties
```

`retention.ms=86400000` keeps one day of messages. That bounds disk use
and how much gets rescored if the Spark checkpoint is ever lost (see
"Checkpoint and broker data" below).

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

- [ ] `ls -l streaming/kafka_server_jaas.conf` shows a regular file with mode `-rw-------`, and `git status` does not list it
- [ ] `docker compose -f streaming/docker-compose.kafka.yml ps` shows the broker running
- [ ] From your own machine: `nc -zv <VM_PUBLIC_HOST> 9092` succeeds (port reachable)
- [ ] Listing topics with the SASL client config succeeds and shows `fraudguard.transactions`
- [ ] A connection *without* the SASL credentials is rejected (confirms auth is actually enforced, not just configured)
- [ ] `ssh -i fraudguard-streaming-deploy-key deploy@<VM_PUBLIC_HOST> "ls /opt/fraudguard/spark-checkpoint"` succeeds

## Troubleshooting

If the broker refuses authentication or fails to start, check
`docker compose -f streaming/docker-compose.kafka.yml logs kafka` for
JAAS/SASL errors, and confirm the mounted file path:
`docker compose -f streaming/docker-compose.kafka.yml exec kafka cat /etc/kafka/secrets/kafka_server_jaas.conf`
must print the `KafkaServer` section. "Is a directory" means the file
didn't exist on the VM when the container was created: generate it (step
4), remove the empty directory Docker created in its place, then
`docker compose -f streaming/docker-compose.kafka.yml up -d --force-recreate`.

If CLI calls or broker startup hang or time out, check that the VM can
reach its own public address on 9092 (see "Advertised address" in step 4).

## Checkpoint and broker data

The Spark consumer's checkpoint (`/opt/fraudguard/spark-checkpoint` on the
VM, rsynced by the workflow) records which topic offsets have already been
processed. It must stay consistent with the broker's data volume:

- **Wiping the broker's data volume** (e.g. `docker compose ... down -v`)
  resets the topic's offsets to 0. Clear `/opt/fraudguard/spark-checkpoint`
  on the VM at the same time (`sudo -u deploy sh -c 'rm -rf /opt/fraudguard/spark-checkpoint/*'`);
  otherwise Spark's checkpointed offsets exceed the empty topic's and the
  consumer fails with a data-loss error.
- **Losing the checkpoint** (while the broker data survives) makes the
  consumer restart at `earliest` and rescore everything still retained in
  the topic, writing duplicate decisions. The topic's retention
  (`retention.ms=86400000`, one day, set in step 6) bounds how much.

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
