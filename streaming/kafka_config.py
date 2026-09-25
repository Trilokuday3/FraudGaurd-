"""TLS client-certificate (mTLS) settings for the Aiven Kafka service.

The three PEM blobs come from environment variables (GitHub secrets in CI),
never from files in the repo. kafka-python only accepts file paths, so the
producer gets them written to a private temp dir; Spark's Kafka source
accepts inline PEM, so it gets the text directly.
"""

import os
import tempfile
from dataclasses import dataclass


@dataclass(frozen=True)
class KafkaTlsMaterial:
    ca_cert: str
    service_cert: str
    service_key: str


def load_tls_material() -> KafkaTlsMaterial:
    return KafkaTlsMaterial(
        ca_cert=os.environ["KAFKA_CA_CERT"],
        service_cert=os.environ["KAFKA_SERVICE_CERT"],
        service_key=os.environ["KAFKA_SERVICE_KEY"],
    )


def _write_private(directory: str, name: str, content: str) -> str:
    path = os.path.join(directory, name)
    # 0o600 from creation, so the private key is never briefly world-readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(content if content.endswith("\n") else content + "\n")
    return path


def producer_ssl_kwargs(material: KafkaTlsMaterial, directory: str | None = None) -> dict:
    """kafka-python KafkaProducer kwargs for mTLS. Writes the PEMs into
    `directory` (a fresh temp dir when omitted)."""
    directory = directory or tempfile.mkdtemp(prefix="fraudguard-kafka-")
    return {
        "security_protocol": "SSL",
        "ssl_cafile": _write_private(directory, "ca.pem", material.ca_cert),
        "ssl_certfile": _write_private(directory, "service.cert", material.service_cert),
        "ssl_keyfile": _write_private(directory, "service.key", material.service_key),
    }


def spark_ssl_options(material: KafkaTlsMaterial) -> dict[str, str]:
    """Spark Kafka-source options for mTLS, with the PEMs passed inline
    (supported by the Kafka client bundled with Spark 3.5)."""
    return {
        "kafka.security.protocol": "SSL",
        "kafka.ssl.truststore.type": "PEM",
        "kafka.ssl.truststore.certificates": material.ca_cert,
        "kafka.ssl.keystore.type": "PEM",
        "kafka.ssl.keystore.certificate.chain": material.service_cert,
        "kafka.ssl.keystore.key": material.service_key,
    }
