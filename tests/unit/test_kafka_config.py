import os
import stat

import pytest

from streaming.kafka_config import (
    KafkaTlsMaterial,
    load_tls_material,
    producer_ssl_kwargs,
    spark_ssl_options,
)

MATERIAL = KafkaTlsMaterial(
    ca_cert="-----BEGIN CERTIFICATE-----\nCA\n-----END CERTIFICATE-----",
    service_cert="-----BEGIN CERTIFICATE-----\nSVC\n-----END CERTIFICATE-----",
    service_key="-----BEGIN PRIVATE KEY-----\nKEY\n-----END PRIVATE KEY-----",
)


def test_load_tls_material_reads_the_three_env_vars(monkeypatch):
    monkeypatch.setenv("KAFKA_CA_CERT", "ca")
    monkeypatch.setenv("KAFKA_SERVICE_CERT", "cert")
    monkeypatch.setenv("KAFKA_SERVICE_KEY", "key")

    assert load_tls_material() == KafkaTlsMaterial("ca", "cert", "key")


def test_load_tls_material_fails_loudly_when_a_secret_is_missing(monkeypatch):
    monkeypatch.delenv("KAFKA_SERVICE_KEY", raising=False)
    monkeypatch.setenv("KAFKA_CA_CERT", "ca")
    monkeypatch.setenv("KAFKA_SERVICE_CERT", "cert")

    with pytest.raises(KeyError, match="KAFKA_SERVICE_KEY"):
        load_tls_material()


def test_producer_kwargs_use_ssl_and_write_each_pem_to_its_file(tmp_path):
    kwargs = producer_ssl_kwargs(MATERIAL, directory=str(tmp_path))

    assert kwargs["security_protocol"] == "SSL"
    assert "sasl_mechanism" not in kwargs
    assert open(kwargs["ssl_cafile"]).read().strip() == MATERIAL.ca_cert
    assert open(kwargs["ssl_certfile"]).read().strip() == MATERIAL.service_cert
    assert open(kwargs["ssl_keyfile"]).read().strip() == MATERIAL.service_key


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes are not enforced on Windows")
def test_producer_private_key_file_is_owner_only(tmp_path):
    kwargs = producer_ssl_kwargs(MATERIAL, directory=str(tmp_path))

    mode = stat.S_IMODE(os.stat(kwargs["ssl_keyfile"]).st_mode)
    assert mode == 0o600


def test_spark_options_pass_the_pems_inline_as_ssl_with_pem_stores():
    options = spark_ssl_options(MATERIAL)

    assert options == {
        "kafka.security.protocol": "SSL",
        "kafka.ssl.truststore.type": "PEM",
        "kafka.ssl.truststore.certificates": MATERIAL.ca_cert,
        "kafka.ssl.keystore.type": "PEM",
        "kafka.ssl.keystore.certificate.chain": MATERIAL.service_cert,
        "kafka.ssl.keystore.key": MATERIAL.service_key,
    }
