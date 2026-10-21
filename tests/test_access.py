"""Access fallbacks and detection, offline (spec, Conventions 5)."""

from __future__ import annotations

import numpy as np
import pytest

import worldarray as wa
from conftest import pattern, write_zarr2, write_zarr2_group
from worldarray import store


@pytest.fixture(autouse=True)
def _forget_admissions(monkeypatch):
    monkeypatch.setattr(store, "_admitted", {})


def test_refusal_moves_on_and_the_way_in_is_remembered():
    specs = [
        {"driver": "s3", "bucket": "b", "aws_credentials": {"type": "anonymous"}},
        {"driver": "s3", "bucket": "b"},
    ]
    calls = []

    def attempt(spec):
        calls.append(spec.get("aws_credentials", {}).get("type", "default"))
        if "aws_credentials" in spec:
            raise RuntimeError("PERMISSION_DENIED: Access denied")
        return "ok"

    assert store.with_access(specs, attempt) == "ok"
    assert calls == ["anonymous", "default"]
    assert store.with_access(specs, attempt) == "ok"
    assert calls[-1] == "default" and len(calls) == 3  # remembered: default tried first, once


def test_other_errors_raise_at_once():
    specs = [
        {"driver": "s3", "bucket": "b", "aws_credentials": {"type": "anonymous"}},
        {"driver": "s3", "bucket": "b"},
    ]

    def attempt(spec):
        raise RuntimeError("NOT_FOUND: no such key")

    with pytest.raises(RuntimeError, match="NOT_FOUND"):
        store.with_access(specs, attempt)


def test_a_refused_probe_key_counts_as_missing_when_another_key_was_read(monkeypatch):
    present = {"zarr.json": b'{"node_type": "array"}'}

    def fake_read_key(spec, key):
        if spec.get("aws_credentials", {}).get("type") != "anonymous":
            raise AssertionError("fell back to credentials for a key that is merely missing")
        if key in present:
            return present[key]
        raise RuntimeError("PERMISSION_DENIED: listing is off, so a missing key is a 403")

    monkeypatch.setattr(store, "_read_key", fake_read_key)
    loc = store.Location("s3://bucket/vol.zarr/s0")
    found = loc.read_many(store.PROBE_KEYS)
    assert found["zarr.json"] == present["zarr.json"]
    assert all(found[k] is None for k in store.PROBE_KEYS if k != "zarr.json")


def test_all_probe_keys_refused_falls_back(monkeypatch):
    def fake_read_key(spec, key):
        if spec.get("aws_credentials", {}).get("type") == "anonymous":
            raise RuntimeError("PERMISSION_DENIED")
        return b"{}" if key == ".zarray" else None

    monkeypatch.setattr(store, "_read_key", fake_read_key)
    found = store.Location("s3://bucket/vol.zarr/s0").read_many(store.PROBE_KEYS)
    assert found[".zarray"] == b"{}" and found["zarr.json"] is None


def test_detection_ignores_the_suffix_and_finds_precomputed_scale_keys(tmp_path):
    misnamed = tmp_path / "actually_zarr.n5"
    write_zarr2_group(misnamed)
    write_zarr2(misnamed / "s0", pattern((2, 2, 2)))
    assert store.detect(str(misnamed / "s0")).format == "zarr2"
    assert store.detect(str(misnamed)).kind == "group"
    with pytest.raises(FileNotFoundError, match="none of"):
        store.detect(str(tmp_path / "nothing_here"))
    assert store.normalize_path("precomputed://gs://bucket/vol/") == "gs://bucket/vol"
    assert store.kvstore_specs("s3://bucket/vol")[0]["aws_credentials"] == {"type": "anonymous"}
    assert [s["driver"] for s in store.kvstore_specs("gs://bucket/vol")] == ["gcs", "http"]


def test_cleaned_zarray_drops_unknown_compressor_members():
    zarray = {"compressor": {"id": "zstd", "level": 3, "checksum": False}, "shape": [1]}
    assert store.cleaned_zarray(zarray)["compressor"] == {"id": "zstd", "level": 3}
    assert store.cleaned_zarray({"compressor": None})["compressor"] is None


def test_contexts_and_retry_timeouts():
    assert wa.shared_context(0) is wa.shared_context(0)
    assert wa.capped_context() is not wa.capped_context()
    assert wa.capped_context_spec(concurrency=2)["s3_request_concurrency"] == {"limit": 2}
    r = wa.Retry()
    assert (
        r.timeout_for(1 << 20, remote=True) == 5.0
        and r.timeout_for(1600 << 20, remote=True) == 100.0
    )
    assert (
        r.timeout_for(1 << 30, remote=False) == 5.0
        and r.timeout_for(16 << 30, remote=False) == 30.0
    )
    assert wa.Retry(timeout=7).timeout_for(10**12, remote=True) == 7.0


def test_metadata_reads_carry_capped_retries():
    spec = store.metadata_spec({"driver": "s3", "bucket": "b", "path": "p/"})
    assert spec["context"]["s3_request_retries"] == store.METADATA_RETRIES
    assert "context" not in store.metadata_spec({"driver": "file", "path": "/x/"})
    assert wa.corner_to_centre((0, 0, 0, 0), (1, 2, 4, 8)) == (0.0, 1.0, 2.0, 4.0)
    assert wa.default_axes(4) == ("c", "z", "y", "x")


def test_s3_endpoint_and_region_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://localhost:5000")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    specs = store.kvstore_specs("s3://bucket/vol")
    assert all(
        s["endpoint"] == "http://localhost:5000" and s["aws_region"] == "us-west-2" for s in specs
    )
    monkeypatch.delenv("AWS_ENDPOINT_URL")
    monkeypatch.delenv("AWS_DEFAULT_REGION")
    assert "endpoint" not in store.kvstore_specs("s3://bucket/vol")[0]


def test_s3_endpoint_service_variable_and_gcs_cache_key(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "http://s3.local:9000")
    assert store.kvstore_specs("s3://bucket/v")[0]["endpoint"] == "http://s3.local:9000"
    store._metadata_kvstore.cache_clear()
    monkeypatch.setenv("TENSORSTORE_GCS_HTTP_URL", "http://a.local")
    spec = '{"driver": "memory"}'
    first = store._metadata_kvstore(spec, "http://a.local")
    assert store._metadata_kvstore(spec, "http://b.local") is not first  # keyed on the variable too


def test_paths_near_the_filesystem_root_do_not_break_the_kvstore():
    assert store.kvstore_specs("/") == [{"driver": "file", "path": "/"}]
    assert store.kvstore_specs("/tmp") == [{"driver": "file", "path": "/tmp/"}]
    assert wa.detect("/", missing_ok=True) is None  # probes the root's keys without raising
    assert store.parent("/tmp") == "/" and store.parent("/") is None


def test_roots_are_never_probed_and_bad_ancestors_do_not_break_an_open(tmp_path, monkeypatch):
    assert (
        store.is_root("/")
        and store.is_root("http://host")
        and store.is_root("s3://bucket")
        and store.is_root(None)
    )
    assert not store.is_root("/tmp") and not store.is_root("http://host/a")
    import importlib

    md = importlib.import_module(
        "worldarray.metadata"
    )  # wa.metadata() the function shadows the module name

    calls = []
    real = md._group_attrs_at

    def flaky(path, fmt):
        calls.append(path)
        if path.endswith("boom"):
            raise RuntimeError("UNKNOWN: [HTTP code 301]")
        return real(path, fmt)

    monkeypatch.setattr(md, "_group_attrs_at", flaky)
    group = tmp_path / "boom" / "plain.zarr"
    write_zarr2(group / "s0", pattern((2, 2, 2)))  # no parent document at all
    assert (
        wa.open(str(group / "s0"), units=None).meta.placed_by is None
    )  # the 301 ancestor ends the walk quietly


def test_list_chunk_keys_needs_no_metadata(cube, tmp_path):
    path, _ = cube
    assert wa.list_chunk_keys(path) == sorted(np.ndindex(2, 2, 2))
    bare = tmp_path / "bare_chunks"
    bare.mkdir()
    for key in ("0.0.0", "0.1.0", "notes.txt"):
        (bare / key).write_bytes(b"x")
    assert wa.list_chunk_keys(str(bare), ndim=3) == [(0, 0, 0), (0, 1, 0)]


def test_capped_context_can_leave_the_network_alone():
    spec = wa.capped_context_spec(concurrency=2, network=False)
    assert "s3_request_concurrency" not in spec and spec["file_io_concurrency"] == {"limit": 2}
    assert "s3_request_concurrency" in wa.capped_context_spec()
