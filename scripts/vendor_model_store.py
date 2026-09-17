"""Copy a pruned MLflow file-store snapshot (one deployed run + its sibling
candidate runs, same experiment) into a committed directory, so the
deployed app can point MLFLOW_TRACKING_URI at it without needing a running
MLflow server or the full local mlruns/ tree. See
docs/superpowers/specs/2026-09-16-deployment-portfolio-design.md, "Model
artifacts"."""

import argparse
import os
import re
import shutil
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

CANDIDATE_RUN_NAMES = ["baseline", "random_forest", "xgboost", "lightgbm"]


def vendor_model_store(run_id: str, tracking_uri: str, output_dir: str) -> list[str]:
    """Copy the deployed run + its candidate sibling runs into `output_dir`,
    preserving MLflow's own `<experiment_id>/<run_id>/...` layout. Returns
    the list of run IDs copied (deployed run first)."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    deployed_run = client.get_run(run_id)
    experiment_id = deployed_run.info.experiment_id

    run_ids = [run_id]
    for name in CANDIDATE_RUN_NAMES:
        matches = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.`mlflow.runName` = '{name}'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if matches and matches[0].info.run_id not in run_ids:
            run_ids.append(matches[0].info.run_id)

    source_root = Path(tracking_uri) / experiment_id
    dest_root = Path(output_dir) / experiment_id
    dest_root.mkdir(parents=True, exist_ok=True)

    experiment_meta = source_root / "meta.yaml"
    shutil.copy2(experiment_meta, dest_root / "meta.yaml")

    for rid in run_ids:
        src = source_root / rid
        dst = dest_root / rid
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # MLflow's newer "Logged Model" registry stores actual model artifacts
    # (MLmodel/model.pkl/...) under <experiment_id>/models/m-<hash>/, not
    # under the run's own artifacts/ folder -- a run's artifacts/ directory
    # only holds plain logged files (CSVs, PNGs, JSON). Each run's
    # outputs/m-<hash>/ subdirectory names which logged models it produced,
    # so mlflow.sklearn.load_model("runs:/<id>/<name>") silently fails to
    # find anything if models/ isn't vendored alongside the run. It's small
    # (whole-experiment total, not per-run), so copy it wholesale rather
    # than parsing each run's outputs/ to select a subset.
    models_src = source_root / "models"
    if models_src.is_dir():
        models_dst = dest_root / "models"
        if models_dst.exists():
            shutil.rmtree(models_dst)
        shutil.copytree(models_src, models_dst)

    _scrub_local_metadata(dest_root)

    return run_ids


def _scrub_local_metadata(dest_root: Path) -> None:
    """Remove the dev machine's OS username and local absolute source path
    from the vendored snapshot -- MLflow records both at logging time
    (`tags/mlflow.user`, `tags/mlflow.source.name`, and `meta.yaml`'s
    `user_id` field) for provenance, but neither is read anywhere in
    `serving/` (only `artifact_uri`/`artifact_location`, repaired separately
    by `serving/ml_loader.py`, matter for loading). This directory is
    committed to a public repo, so it shouldn't carry that local detail."""
    for tags_dir in dest_root.glob("**/tags"):
        for name in ("mlflow.user", "mlflow.source.name"):
            tag_file = tags_dir / name
            if tag_file.exists():
                tag_file.unlink()
    for meta_path in dest_root.glob("**/meta.yaml"):
        _rewrite_meta_field_inline(meta_path, "user_id", "''")


def _rewrite_meta_field_inline(meta_path: Path, field: str, value: str) -> None:
    content = meta_path.read_text()
    repaired = re.sub(rf"^{field}:.*$", f"{field}: {value}", content, count=1, flags=re.MULTILINE)
    if repaired != content:
        meta_path.write_text(repaired)


def main() -> None:
    parser = argparse.ArgumentParser(prog="vendor_model_store")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tracking-uri", default="./mlruns")
    parser.add_argument("--output-dir", default="./deploy/model_store")
    args = parser.parse_args()

    run_ids = vendor_model_store(args.run_id, args.tracking_uri, args.output_dir)
    print(f"Vendored {len(run_ids)} runs into {args.output_dir}: {run_ids}")


if __name__ == "__main__":
    main()
