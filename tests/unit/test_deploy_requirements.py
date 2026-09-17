"""deploy/requirements.txt must be a subset of requirements-dev.txt's
package names (never drift to a different pin), and must not include any
dev-only tooling package."""

from pathlib import Path

DEV_ONLY_PACKAGES = {"jupyter", "nbconvert", "matplotlib", "black", "ruff", "pytest"}


def _package_names(path: str) -> set[str]:
    names = set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("-e") or line.startswith("#"):
            continue
        name = line.split(">=")[0].split("[")[0].strip()
        names.add(name)
    return names


def test_deploy_requirements_excludes_dev_only_packages():
    deploy_packages = _package_names("deploy/requirements.txt")
    assert deploy_packages.isdisjoint(DEV_ONLY_PACKAGES)


def test_deploy_requirements_is_subset_of_dev_requirements():
    dev_packages = _package_names("requirements-dev.txt")
    deploy_packages = _package_names("deploy/requirements.txt")
    assert deploy_packages <= dev_packages
