def _get_version() -> str:
    from pathlib import Path

    import versioningit

    import RHEED camera script

    RHEED camera script_path = Path(RHEED camera script.__file__).parent
    return versioningit.get_version(project_dir=RHEED camera script_path.parent)


__version__ = _get_version()
