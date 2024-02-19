def _get_version() -> str:
    from pathlib import Path

    import versioningit

    import RHEED_camera_script

    RHEED_camera_script_path = Path(RHEED_camera_script.__file__).parent
    return versioningit.get_version(project_dir=RHEED_camera_script_path.parent)


__version__ = _get_version()
