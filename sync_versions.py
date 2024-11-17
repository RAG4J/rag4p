import toml

def sync_versions():
    # Load pyproject.toml
    with open("pyproject.toml", "r") as f:
        pyproject = toml.load(f)

    # Load poetry.lock
    with open("poetry.lock", "r") as f:
        lock = toml.load(f)

    # Map dependencies
    dependencies = lock.get("package", [])
    locked_versions = {pkg["name"]: pkg["version"] for pkg in dependencies}

    # Update pyproject.toml
    updated = False
    for dep in pyproject["tool"]["poetry"]["dependencies"]:
        if dep in locked_versions:
            current_version = pyproject["tool"]["poetry"]["dependencies"][dep]
            if current_version != locked_versions[dep]:
                pyproject["tool"]["poetry"]["dependencies"][dep] = locked_versions[dep]
                updated = True

    if updated:
        # Save updated pyproject.toml
        with open("pyproject.toml", "w") as f:
            toml.dump(pyproject, f)
        print("pyproject.toml updated with locked versions.")
    else:
        print("pyproject.toml already matches locked versions.")

if __name__ == "__main__":
    sync_versions()