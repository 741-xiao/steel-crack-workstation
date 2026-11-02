"""Windows DLL loading order for the desktop application."""

import subprocess
import sys


def test_gui_import_keeps_torch_loadable() -> None:
    """Importing the GUI must not make the PyTorch runtime fail to load."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import industrial_crack.gui; import torch; print('BOOT_OK')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "BOOT_OK" in result.stdout
