import sys
from pathlib import Path

if __name__ == "__main__":
    # Remove the CWD from sys.path while we load stuff.
    # This is added back by InteractiveShellApp.init_path()
    if sys.path[0] == "" or Path(sys.path[0]) == Path.cwd():
        del sys.path[0]
# C:\Users\subho\Car_Recognition\venv\Lib\site-packages\ipykernel_launcher.py
    from ipykernel import kernelapp as app

    app.launch_new_instance()
