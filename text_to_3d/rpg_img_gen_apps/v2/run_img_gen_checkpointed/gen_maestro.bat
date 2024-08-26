@echo off
setlocal enabledelayedexpansion

REM Loop over each generated script in the current directory
for %%f in (generate_images_split_*.py) do (
    echo Running %%f
    python "%%f"
)

echo All scripts processed.
pause
