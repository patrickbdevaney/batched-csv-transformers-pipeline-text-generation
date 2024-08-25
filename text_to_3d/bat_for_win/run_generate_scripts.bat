@echo off
setlocal enabledelayedexpansion

REM Loop over each file in the current directory with the specified pattern
for %%f in (img_desc_input_csv_split_??.csv) do (
    echo Processing %%f
    python img_desc_script_generator.py "%%f"
)

echo All files processed.
pause
