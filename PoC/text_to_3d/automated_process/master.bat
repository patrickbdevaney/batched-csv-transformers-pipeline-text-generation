@echo off
REM Ensure that commands are visible
echo Starting the orchestration process...

REM Check if the generated_images directory exists
if exist "C:\Users\Patrick Devaney\master\generated_images" (
    echo Directory 'generated_images' already exists. Skipping description and image generation.
    goto :skip_description_and_image_generation
)

REM Step 1: Run the description generation script
echo Running description generation script...
python "C:\Users\Patrick Devaney\master\master.py"
if %errorlevel% neq 0 (
    echo Error occurred during description generation.
    exit /b %errorlevel%
)

REM Step 2: Activate the Python virtual environment for image generation and run the image generation script
echo Activating Python virtual environment and running image generation script...
call "C:\Users\Patrick Devaney\flux\Scripts\activate.bat"
python "C:\Users\Patrick Devaney\master\oneshot_v3.py"
if %errorlevel% neq 0 (
    echo Error occurred during image generation.
    exit /b %errorlevel%
)

:skip_description_and_image_generation

REM Step 3: Activate Conda environment and run the mesh generation script
echo Activating Conda environment and running mesh generation script...
call "C:\Users\Patrick Devaney\miniconda3\Scripts\activate.bat" InstantMesh

REM Directory containing the generated images
set "image_dir=C:\Users\Patrick Devaney\master\generated_images"

REM Path to the run.py script and configuration file
set "run_script_path=C:\InstantMesh\appv2.py"
set "config_file=C:\InstantMesh\configs\instant-mesh-large.yaml"

REM Loop over all images in the directory
for %%f in ("%image_dir%\*.png") do (
    echo Processing "%%f"...
    python "%run_script_path%" "%config_file%" "%%f" --save_video
    if %errorlevel% neq 0 (
        echo Error occurred during mesh generation for "%%f".
        exit /b %errorlevel%
    )
    echo Finished processing "%%f"
)

REM Process completed successfully
echo Process completed successfully.
