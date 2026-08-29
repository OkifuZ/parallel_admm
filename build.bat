@echo off
setlocal

REM vcpkg root: use env or default
if "%VCPKG_ROOT%"=="" set VCPKG_ROOT=C:\codebase\vcpkg
echo Using VCPKG_ROOT=%VCPKG_ROOT%

REM 1. Init submodules
echo [1/3] Initializing submodules...
git submodule update --init --recursive
if %ERRORLEVEL% neq 0 (
    echo ERROR: git submodule update failed
    exit /b 1
)

REM 2. Configure
echo [2/3] Configuring CMake...
if not exist build mkdir build
pushd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake -DCMAKE_CUDA_ARCHITECTURES=86
if %ERRORLEVEL% neq 0 (
    echo ERROR: CMake configure failed
    popd
    exit /b 1
)

REM 3. Build
echo [3/3] Building Release...
cmake --build . --config Release --parallel
set BUILD_EXIT=%ERRORLEVEL%
popd

if %BUILD_EXIT% neq 0 (
    echo ERROR: Build failed
    exit /b 1
)

echo.
echo Done. Output: build\Release\main.exe
exit /b 0
