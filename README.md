# MRBNN
## Build
You can open the folder with your IDE to do automatic configuration. If you want to do so manually, run commands below:
```bash
mkdir build
# Configure the project.
cmake -S . -B build
# Build the target.
cmake --build build --config Release --target main -j 32
```

Since we use `FetchContent` to make the project self-contained, you're supporsed to be able to access github during configuration.

## Data Download
Will be uploaded after publication. The paths of volumes specified in the configuration file should contain only ASCII characters.

## Special Notes
For C++/CUDA compiler bugs that cause internal compiler error (ICE), we note that:
1. To successfully build ExternalTCNN for the first time, you may need to build it twice since we find that it occasionally encounters wrong compiler flags and ICE. But this seems to only happen for the first time.
2. We have to use C++17 instead of C++20 canonically for tcnn-related code, otherwise ICE occurs in tcnn headers. We also notice possible ABI incompatibility since tcnn library is compiled with `-std=c++14`, but practically it's okay for kindness of vendors. Nevertheless, we isolate them as shared library to minimize such reliance.

For running some targets except for `main` in Windows, you may need to copy ExternalTCNN.dll to the executable directory though they don't really use TCNN, since shared libraries are eagerly loaded when specified in CMake.