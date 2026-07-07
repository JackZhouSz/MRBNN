/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "FileDialog.hpp"
#include "Utils.hpp"

#ifdef _WIN32
#include <comdef.h>
#include <shobjidl.h>
#include <windows.h>

#pragma comment(lib, "ole32.lib")

COMContextGuard::COMContextGuard()
{
    if (FAILED(CoInitializeEx(NULL, COINIT_APARTMENTTHREADED)))
        throw std::runtime_error{ "COM initialization fails." };
}
COMContextGuard::~COMContextGuard()
{
    CoUninitialize();
}

static std::optional<std::filesystem::path> OpenDialog(bool folderOnly)
{
    auto deleter = [](auto ptr) { ptr->Release(); };

    IFileOpenDialog *pFolderDialog = nullptr;
    auto hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_INPROC_SERVER,
                               IID_PPV_ARGS(&pFolderDialog));
    if (FAILED(hr))
        return {};

    UniqueResource<IFileOpenDialog *, decltype(deleter)> _1{ pFolderDialog };

    DWORD dwOptions = FOS_PATHMUSTEXIST | FOS_FILEMUSTEXIST;
    if (folderOnly)
        dwOptions |= FOS_PICKFOLDERS;

    hr = pFolderDialog->SetOptions(dwOptions);
    if (FAILED(hr))
        return {};

    hr = pFolderDialog->Show(NULL); // Pass hwnd if needed
    if (FAILED(hr))
        return {};

    IShellItem *pItem = nullptr;
    hr = pFolderDialog->GetResult(&pItem);
    if (FAILED(hr))
        return {};
    UniqueResource<IShellItem *, decltype(deleter)> _2{ pItem };

    PWSTR pszFolderPath = nullptr;
    UniqueResource<PWSTR, Deleter<CoTaskMemFree>> _3{ pszFolderPath };
    hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFolderPath);
    if (FAILED(hr))
        return {};

    return std::filesystem::path{ pszFolderPath };
}

// Though it may be more efficient to initialize once and utilize
// dialog over and over again, such performance is not part of our
// algorithm and thus is not really cared.
std::optional<std::filesystem::path> OpenFileDialog()
{
    return OpenDialog(false);
}

std::optional<std::filesystem::path> OpenFolderDialog()
{
    return OpenDialog(true);
}

#else
// Do nothing in these functions.
COMContextGuard::COMContextGuard() {}
COMContextGuard::~COMContextGuard() {}

std::optional<std::filesystem::path> OpenFileDialog()
{
    return {};
}
std::optional<std::filesystem::path> OpenFolderDialog()
{
    return {};
}
#endif