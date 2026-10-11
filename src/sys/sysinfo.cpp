#include "sys/sysinfo.h"

#include <cstdio>
#include <cstring>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(_MSC_VER)
#include <intrin.h>
static void cpuid(unsigned leaf, unsigned r[4]) {
    int x[4];
    __cpuid(x, (int) leaf);
    for (int i = 0; i < 4; i++) {
        r[i] = (unsigned) x[i];
    }
}
#else
#include <cpuid.h>
static void cpuid(unsigned leaf, unsigned r[4]) {
    __cpuid(leaf, r[0], r[1], r[2], r[3]);
}
#endif
static std::string cpu_brand() {
    unsigned r[4];
    cpuid(0x80000000u, r);
    if (r[0] < 0x80000004u) {
        return {};
    }
    char s[49] = {};
    for (unsigned i = 0; i < 3; i++) {
        cpuid(0x80000002u + i, r);
        std::memcpy(s + 16 * i, r, 16);
    }
    std::string b(s);
    b.erase(0, b.find_first_not_of(' '));
    while (!b.empty() && b.back() == ' ') {
        b.pop_back();
    }
    return b;
}
#else
static std::string cpu_brand() {
    return {};
}
#endif

#if defined(_WIN32)

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shellapi.h>
#include <winioctl.h>
#include <shobjidl.h>
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "shell32.lib")

namespace e8::sys {

static std::wstring widen(const std::string & s) {
    if (s.empty()) {
        return {};
    }
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0);
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), w.data(), n);
    return w;
}

static std::string narrow(const wchar_t * w) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) {
        return {};
    }
    std::string s((size_t) n - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

static std::string os_name() {
    using RtlGetVersionFn = LONG(WINAPI *)(PRTL_OSVERSIONINFOW);
    RTL_OSVERSIONINFOW vi{};
    vi.dwOSVersionInfoSize = sizeof(vi);
    HMODULE ntdll          = GetModuleHandleW(L"ntdll.dll");
    auto    fn = ntdll ? reinterpret_cast<RtlGetVersionFn>(reinterpret_cast<void *>(GetProcAddress(ntdll, "RtlGetVersion"))) : nullptr;
    if (!fn || fn(&vi) != 0) {
        return "Windows";
    }
    const char * name = vi.dwMajorVersion == 10 && vi.dwBuildNumber >= 22000 ? "Windows 11" : "Windows";
    return std::string(name) + " build " + std::to_string(vi.dwBuildNumber);
}

static int physical_cores() {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) {
        return 0;
    }
    std::vector<char> buf(len);
    auto * p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, p, &len)) {
        return 0;
    }
    int n = 0;
    for (DWORD off = 0; off < len;) {
        auto * e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data() + off);
        n += e->Relationship == RelationProcessorCore;
        off += e->Size;
    }
    return n;
}

Info query() {
    Info i;
    i.os             = os_name();
    i.cpu            = cpu_brand();
    i.logical_cores  = (int) std::thread::hardware_concurrency();
    i.physical_cores = physical_cores();
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        i.ram_total = ms.ullTotalPhys;
        i.ram_avail = ms.ullAvailPhys;
    }
    return i;
}

int64_t free_space(const std::string & dir) {
    ULARGE_INTEGER avail{};
    if (!GetDiskFreeSpaceExW(widen(dir).c_str(), &avail, nullptr, nullptr)) {
        return -1;
    }
    return (int64_t) avail.QuadPart;
}

static const char * bus_name(STORAGE_BUS_TYPE t) {
    switch (t) {
        case BusTypeNvme:
            return "NVMe";
        case BusTypeSata:
            return "SATA";
        case BusTypeUsb:
            return "USB";
        case BusTypeSas:
            return "SAS";
        case BusTypeRAID:
            return "RAID";
        case BusTypeSpaces:
            return "Storage Spaces";
        default:
            return "other bus";
    }
}

std::string describe_drive(const std::string & path) {
    wchar_t root[MAX_PATH];
    if (!GetVolumePathNameW(widen(path).c_str(), root, MAX_PATH)) {
        return {};
    }
    std::wstring vol = root;  // "D:\"
    if (vol.size() < 2 || vol[1] != L':') {
        return {};
    }
    const std::wstring dev = L"\\\\.\\" + vol.substr(0, 2);
    HANDLE h = CreateFileW(dev.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return {};
    }
    STORAGE_PROPERTY_QUERY q{};
    q.PropertyId = StorageDeviceProperty;
    q.QueryType  = PropertyStandardQuery;
    alignas(8) char buf[1024] = {};
    DWORD           got       = 0;
    std::string     out;
    if (DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, &q, sizeof(q), buf, sizeof(buf), &got, nullptr) &&
        got >= sizeof(STORAGE_DEVICE_DESCRIPTOR)) {
        const auto * d = reinterpret_cast<const STORAGE_DEVICE_DESCRIPTOR *>(buf);
        if (d->ProductIdOffset && d->ProductIdOffset < got) {
            out = buf + d->ProductIdOffset;
            while (!out.empty() && out.back() == ' ') {
                out.pop_back();
            }
        }
        out += std::string(out.empty() ? "" : " ") + "[" + bus_name(d->BusType) + "]";
    }
    CloseHandle(h);
    return out;
}

std::vector<DriveTemp> drive_temps() {
    std::vector<DriveTemp> out;
    for (int i = 0; i < 32; i++) {
        const std::wstring dev = L"\\\\.\\PhysicalDrive" + std::to_wstring(i);
        HANDLE h = CreateFileW(dev.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            continue;
        }
        STORAGE_PROPERTY_QUERY q{};
        q.QueryType = PropertyStandardQuery;
        alignas(8) char buf[1024] = {};
        DWORD           got       = 0;
        DriveTemp       d;
        q.PropertyId = StorageDeviceTemperatureProperty;
        const bool has_temp = DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, &q, sizeof(q), buf, sizeof(buf), &got, nullptr) &&
                              got >= sizeof(STORAGE_TEMPERATURE_DATA_DESCRIPTOR) &&
                              reinterpret_cast<const STORAGE_TEMPERATURE_DATA_DESCRIPTOR *>(buf)->InfoCount > 0;
        if (has_temp) {
            d.temp_c = reinterpret_cast<const STORAGE_TEMPERATURE_DATA_DESCRIPTOR *>(buf)->TemperatureInfo[0].Temperature;
            q.PropertyId = StorageDeviceProperty;
            std::memset(buf, 0, sizeof(buf));
            if (DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, &q, sizeof(q), buf, sizeof(buf), &got, nullptr) &&
                got >= sizeof(STORAGE_DEVICE_DESCRIPTOR)) {
                const auto * dd = reinterpret_cast<const STORAGE_DEVICE_DESCRIPTOR *>(buf);
                if (dd->ProductIdOffset && dd->ProductIdOffset < got) {
                    d.name = buf + dd->ProductIdOffset;
                    while (!d.name.empty() && d.name.back() == ' ') {
                        d.name.pop_back();
                    }
                }
                d.name += std::string(d.name.empty() ? "" : " ") + "[" + bus_name(dd->BusType) + "]";
            }
            out.push_back(d);
        }
        CloseHandle(h);
    }
    return out;
}

// NVML from the driver (System32\nvml.dll). A driver update or reset invalidates it under us (calls then fault inside
// nvml.dll), so every call is guarded: after a fault the library is dropped and loaded again 30 s later, which picks
// up the new driver's copy.
namespace {
using nvml_init_t  = int (*)();
using nvml_dev_t   = int (*)(unsigned, void **);
using nvml_temp_t  = int (*)(void *, int, unsigned *);
using nvml_power_t = int (*)(void *, unsigned *);
struct NvmlUtil {
    unsigned gpu, memory;
};
using nvml_util_t = int (*)(void *, NvmlUtil *);
struct NvmlMem {
    unsigned long long total, free, used;
};
using nvml_mem_t  = int (*)(void *, NvmlMem *);
using nvml_pcie_t = int (*)(void *, int, unsigned *);
struct Nvml {
    HMODULE      lib   = nullptr;
    void *       dev   = nullptr;
    nvml_temp_t  temp  = nullptr;
    nvml_power_t power = nullptr;
    nvml_util_t  util  = nullptr;
    nvml_mem_t   mem   = nullptr;
    nvml_pcie_t  pcie  = nullptr;
    ULONGLONG    retry = 0;  // GetTickCount64 time of the next load attempt
};

// 1 = opened, 0 = no NVIDIA driver, -1 = faulted (plain C: an SEH frame needs a function without destructors)
int nvml_open(Nvml * n) {
#ifdef _MSC_VER
    __try {
#endif
        n->lib = LoadLibraryW(L"nvml.dll");
        if (!n->lib) return 0;
        auto init = reinterpret_cast<nvml_init_t>(GetProcAddress(n->lib, "nvmlInit_v2"));
        auto get  = reinterpret_cast<nvml_dev_t>(GetProcAddress(n->lib, "nvmlDeviceGetHandleByIndex_v2"));
        n->temp   = reinterpret_cast<nvml_temp_t>(GetProcAddress(n->lib, "nvmlDeviceGetTemperature"));
        n->power  = reinterpret_cast<nvml_power_t>(GetProcAddress(n->lib, "nvmlDeviceGetPowerUsage"));
        n->util   = reinterpret_cast<nvml_util_t>(GetProcAddress(n->lib, "nvmlDeviceGetUtilizationRates"));
        n->mem    = reinterpret_cast<nvml_mem_t>(GetProcAddress(n->lib, "nvmlDeviceGetMemoryInfo"));
        n->pcie   = reinterpret_cast<nvml_pcie_t>(GetProcAddress(n->lib, "nvmlDeviceGetPcieThroughput"));
        if (!init || !get || init() != 0 || get(0, &n->dev) != 0) {
            n->dev = nullptr;
            return 0;
        }
        return 1;
#ifdef _MSC_VER
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        n->dev = nullptr;
        return -1;
    }
#endif
}

int nvml_read(Nvml * n, GpuSensors * s) {
#ifdef _MSC_VER
    __try {
#endif
        unsigned v = 0;
        if (n->temp && n->temp(n->dev, 0 /* NVML_TEMPERATURE_GPU */, &v) == 0) s->temp_c = (int) v;
        if (n->power && n->power(n->dev, &v) == 0) s->power_w = v / 1000.0;
        NvmlUtil u{};
        if (n->util && n->util(n->dev, &u) == 0) s->mem_util = (int) u.memory;
        NvmlMem m{};
        if (n->mem && n->mem(n->dev, &m) == 0) {
            s->vram_total = m.total;
            s->vram_used  = m.used;
        }
        if (n->pcie && n->pcie(n->dev, 1 /* NVML_PCIE_UTIL_RX_BYTES */, &v) == 0) s->pcie_rx_gbs = v * 1024.0 / 1e9;  // KB/s
        if (n->pcie && n->pcie(n->dev, 0 /* NVML_PCIE_UTIL_TX_BYTES */, &v) == 0) s->pcie_tx_gbs = v * 1024.0 / 1e9;
        return 1;
#ifdef _MSC_VER
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return -1;
    }
#endif
}
} // namespace

GpuSensors gpu_sensors() {
    static Nvml n;
    static bool loaded = false;
    GpuSensors  s;
    if (!loaded) {
        if (GetTickCount64() < n.retry) return s;
        loaded = true;
        if (nvml_open(&n) < 0) {
            fprintf(stderr, "nvml: faulted while loading (driver changing?); retrying in 30 s\n");
            loaded = false;
            n       = Nvml{};
            n.retry = GetTickCount64() + 30000;
            return s;
        }
    }
    if (!n.dev) return s;
    if (nvml_read(&n, &s) < 0) {  // the driver went away: forget this copy of the library and load it again later
        fprintf(stderr, "nvml: faulted (driver update or reset?); reloading in 30 s\n");
        if (n.lib) FreeLibrary(n.lib);
        loaded = false;
        n       = Nvml{};
        n.retry = GetTickCount64() + 30000;
        return GpuSensors{};
    }
    s.ok = true;
    return s;
}

std::vector<BoardSensor> board_sensors() {
    return {};
}

std::vector<std::string> fixed_drives() {
    std::vector<std::string> out;
    wchar_t                  buf[512];
    const DWORD              n = GetLogicalDriveStringsW(512, buf);
    for (DWORD i = 0; i < n && buf[i];) {
        const wchar_t * root = buf + i;
        if (GetDriveTypeW(root) == DRIVE_FIXED) {
            out.push_back(narrow(root));
        }
        i += (DWORD) wcslen(root) + 1;
    }
    return out;
}

int run_process(const std::vector<std::string> & args, const std::string & log_path, const std::atomic<bool> & cancel,
                std::string & err) {
    if (args.empty()) {
        err = "no program";
        return -1;
    }
    // command line with the usual quoting rules (backslashes before a quote doubled)
    std::wstring cmd;
    for (const auto & a : args) {
        const std::wstring w = widen(a);
        if (!cmd.empty()) cmd += L' ';
        if (!w.empty() && w.find_first_of(L" \t\"") == std::wstring::npos) {
            cmd += w;
            continue;
        }
        cmd += L'"';
        size_t bs = 0;
        for (wchar_t c : w) {
            if (c == L'\\') { bs++; continue; }
            cmd.append(c == L'"' ? bs * 2 + 1 : bs, L'\\');
            bs = 0;
            cmd += c;
        }
        cmd.append(bs * 2, L'\\');
        cmd += L'"';
    }
    SECURITY_ATTRIBUTES sa{ sizeof(sa), nullptr, TRUE };
    HANDLE log = CreateFileW(widen(log_path).c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL, nullptr);
    if (log == INVALID_HANDLE_VALUE) {
        err = "cannot open the log " + log_path;
        return -1;
    }
    STARTUPINFOW si{};
    si.cb         = sizeof(si);
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdInput  = nullptr;
    si.hStdOutput = log;
    si.hStdError  = log;
    PROCESS_INFORMATION pi{};
    std::vector<wchar_t> buf(cmd.begin(), cmd.end());
    buf.push_back(0);
    const BOOL ok = CreateProcessW(nullptr, buf.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    CloseHandle(log);
    if (!ok) {
        err = "cannot start " + args[0] + " (error " + std::to_string(GetLastError()) + ")";
        return -1;
    }
    CloseHandle(pi.hThread);
    int code = 0;
    for (;;) {
        if (WaitForSingleObject(pi.hProcess, 200) == WAIT_OBJECT_0) {
            DWORD c = 0;
            GetExitCodeProcess(pi.hProcess, &c);
            code = (int) c;
            break;
        }
        if (cancel.load()) {
            TerminateProcess(pi.hProcess, 1);
            WaitForSingleObject(pi.hProcess, 5000);
            code = -2;
            break;
        }
    }
    CloseHandle(pi.hProcess);
    return code;
}

bool wait_pid(int pid, int ms) {
    HANDLE h = OpenProcess(SYNCHRONIZE, FALSE, (DWORD) pid);
    if (!h) return true;  // gone already
    const bool done = WaitForSingleObject(h, (DWORD) ms) == WAIT_OBJECT_0;
    CloseHandle(h);
    return done;
}

bool spawn_detached(const std::vector<std::string> & args, std::string & err) {
    std::wstring cmd;
    for (const auto & a : args) {  // same quoting as run_process
        const std::wstring w = widen(a);
        if (!cmd.empty()) cmd += L' ';
        if (!w.empty() && w.find_first_of(L" \t\"") == std::wstring::npos) {
            cmd += w;
            continue;
        }
        cmd += L'"';
        size_t bs = 0;
        for (wchar_t c : w) {
            if (c == L'\\') { bs++; continue; }
            cmd.append(c == L'"' ? bs * 2 + 1 : bs, L'\\');
            bs = 0;
            cmd += c;
        }
        cmd.append(bs * 2, L'\\');
        cmd += L'"';
    }
    STARTUPINFOW si{};
    si.cb         = sizeof(si);
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
    for (HANDLE h : { si.hStdOutput, si.hStdError })  // the log handles must be inheritable for the child to share them
        if (h && h != INVALID_HANDLE_VALUE) SetHandleInformation(h, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);
    PROCESS_INFORMATION pi{};
    std::vector<wchar_t> buf(cmd.begin(), cmd.end());
    buf.push_back(0);
    if (!CreateProcessW(nullptr, buf.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP, nullptr, nullptr, &si, &pi)) {
        err = "cannot start " + args[0] + " (error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return true;
}

int self_pid() {
    return (int) GetCurrentProcessId();
}

CpuSample cpu_sample() {
    auto u = [](const FILETIME & f) { return ((uint64_t) f.dwHighDateTime << 32) | f.dwLowDateTime; };
    FILETIME  idle, kern, user, c, e, pk, pu;
    CpuSample s;
    if (GetSystemTimes(&idle, &kern, &user)) {  // kernel time includes idle time
        s.total = u(kern) + u(user);
        s.busy  = s.total - u(idle);
    }
    if (GetProcessTimes(GetCurrentProcess(), &c, &e, &pk, &pu)) s.self = u(pk) + u(pu);
    return s;
}

std::string pick_path(bool folder, const std::string & title, const std::string & start, const std::string & pattern) {
    std::string out;
    // COM wants a single-threaded apartment of its own: run the dialog on a fresh thread
    std::thread([&] {
        if (FAILED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE))) return;
        IFileOpenDialog * dlg = nullptr;
        if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dlg)))) {
            DWORD opt = 0;
            dlg->GetOptions(&opt);
            dlg->SetOptions(opt | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST | (folder ? FOS_PICKFOLDERS : FOS_FILEMUSTEXIST));
            if (!title.empty()) dlg->SetTitle(widen(title).c_str());
            const std::wstring pat = widen(pattern);
            if (!folder && !pat.empty()) {
                COMDLG_FILTERSPEC spec[] = { { pat.c_str(), pat.c_str() }, { L"All files", L"*.*" } };
                dlg->SetFileTypes(2, spec);
            }
            IShellItem * dir = nullptr;
            if (!start.empty() && SUCCEEDED(SHCreateItemFromParsingName(widen(start).c_str(), nullptr, IID_PPV_ARGS(&dir)))) {
                dlg->SetFolder(dir);
                dir->Release();
            }
            // an invisible topmost owner puts the dialog in front of the browser
            HWND owner = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW, L"STATIC", L"", WS_POPUP, 0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
            if (owner) {
                ShowWindow(owner, SW_SHOWNOACTIVATE);
                SetForegroundWindow(owner);
            }
            if (SUCCEEDED(dlg->Show(owner))) {
                IShellItem * item = nullptr;
                if (SUCCEEDED(dlg->GetResult(&item))) {
                    PWSTR p = nullptr;
                    if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &p))) {
                        out = narrow(p);
                        CoTaskMemFree(p);
                    }
                    item->Release();
                }
            }
            if (owner) DestroyWindow(owner);
            dlg->Release();
        }
        CoUninitialize();
    }).join();
    return out;
}

std::string self_exe() {
    wchar_t p[MAX_PATH * 4];
    const DWORD n = GetModuleFileNameW(nullptr, p, (DWORD) (sizeof(p) / sizeof(p[0])));
    return n ? narrow(p) : std::string();
}

void set_thread_high_perf() {
#if defined(THREAD_POWER_THROTTLING_CURRENT_VERSION)
    THREAD_POWER_THROTTLING_STATE t{};
    t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
    t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
    t.StateMask   = 0;
    SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, sizeof(t));
#endif
}

std::vector<std::string> utf8_args(int argc, char ** argv) {
    int        n  = 0;
    LPWSTR *   wa = CommandLineToArgvW(GetCommandLineW(), &n);
    std::vector<std::string> out;
    if (!wa) {
        out.assign(argv, argv + argc);
        return out;
    }
    for (int i = 0; i < n; i++) {
        out.push_back(narrow(wa[i]));
    }
    LocalFree(wa);
    return out;
}

std::string join_path(const std::string & dir, const std::string & name) {
    if (!dir.empty() && (dir.back() == '\\' || dir.back() == '/')) {
        return dir + name;
    }
    return dir + "\\" + name;
}

bool is_directory(const std::string & path) {
    const DWORD a = GetFileAttributesW(widen(path).c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}

bool remove_file(const std::string & path) {
    return DeleteFileW(widen(path).c_str()) != 0;
}

bool make_dir(const std::string & path) {
    return CreateDirectoryW(widen(path).c_str(), nullptr) != 0 || GetLastError() == ERROR_ALREADY_EXISTS;
}

bool remove_dir(const std::string & path) {
    return RemoveDirectoryW(widen(path).c_str()) != 0;
}

} // namespace e8::sys

#else  // POSIX

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <set>
#include <signal.h>
#include <spawn.h>
#include <sstream>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/wait.h>
#include <sys/statvfs.h>
#include <sys/utsname.h>
#include <unistd.h>

extern char ** environ;

namespace e8::sys {

static uint64_t meminfo_kb(const char * key) {
    std::ifstream f("/proc/meminfo");
    std::string   k;
    uint64_t      v = 0;
    std::string   unit;
    while (f >> k >> v >> unit) {
        if (k == key) {
            return v;
        }
    }
    return 0;
}

static int physical_cores() {
    std::ifstream                       f("/proc/cpuinfo");
    std::string                         line;
    std::string                         phys;
    std::set<std::pair<std::string, std::string>> cores;
    while (std::getline(f, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const std::string key = line.substr(0, line.find_last_not_of(" \t", colon - 1) + 1);
        const std::string val = colon + 2 <= line.size() ? line.substr(colon + 2) : "";
        if (key == "physical id") {
            phys = val;
        } else if (key == "core id") {
            cores.insert({ phys, val });
        }
    }
    return (int) cores.size();
}

Info query() {
    Info           i;
    struct utsname u {};
    i.os             = uname(&u) == 0 ? std::string(u.sysname) + " " + u.release : "POSIX";
    i.cpu            = cpu_brand();
    i.logical_cores  = (int) std::thread::hardware_concurrency();
    i.physical_cores = physical_cores();
    if (i.physical_cores == 0) {
        i.physical_cores = i.logical_cores;
    }
    i.ram_total = meminfo_kb("MemTotal:") * 1024;
    i.ram_avail = meminfo_kb("MemAvailable:") * 1024;
    return i;
}

int64_t free_space(const std::string & dir) {
    struct statvfs s {};
    if (statvfs(dir.c_str(), &s) != 0) {
        return -1;
    }
    return (int64_t) s.f_bavail * (int64_t) s.f_frsize;
}

static std::string read_line(const std::string & path) {
    std::ifstream f(path);
    std::string   l;
    std::getline(f, l);
    while (!l.empty() && (l.back() == ' ' || l.back() == '\n')) l.pop_back();
    return l;
}

// whole disk under /sys/devices for a /sys/dev/block/M:m or /sys/block/X path (partitions map to their disk)
static std::string sys_disk(const std::string & link) {
    char buf[PATH_MAX];
    if (!realpath(link.c_str(), buf)) return {};
    std::string dev = buf;
    if (access((dev + "/partition").c_str(), F_OK) == 0) dev = dev.substr(0, dev.rfind('/'));
    return dev;
}

static std::string describe_sys_disk(const std::string & dev, int depth = 0) {
    std::string out = read_line(dev + "/device/model");
    // device-mapper (LVM, LUKS) and md RAID have no model: describe the first underlying disk
    if (out.empty() && depth < 4) {
        if (DIR * d = opendir((dev + "/slaves").c_str())) {
            std::string under;
            for (dirent * e; (e = readdir(d));) {
                if (e->d_name[0] != '.') {
                    under = sys_disk(dev + "/slaves/" + e->d_name);
                    break;
                }
            }
            closedir(d);
            if (!under.empty()) return describe_sys_disk(under, depth + 1);
        }
    }
    const std::string name = dev.substr(dev.rfind('/') + 1);
    std::string       bus;
    if (dev.find("/usb") != std::string::npos) bus = "USB";
    else if (name.rfind("nvme", 0) == 0) bus = "NVMe";
    else if (name.rfind("sd", 0) == 0) bus = "SATA";
    else if (name.rfind("vd", 0) == 0) bus = "Virtio";
    else if (name.rfind("mmcblk", 0) == 0) bus = "SD";
    if (!bus.empty()) out += std::string(out.empty() ? "" : " ") + "[" + bus + "]";
    return out;
}

std::string describe_drive(const std::string & path) {
    struct stat st {};
    if (stat(path.c_str(), &st) != 0) return {};
    const std::string dev =
        sys_disk("/sys/dev/block/" + std::to_string(major(st.st_dev)) + ":" + std::to_string(minor(st.st_dev)));
    return dev.empty() ? std::string() : describe_sys_disk(dev);  // empty: tmpfs, overlay, network file systems
}

std::vector<std::string> fixed_drives() {
    // mount points of local disks, one per device (the shortest path), without boot partitions, loops and snaps
    std::ifstream                                    f("/proc/mounts");
    std::string                                      dev, mnt, fs, rest;
    std::vector<std::pair<std::string, std::string>> seen;  // device, mount point
    while (f >> dev >> mnt >> fs && std::getline(f, rest)) {
        if (dev.rfind("/dev/", 0) != 0 || dev.rfind("/dev/loop", 0) == 0) continue;
        if (fs == "vfat" || fs == "squashfs" || mnt.rfind("/boot", 0) == 0 || mnt.rfind("/snap", 0) == 0) continue;
        std::string m;  // /proc/mounts escapes spaces and the like as \040
        for (size_t i = 0; i < mnt.size(); i++) {
            if (mnt[i] == '\\' && i + 3 < mnt.size()) {
                m += (char) std::strtol(mnt.substr(i + 1, 3).c_str(), nullptr, 8);
                i += 3;
            } else {
                m += mnt[i];
            }
        }
        auto it = std::find_if(seen.begin(), seen.end(), [&](const auto & x) { return x.first == dev; });
        if (it == seen.end()) seen.push_back({ dev, m });
        else if (m.size() < it->second.size()) it->second = m;
    }
    std::vector<std::string> out;
    for (const auto & x : seen) out.push_back(x.second);
    return out;
}

std::vector<DriveTemp> drive_temps() {
    // hwmon devices of NVMe controllers ("nvme") and SATA disks (the "drivetemp" module); no root needed
    std::vector<DriveTemp> out;
    DIR *                  d = opendir("/sys/class/hwmon");
    if (!d) return out;
    for (dirent * e; (e = readdir(d));) {
        if (e->d_name[0] == '.') continue;
        const std::string h    = std::string("/sys/class/hwmon/") + e->d_name;
        const std::string name = read_line(h + "/name");
        if (name != "nvme" && name != "drivetemp") continue;
        const std::string t = read_line(h + "/temp1_input");
        if (t.empty()) continue;
        DriveTemp dt;
        dt.temp_c = std::atoi(t.c_str()) / 1000;
        dt.name   = read_line(h + "/device/model");
        dt.name += std::string(dt.name.empty() ? "" : " ") + (name == "nvme" ? "[NVMe]" : "[SATA]");
        out.push_back(dt);
    }
    closedir(d);
    std::sort(out.begin(), out.end(), [](const DriveTemp & a, const DriveTemp & b) { return a.name < b.name; });
    return out;
}

std::vector<BoardSensor> board_sensors() {
    std::vector<BoardSensor> out;
    DIR *                    d = opendir("/sys/class/hwmon");
    if (!d) return out;
    std::vector<std::string> dirs;
    for (dirent * e; (e = readdir(d));)
        if (e->d_name[0] != '.') dirs.push_back(std::string("/sys/class/hwmon/") + e->d_name);
    closedir(d);
    std::sort(dirs.begin(), dirs.end());
    int dimm = 0;
    for (const auto & h : dirs) {
        const std::string drv = read_line(h + "/name");
        std::string       kind;
        if (drv == "k10temp" || drv == "zenpower") kind = "amdcpu";
        else if (drv == "coretemp") kind = "intelcpu";
        else if (drv == "spd5118" || drv == "jc42") kind = "memory";
        else if (drv == "amdgpu") kind = "gpu-amd";
        else if (drv.rfind("nct", 0) == 0 || drv.rfind("it8", 0) == 0 || drv.rfind("f718", 0) == 0 || drv.rfind("w83", 0) == 0 ||
                 drv == "asus_ec_sensors" || drv == "gigabyte_wmi" || drv == "asus_wmi_sensors")
            kind = "lpc";
        else continue;
        const std::string hw = kind == "lpc" ? drv : read_line(h + "/device/model").empty() ? drv : read_line(h + "/device/model");
        for (const char * type : { "temp", "in" }) {
            for (int i = 0; i < 32; i++) {
                const std::string base = h + "/" + type + std::to_string(i);
                const std::string v    = read_line(base + "_input");
                if (v.empty()) continue;
                std::string label = read_line(base + "_label");
                const bool  volt  = type[0] == 'i';
                if (kind == "memory") label = "DIMM #" + std::to_string(++dimm);
                else if (label.empty()) {
                    if (kind == "lpc" && volt) continue;  // unlabeled board voltages: which rail is which depends on the board
                    label = kind == "amdcpu" && !volt ? "Tctl" : type + std::to_string(i);
                }
                // LibreHardwareMonitor's names, which the Sensors page looks for
                if (label.rfind("Tccd", 0) == 0) label = "CCD" + label.substr(4);
                if (kind == "lpc" && (label == "SYSTIN" || label == "Motherboard" || label == "SYSTEM")) label = "Motherboard";
                BoardSensor b;
                b.kind  = kind;
                b.hw    = hw;
                b.name  = label;
                b.volt  = volt;
                b.value = std::atof(v.c_str()) / 1000.0;  // millidegrees / millivolts
                out.push_back(b);
            }
        }
    }
    return out;
}

// NVML from the driver (libnvidia-ml.so.1), loaded at run time; retried every 30 s while there is none
namespace {
struct NvmlUtil {
    unsigned gpu, memory;
};
struct NvmlMem {
    unsigned long long total, free, used;
};
struct Nvml {
    void * lib = nullptr;
    void * dev = nullptr;
    int (*temp)(void *, int, unsigned *) = nullptr;
    int (*power)(void *, unsigned *)     = nullptr;
    int (*util)(void *, NvmlUtil *)      = nullptr;
    int (*mem)(void *, NvmlMem *)        = nullptr;
    int (*pcie)(void *, int, unsigned *) = nullptr;
    std::chrono::steady_clock::time_point retry{};
};
template <class F> void sym(void * lib, const char * name, F & f) {
    f = reinterpret_cast<F>(dlsym(lib, name));
}
} // namespace

static GpuSensors nvml_sensors() {
    static std::mutex           mu;
    static Nvml                 n;
    std::lock_guard<std::mutex> lk(mu);
    GpuSensors                  s;
    if (!n.dev) {
        const auto now = std::chrono::steady_clock::now();
        if (now < n.retry) return s;
        n.retry = now + std::chrono::seconds(30);
        if (!n.lib) n.lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!n.lib) return s;
        int (*init)()                 = nullptr;
        int (*get)(unsigned, void **) = nullptr;
        sym(n.lib, "nvmlInit_v2", init);
        sym(n.lib, "nvmlDeviceGetHandleByIndex_v2", get);
        sym(n.lib, "nvmlDeviceGetTemperature", n.temp);
        sym(n.lib, "nvmlDeviceGetPowerUsage", n.power);
        sym(n.lib, "nvmlDeviceGetUtilizationRates", n.util);
        sym(n.lib, "nvmlDeviceGetMemoryInfo", n.mem);
        sym(n.lib, "nvmlDeviceGetPcieThroughput", n.pcie);
        if (!init || !get || init() != 0 || get(0, &n.dev) != 0) {
            n.dev = nullptr;
            return s;
        }
    }
    unsigned v = 0;
    if (n.temp && n.temp(n.dev, 0 /* NVML_TEMPERATURE_GPU */, &v) == 0) s.temp_c = (int) v;
    if (n.power && n.power(n.dev, &v) == 0) s.power_w = v / 1000.0;
    NvmlUtil u{};
    if (n.util && n.util(n.dev, &u) == 0) s.mem_util = (int) u.memory;
    NvmlMem m{};
    if (n.mem && n.mem(n.dev, &m) == 0) {
        s.vram_total = m.total;
        s.vram_used  = m.used;
    }
    if (n.pcie && n.pcie(n.dev, 1 /* NVML_PCIE_UTIL_RX_BYTES */, &v) == 0) s.pcie_rx_gbs = v * 1024.0 / 1e9;  // KB/s
    if (n.pcie && n.pcie(n.dev, 0 /* NVML_PCIE_UTIL_TX_BYTES */, &v) == 0) s.pcie_tx_gbs = v * 1024.0 / 1e9;
    s.ok = true;
    return s;
}

// AMD GPU (the first one this process can use) from the amdgpu driver's sysfs (hwmon temperature and power, VRAM and memory load): no ROCm library and no
// root, and the same files on every ROCm version (libamd_smi's structs change between releases)
static GpuSensors amdgpu_sensors() {
    static std::string dev;  // /sys/class/drm/cardN/device of the first AMD GPU, found once
    static bool        looked = false;
    GpuSensors         s;
    if (!looked) {
        looked = true;
        if (DIR * d = opendir("/sys/class/drm")) {
            std::vector<std::string> cards;
            for (dirent * e; (e = readdir(d));) {
                const std::string n = e->d_name;
                if (n.rfind("card", 0) == 0 && n.find('-') == std::string::npos) cards.push_back(n);
            }
            closedir(d);
            std::sort(cards.begin(), cards.end(), [](const std::string & a, const std::string & b) {
                return std::atoi(a.c_str() + 4) < std::atoi(b.c_str() + 4);  // card2 before card10
            });
            // a container sees every card of the host in /sys but gets only its own GPU's render node in /dev/dri:
            // take the first AMD card whose render node this process can open (on a desktop: the first AMD card)
            std::string first;
            for (const auto & c : cards) {
                const std::string p = "/sys/class/drm/" + c + "/device";
                if (read_line(p + "/vendor") != "0x1002" || read_line(p + "/mem_info_vram_total").empty()) continue;
                if (first.empty()) first = p;
                bool mine = false;
                if (DIR * r = opendir((p + "/drm").c_str())) {
                    for (dirent * e; (e = readdir(r));) {
                        const std::string n = e->d_name;
                        if (n.rfind("renderD", 0) == 0 && access(("/dev/dri/" + n).c_str(), R_OK | W_OK) == 0) mine = true;
                    }
                    closedir(r);
                }
                if (mine) {
                    dev = p;
                    break;
                }
            }
            if (dev.empty()) dev = first;
        }
    }
    if (dev.empty()) return s;
    s.vram_total = std::strtoull(read_line(dev + "/mem_info_vram_total").c_str(), nullptr, 10);
    s.vram_used  = std::strtoull(read_line(dev + "/mem_info_vram_used").c_str(), nullptr, 10);
    const std::string busy = read_line(dev + "/mem_busy_percent");
    if (!busy.empty()) s.mem_util = std::atoi(busy.c_str());
    if (DIR * d = opendir((dev + "/hwmon").c_str())) {
        for (dirent * e; (e = readdir(d));) {
            if (e->d_name[0] == '.') continue;
            const std::string h = dev + "/hwmon/" + e->d_name;
            // temp2 is the junction (hotspot) where present, temp1 the edge
            std::string t = read_line(h + "/temp2_input");
            if (t.empty()) t = read_line(h + "/temp1_input");
            if (!t.empty()) s.temp_c = std::atoi(t.c_str()) / 1000;
            std::string w = read_line(h + "/power1_average");
            if (w.empty()) w = read_line(h + "/power1_input");
            if (!w.empty()) s.power_w = std::atof(w.c_str()) / 1e6;  // microwatts
            break;
        }
        closedir(d);
    }
    s.ok = s.vram_total > 0;
    return s;
}

GpuSensors gpu_sensors() {
    GpuSensors s = nvml_sensors();
    return s.ok ? s : amdgpu_sensors();
}

// argv for posix_spawn (pointers into `args`)
static std::vector<char *> argv_of(const std::vector<std::string> & args) {
    std::vector<char *> v;
    for (const auto & a : args) v.push_back(const_cast<char *>(a.c_str()));
    v.push_back(nullptr);
    return v;
}

int run_process(const std::vector<std::string> & args, const std::string & log_path, const std::atomic<bool> & cancel,
                std::string & err) {
    if (args.empty()) {
        err = "no program";
        return -1;
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 1, log_path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    posix_spawn_file_actions_adddup2(&fa, 1, 2);
    posix_spawnattr_t at;
    posix_spawnattr_init(&at);
    posix_spawnattr_setflags(&at, POSIX_SPAWN_SETPGROUP);  // a group of its own: cancelling signals the whole job
    posix_spawnattr_setpgroup(&at, 0);
    pid_t     pid = 0;
    auto      av  = argv_of(args);
    const int rc  = posix_spawnp(&pid, args[0].c_str(), &fa, &at, av.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    posix_spawnattr_destroy(&at);
    if (rc != 0) {
        err = "cannot start " + args[0] + " (" + std::strerror(rc) + ")";
        return -1;
    }
    int status = 0;
    for (;;) {
        const pid_t r = waitpid(pid, &status, WNOHANG);
        if (r == pid) return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
        if (r < 0 && errno != EINTR) return -1;
        if (cancel.load()) {  // SIGTERM, then SIGKILL after 5 s
            kill(-pid, SIGTERM);
            for (int i = 0; i < 50 && waitpid(pid, &status, WNOHANG) == 0; i++) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (kill(pid, 0) == 0) {
                kill(-pid, SIGKILL);
                waitpid(pid, &status, 0);
            }
            return -2;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

std::string self_exe() {
    char          buf[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    return n > 0 ? std::string(buf, (size_t) n) : std::string();
}

// Runs a program and returns its stdout without the trailing newline; empty when it fails or exits non-zero
static std::string capture(const std::vector<std::string> & args) {
    int fd[2];
    if (pipe(fd) != 0) return {};
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fd[1], 1);
    posix_spawn_file_actions_addclose(&fa, fd[0]);
    posix_spawn_file_actions_addclose(&fa, fd[1]);
    pid_t     pid = 0;
    auto      av  = argv_of(args);
    const int rc  = posix_spawnp(&pid, args[0].c_str(), &fa, nullptr, av.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    close(fd[1]);
    std::string out;
    if (rc == 0) {
        char buf[4096];
        for (;;) {
            const ssize_t n = read(fd[0], buf, sizeof(buf));
            if (n > 0) out.append(buf, (size_t) n);
            else if (n == 0 || errno != EINTR) break;
        }
    }
    close(fd[0]);
    if (rc != 0) return {};
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return {};
    while (!out.empty() && out.back() == '\n') out.pop_back();
    return out;
}

static bool on_path(const char * prog) {
    const char * p = std::getenv("PATH");
    if (!p) return false;
    const std::string path = p;
    for (size_t a = 0; a <= path.size();) {
        size_t b = path.find(':', a);
        if (b == std::string::npos) b = path.size();
        if (access((path.substr(a, b - a) + "/" + prog).c_str(), X_OK) == 0) return true;
        a = b + 1;
    }
    return false;
}

std::string pick_path(bool folder, const std::string & title, const std::string & start, const std::string & pattern) {
    // the desktop's own dialog when there is a desktop session (kdialog on KDE, zenity elsewhere); without one (a
    // server, a pod) this returns empty and the web UI takes a typed path
    if (!std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY")) return {};
    const std::string dir = !start.empty() && is_directory(start) ? start : std::string(".");
    const std::string t   = title.empty() ? std::string("Select") : title;
    const bool        kde = std::getenv("KDE_FULL_SESSION") != nullptr;
    if (on_path("kdialog") && (kde || !on_path("zenity"))) {
        if (folder) return capture({ "kdialog", "--title", t, "--getexistingdirectory", dir });
        return capture({ "kdialog", "--title", t, "--getopenfilename", dir, pattern.empty() ? "*" : pattern });
    }
    if (on_path("zenity")) {
        std::vector<std::string> a = { "zenity", "--file-selection", "--title=" + t, "--filename=" + join_path(dir, "") };
        if (folder) a.push_back("--directory");
        if (!folder && !pattern.empty()) a.push_back("--file-filter=" + pattern);
        return capture(a);
    }
    return {};
}

bool wait_pid(int pid, int ms) {
    // kill(pid, 0) fails with ESRCH once the process is gone (it is not our child, so it leaves no zombie here)
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (kill((pid_t) pid, 0) == 0 || errno == EPERM) {
        if (std::chrono::steady_clock::now() >= end) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return true;
}

bool spawn_detached(const std::vector<std::string> & args, std::string & err) {
    if (args.empty()) {
        err = "no program";
        return false;
    }
    // a session of its own, so it outlives this process and its terminal; stdout / stderr inherited (same log)
    posix_spawnattr_t at;
    posix_spawnattr_init(&at);
#if defined(POSIX_SPAWN_SETSID)
    posix_spawnattr_setflags(&at, POSIX_SPAWN_SETSID);
#else
    posix_spawnattr_setflags(&at, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&at, 0);
#endif
    pid_t     pid = 0;
    auto      av  = argv_of(args);
    const int rc  = posix_spawnp(&pid, args[0].c_str(), nullptr, &at, av.data(), environ);
    posix_spawnattr_destroy(&at);
    if (rc != 0) {
        err = "cannot start " + args[0] + " (" + std::strerror(rc) + ")";
        return false;
    }
    return true;
}

int self_pid() {
    return (int) getpid();
}

CpuSample cpu_sample() {
    // /proc/stat's first line: all cores' jiffies (user nice system idle iowait irq softirq steal ...); this process's
    // utime + stime (fields 14 and 15 of /proc/self/stat) are in the same unit
    CpuSample     s;
    std::ifstream f("/proc/stat");
    std::string   cpu;
    uint64_t      v[8] = {};
    if (f >> cpu >> v[0] >> v[1] >> v[2] >> v[3] >> v[4] >> v[5] >> v[6] >> v[7] && cpu == "cpu") {
        for (uint64_t x : v) s.total += x;
        s.busy = s.total - v[3] - v[4];
    }
    std::ifstream     ps("/proc/self/stat");
    const std::string st((std::istreambuf_iterator<char>(ps)), std::istreambuf_iterator<char>());
    const size_t      paren = st.rfind(')');  // the command name may contain spaces
    if (paren != std::string::npos && paren + 2 < st.size()) {
        std::istringstream is(st.substr(paren + 2));
        std::string        tok;
        for (int field = 3; is >> tok && field <= 15; field++) {
            if (field >= 14) s.self += std::strtoull(tok.c_str(), nullptr, 10);
        }
    }
    return s;
}

void set_thread_high_perf() {}

std::vector<std::string> utf8_args(int argc, char ** argv) {
    return std::vector<std::string>(argv, argv + argc);
}

std::string join_path(const std::string & dir, const std::string & name) {
    if (!dir.empty() && dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

bool is_directory(const std::string & path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool remove_file(const std::string & path) {
    return unlink(path.c_str()) == 0;
}

bool make_dir(const std::string & path) {
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

bool remove_dir(const std::string & path) {
    return rmdir(path.c_str()) == 0;
}

} // namespace e8::sys

#endif
