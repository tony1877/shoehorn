#include "util/mmap.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <mutex>
#include <vector>

namespace e8::util {

namespace {
struct Live {
    const uint8_t * p;
    uint64_t        n;
    int             fd;
};
std::mutex        g_mu;
std::vector<Live> g_live;
} // namespace

int mapped_fd(const void * p, uint64_t & offset) {
    std::lock_guard<std::mutex> lk(g_mu);
    for (const Live & l : g_live) {
        if ((const uint8_t *) p >= l.p && (const uint8_t *) p < l.p + l.n) {
            offset = (uint64_t) ((const uint8_t *) p - l.p);
            return l.fd;
        }
    }
    return -1;
}

MappedFile::~MappedFile() {
#if defined(_WIN32)
    if (data_) UnmapViewOfFile(data_);
    if (map_) CloseHandle(map_);
    if (file_ && file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
#else
    if (data_) {
        std::lock_guard<std::mutex> lk(g_mu);
        for (size_t i = 0; i < g_live.size(); i++)
            if (g_live[i].p == data_) g_live.erase(g_live.begin() + (long) i--);
    }
    if (data_) munmap((void *) data_, size_);
    if (fd_ >= 0) close(fd_);
#endif
}

bool MappedFile::open(const std::string & path, std::string & err) {
#if defined(_WIN32)
    const int    n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), n);
    file_ = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (file_ == INVALID_HANDLE_VALUE) {
        err = "cannot open " + path;
        return false;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(file_, &sz)) {
        err = "cannot size " + path;
        return false;
    }
    size_ = (uint64_t) sz.QuadPart;
    map_  = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!map_) {
        err = "CreateFileMapping failed for " + path;
        return false;
    }
    data_ = (const uint8_t *) MapViewOfFile(map_, FILE_MAP_READ, 0, 0, 0);
    if (!data_) {
        err = "MapViewOfFile failed for " + path;
        return false;
    }
#else
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        err = "cannot open " + path;
        return false;
    }
    struct stat st;
    fstat(fd_, &st);
    size_ = (uint64_t) st.st_size;
    void * p = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) {
        err = "mmap failed for " + path;
        return false;
    }
    data_ = (const uint8_t *) p;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_live.push_back({ data_, size_, fd_ });
    }
#endif
    return true;
}

} // namespace e8::util
