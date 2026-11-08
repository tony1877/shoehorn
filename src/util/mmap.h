#pragma once

// Read-only memory mapping of a whole file (Windows: CreateFileMapping/MapViewOfFile; POSIX: mmap).

#include <cstddef>
#include <cstdint>
#include <string>

namespace e8::util {

class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();
    MappedFile(const MappedFile &)             = delete;
    MappedFile & operator=(const MappedFile &) = delete;

    bool open(const std::string & path, std::string & err);

    const uint8_t * data() const { return data_; }
    uint64_t        size() const { return size_; }

private:
    const uint8_t * data_ = nullptr;
    uint64_t        size_ = 0;
#if defined(_WIN32)
    void * file_ = nullptr;
    void * map_  = nullptr;
#else
    int fd_ = -1;
#endif
};

// POSIX: the descriptor and file offset behind an address inside a live MappedFile (-1 when none), so prefetchers can
// read the file with pread (large requests) instead of faulting the mapping in 128 KB readahead pieces
int mapped_fd(const void * p, uint64_t & offset);

} // namespace e8::util
