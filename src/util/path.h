#pragma once
// UTF-8 std::string -> filesystem path, on every platform (std::filesystem::u8path is deprecated in C++20)
#include <filesystem>
#include <string>

namespace e8 {

inline std::filesystem::path u8path(const std::string & s) {
    return std::filesystem::path(std::u8string(s.begin(), s.end()));
}

} // namespace e8
