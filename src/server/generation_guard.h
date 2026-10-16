#pragma once
#include <algorithm>
#include <string_view>

namespace e8::cli {
// Detect a sustained adjacent repetition, not repeated snippets elsewhere in
// useful code. Require at least `min_window` bytes (8 KiB; the server asks 32 KiB
// inside a tool call's arguments) and 32 complete copies of the period.
inline bool generation_repeats(std::string_view text, size_t min_window = 8192) {
    for (size_t period = 1; period <= 512; ++period) {
        const size_t window = std::max<size_t>(min_window, period * 32);
        if (text.size() < window) continue;
        const auto tail = text.substr(text.size() - window);
        if (tail.substr(period) == tail.substr(0, window - period)) return true;
    }
    return false;
}
}
