#pragma once
#include "chat.h"

namespace e8::cli {
// Explicit per-request answer cap; tool arguments and reasoning use their own budgets.
inline bool bound_answer(common_chat_msg & message, size_t max_chars) {
    if (!max_chars || !message.tool_calls.empty() || message.content.size() <= max_chars) return false;
    size_t end = max_chars;
    while (end && (static_cast<unsigned char>(message.content[end]) & 0xC0) == 0x80) --end;
    message.content.resize(end);
    return true;
}
}
