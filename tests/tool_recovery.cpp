#include "server/tool_recovery.h"
#include "server/output_policy.h"
#include "server/generation_guard.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>

// not assert(): Release builds define NDEBUG, which would compile every check away
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #c); std::exit(1); } } while (0)

int main() {
    std::string repetitive;
    for (int i = 0; i < 1000; ++i) repetitive += "                   \\\"disgusting\\\", \\\"disgusting\\\", \\\"disgusting\\\", \\\"disgusting\\\",\\n";
    CHECK(e8::cli::generation_repeats(repetitive));
    CHECK(!e8::cli::generation_repeats(repetitive.substr(0, 8191)));
    std::string code;
    for (int i = 0; i < 1000; ++i) code += "const value_" + std::to_string(i) + " = compute(" + std::to_string(i) + ");\n";
    CHECK(!e8::cli::generation_repeats(code));
    CHECK(!e8::cli::generation_repeats(repetitive + code));
    common_chat_tool tool;
    tool.name = "write";
    tool.parameters = R"({"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}}})";
    const std::vector<common_chat_tool> tools{tool};
    const std::string raw = "<tool_call><function=write><parameter=content>\nline é😀\nline 2\n</parameter><parameter=file_path>\na.txt\n</parameter></function></tool_call>";
    std::string previous;
    size_t streamed = 0;
    for (size_t n = 1; n <= raw.size(); ++n) {
        common_chat_msg message;
        e8::cli::lenient_tool_calls(raw.substr(0, n), tools, message, n < raw.size());
        if (message.tool_calls.empty()) continue;
        const auto & arguments = message.tool_calls[0].arguments;
        CHECK(arguments.rfind(previous, 0) == 0);
        previous = arguments;
        ++streamed;
    }
    CHECK(streamed > 50);
    const auto parsed = nlohmann::ordered_json::parse(previous);
    CHECK(parsed["content"] == "line é😀\nline 2");
    CHECK(parsed["file_path"] == "a.txt");
    common_chat_msg strict;
    strict.tool_calls.push_back({"unchanged", "{}", "id"});
    e8::cli::lenient_tool_calls(raw, tools, strict, true);
    CHECK(strict.tool_calls[0].name == "unchanged");
    common_chat_msg answer;
    answer.content = std::string(20000, 'x');
    answer.reasoning_content = "reasoning is separately budgeted";
    CHECK(e8::cli::bound_answer(answer, 16384));
    CHECK(answer.content.size() == 16384);
    answer.content = "éé";
    CHECK(e8::cli::bound_answer(answer, 3));
    CHECK(answer.content == "é");
    CHECK(!e8::cli::bound_answer(strict, 1));
    std::cout << "Incremental reordered file arguments are monotonic and match final JSON\n";
}
