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
    // inside a tool call the window is 32 KiB: 8 KiB of identical data rows is not a loop there
    CHECK(!e8::cli::generation_repeats(repetitive.substr(0, 16384), 32768));
    CHECK(e8::cli::generation_repeats(repetitive.substr(0, 40000), 32768));
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
    // the XML parser's calls replace the strict grammar's (which splits quoted markup into extra calls)
    common_chat_msg strict;
    strict.tool_calls.push_back({"unchanged", "{}", "id"});
    e8::cli::lenient_tool_calls(raw, tools, strict, false);
    CHECK(strict.tool_calls.size() == 1 && strict.tool_calls[0].name == "write");

    // streams `text` byte by byte: the arguments only grow, and the final parse gives `want` calls
    auto stream_check = [&](const std::string & text, const std::vector<common_chat_tool> & ts, size_t want) {
        std::vector<std::string> prev(want);
        for (size_t n = 1; n <= text.size(); ++n) {
            common_chat_msg m;
            e8::cli::lenient_tool_calls(text.substr(0, n), ts, m, n < text.size());
            CHECK(m.tool_calls.size() <= want);
            for (size_t i = 0; i < m.tool_calls.size(); ++i) {
                CHECK(m.tool_calls[i].arguments.rfind(prev[i], 0) == 0);
                prev[i] = m.tool_calls[i].arguments;
            }
        }
        common_chat_msg m;
        e8::cli::lenient_tool_calls(text, ts, m, false);
        CHECK(m.tool_calls.size() == want);
        return m;
    };
    // a file that documents the format: its quoted markup stays text, and is not a second call
    const std::string doc = "# Format\n\n```text\n<tool_call>\n<function=write>\n<parameter=file_path>\nx.txt\n</parameter>\n"
                            "<parameter=content>\nHello\n</parameter>\n</function>\n</tool_call>\n```\n\nEnd `</parameter>` too.";
    auto m1 = stream_check("<tool_call>\n<function=write>\n<parameter=file_path>\nnotes/doc.md\n</parameter>\n<parameter=content>\n" +
                           doc + "\n</parameter>\n</function>\n</tool_call>", tools, 1);
    CHECK(nlohmann::ordered_json::parse(m1.tool_calls[0].arguments)["content"] == doc);
    CHECK(nlohmann::ordered_json::parse(m1.tool_calls[0].arguments)["file_path"] == "notes/doc.md");
    // a parameter the tool does not have, quoted in a file
    const std::string tmpl = "<parameter=name>\nvalue\n</parameter>";
    auto m2 = stream_check("<tool_call>\n<function=write>\n<parameter=file_path>\nt.txt\n</parameter>\n<parameter=content>\n" +
                           tmpl + "\n</parameter>\n</function>\n</tool_call>", tools, 1);
    CHECK(nlohmann::ordered_json::parse(m2.tool_calls[0].arguments)["content"] == tmpl);
    // two calls in a row
    auto m3 = stream_check("<tool_call>\n<function=write>\n<parameter=file_path>\na\n</parameter>\n<parameter=content>\n1\n</parameter>\n"
                           "</function>\n</tool_call>\n<tool_call>\n<function=write>\n<parameter=file_path>\nb\n</parameter>\n"
                           "<parameter=content>\n2\n</parameter>\n</function>\n</tool_call>", tools, 2);
    CHECK(nlohmann::ordered_json::parse(m3.tool_calls[1].arguments)["file_path"] == "b");
    // cut off (end of turn inside the call): never a finished call
    common_chat_msg m4;
    e8::cli::lenient_tool_calls("<tool_call>\n<function=write>\n<parameter=file_path>\nc\n</parameter>\n<parameter=content>\nhalf", tools, m4, false);
    CHECK(m4.tool_calls.empty());
    // markup mentioned while reasoning is not a call
    common_chat_msg m5;
    e8::cli::lenient_tool_calls("<think>\nI will emit <tool_call> next.\n</think>\n<tool_call>\n<function=write>\n<parameter=file_path>\nd\n"
                                "</parameter>\n<parameter=content>\nok\n</parameter>\n</function>\n</tool_call>", tools, m5, false);
    CHECK(m5.tool_calls.size() == 1 && nlohmann::ordered_json::parse(m5.tool_calls[0].arguments)["file_path"] == "d");
    // a file about reasoning tags (no reasoning in the text): its </think> is file text
    const std::string tags = "Use <think> and </think>:\n```\n<think>\nsteps\n</think>\nanswer\n```";
    auto m7 = stream_check("<tool_call>\n<function=write>\n<parameter=file_path>\nr.md\n</parameter>\n<parameter=content>\n" + tags +
                           "\n</parameter>\n</function>\n</tool_call>", tools, 1);
    CHECK(nlohmann::ordered_json::parse(m7.tool_calls[0].arguments)["content"] == tags);
    // non-string arguments with common JSON slips are repaired
    using e8::cli::repair_json;
    CHECK(nlohmann::ordered_json::parse(repair_json("[1, 2, 3,]")) == nlohmann::ordered_json::parse("[1,2,3]"));
    CHECK(nlohmann::ordered_json::parse(repair_json("{'a': True, 'b': None, c: 'say \"hi\"',}"))
          == nlohmann::ordered_json({ { "a", true }, { "b", nullptr }, { "c", "say \"hi\"" } }));
    CHECK(nlohmann::ordered_json::parse(repair_json("```json\n[{\"id\": 1, \"done\": False}]\n```"))
          == nlohmann::ordered_json::parse(R"([{"id":1,"done":false}])"));
    CHECK(nlohmann::ordered_json::parse(repair_json(R"({"text": "it's, fine]", "n": -2.5e3})"))
          == nlohmann::ordered_json::parse(R"({"text":"it's, fine]","n":-2500.0})"));
    common_chat_tool todo;
    todo.name = "todo_write";
    todo.parameters = R"({"type":"object","properties":{"todos":{"type":"array"}}})";
    common_chat_msg m8;
    e8::cli::lenient_tool_calls("<tool_call>\n<function=todo_write>\n<parameter=todos>\n[{'content': 'fix bug', 'status': 'pending'},]\n"
                                "</parameter>\n</function>\n</tool_call>", { todo }, m8, false);
    CHECK(m8.tool_calls.size() == 1);
    CHECK(nlohmann::ordered_json::parse(m8.tool_calls[0].arguments)["todos"][0]["content"] == "fix bug");
    // JSON-style calls (no <function=): the strict parse's calls are kept
    common_chat_msg m10;
    m10.tool_calls.push_back({ "write", R"({"file_path":"j.txt","content":"x"})", "id" });
    e8::cli::lenient_tool_calls("<tool_call>\n{\"name\": \"write\", \"arguments\": {\"file_path\": \"j.txt\", \"content\": \"x\"}}\n</tool_call>",
                                tools, m10, false);
    CHECK(m10.tool_calls.size() == 1 && m10.tool_calls[0].name == "write");
    // a function the request does not offer is passed on by name (the client reports "unknown tool" to the model)
    common_chat_msg m9;
    e8::cli::lenient_tool_calls("<tool_call>\n<function=magic_wand>\n<parameter=spell>\nlumos\n</parameter>\n</function>\n</tool_call>",
                                tools, m9, false);
    CHECK(m9.tool_calls.size() == 1 && m9.tool_calls[0].name == "magic_wand");
    CHECK(nlohmann::ordered_json::parse(m9.tool_calls[0].arguments)["spell"] == "lumos");
    // parameters in any order, typed by the schema
    common_chat_tool sre;
    sre.name = "str_replace_editor";
    sre.parameters = R"({"type":"object","properties":{"command":{"type":"string"},"path":{"type":"string"},"insert_line":{"type":"integer"},"new_str":{"type":"string"}}})";
    auto m6 = stream_check("<tool_call>\n<function=str_replace_editor>\n<parameter=new_str>\nx = 1\n</parameter>\n<parameter=insert_line>\n3\n"
                           "</parameter>\n<parameter=command>\ninsert\n</parameter>\n<parameter=path>\nf.py\n</parameter>\n</function>\n</tool_call>",
                           { sre }, 1);
    const auto a6 = nlohmann::ordered_json::parse(m6.tool_calls[0].arguments);
    CHECK(a6["insert_line"] == 3 && a6["path"] == "f.py" && a6["new_str"] == "x = 1");
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
