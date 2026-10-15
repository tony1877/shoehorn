#pragma once
#include "chat.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdio>

namespace e8::cli {
using json = nlohmann::ordered_json;
// llama.cpp's Qwen XML tool-call grammar rejects a call whose optional parameters come before a required one (the
// model often writes str_replace_editor's file_text before path) and the call then vanishes: the turn looks like it
// ends mid-sentence. When the strict parse of the final text finds no call although the text has a <tool_call>, this
// reads <tool_call><function=NAME><parameter=P>VALUE</parameter>...</function></tool_call> blocks in any order,
// typing each value by the tool's JSON schema (strings as written, anything else parsed as JSON).
// With partial = true (streaming) an unfinished call is returned with its JSON left open, so the arguments only grow.
inline void lenient_tool_calls(const std::string & text, const std::vector<common_chat_tool> & tools, common_chat_msg & msg, bool partial = false) {
    if (!msg.tool_calls.empty()) return;
    size_t p = text.find("<tool_call>");
    if (p == std::string::npos) return;

    auto trim1 = [](std::string v) {
        if (!v.empty() && v.front() == '\n') v.erase(0, 1);
        if (!v.empty() && v.back() == '\n') v.pop_back();
        return v;
    };
    while (p != std::string::npos) {
        const size_t end  = text.find("</tool_call>", p);
        const std::string blk = text.substr(p, end == std::string::npos ? std::string::npos : end - p);
        const size_t fn   = blk.find("<function=");
        if (fn == std::string::npos) break;
        const size_t fe   = blk.find('>', fn);
        if (fe == std::string::npos) break;
        const std::string name = blk.substr(fn + 10, fe - fn - 10);
        json schema = json::object();
        for (const auto & t : tools) {
            if (t.name == name) {
                try {
                    schema = json::parse(t.parameters).value("properties", json::object());
                } catch (...) {}
            }
        }
        json   args = json::object();
        bool value_partial = false;
        size_t q    = fe;
        while ((q = blk.find("<parameter=", q)) != std::string::npos) {
            const size_t ne = blk.find('>', q);
            const size_t ve = blk.find("</parameter>", ne);
            if (ne == std::string::npos) break;
            if (ve == std::string::npos && !partial) break;
            const std::string pn = blk.substr(q + 11, ne - q - 11);
            std::string value = blk.substr(ne + 1, ve == std::string::npos ? std::string::npos : ve - ne - 1);
            if (ve == std::string::npos) {
                // Hold a partly generated closing delimiter rather than expose it as file text.
                const std::string closing = "</parameter>";
                for (size_t n = std::min(value.size(), closing.size() - 1); n > 0; --n) {
                    if (value.compare(value.size() - n, n, closing, 0, n) == 0) {
                        value.resize(value.size() - n);
                        break;
                    }
                }
            }
            std::string v = trim1(value);
            if (ve == std::string::npos && !v.empty()) {
                // Tokens can split a UTF-8 code point; JSON serialization must wait
                // for its remaining bytes rather than throw or publish replacement text.
                size_t lead = v.size() - 1;
                while (lead && (static_cast<unsigned char>(v[lead]) & 0xC0) == 0x80) --lead;
                const auto byte = static_cast<unsigned char>(v[lead]);
                const size_t width = byte >= 0xF0 ? 4 : byte >= 0xE0 ? 3 : byte >= 0xC0 ? 2 : 1;
                if (v.size() - lead < width) v.resize(lead);
            }
            const std::string ty = schema.contains(pn) && schema[pn].contains("type") && schema[pn]["type"].is_string()
                                       ? schema[pn]["type"].get<std::string>() : "string";
            if (ty == "string") {
                args[pn] = v;
                value_partial = ve == std::string::npos;
            } else {
                if (ve == std::string::npos) break;
                try {
                    args[pn] = json::parse(v);
                } catch (...) {
                    args[pn] = v;
                }
            }
            if (ve == std::string::npos) break;
            q = ve + 12;
        }
        common_chat_tool_call tc;
        tc.name      = name;
        tc.arguments = args.dump();
        if (partial && end == std::string::npos) {
            tc.arguments.pop_back(); // leave the object open for subsequent parameters
            if (value_partial) tc.arguments.pop_back(); // leave the last string open
        }
        msg.tool_calls.push_back(tc);
        p = end == std::string::npos ? std::string::npos : text.find("<tool_call>", end);
    }
    if (!msg.tool_calls.empty()) {
        const size_t c = msg.content.find("<tool_call>");
        if (c != std::string::npos) msg.content.erase(c);
        if (!partial) fprintf(stderr, "tool call recovered by the lenient parser (strict grammar rejected it): %s\n", msg.tool_calls[0].name.c_str());
    }
}

}
