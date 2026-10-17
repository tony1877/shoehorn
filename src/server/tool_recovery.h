#pragma once
#include "chat.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>

namespace e8::cli {
using json = nlohmann::ordered_json;

// A non-string tool argument (a list, an object, a number) that is not valid JSON as written, with the slips models
// make: a ```json fence, trailing commas, single-quoted strings, unquoted keys, Python's True / False / None. Returns
// the repaired text (callers parse it; when it still fails, the value goes on as a string and the client's schema
// check tells the model).
inline std::string repair_json(std::string v) {
    auto trim = [](std::string & x) {
        while (!x.empty() && std::isspace(static_cast<unsigned char>(x.back()))) x.pop_back();
        size_t a = 0;
        while (a < x.size() && std::isspace(static_cast<unsigned char>(x[a]))) a++;
        x.erase(0, a);
    };
    trim(v);
    if (v.rfind("```", 0) == 0) {  // ```json ... ```
        const size_t nl = v.find('\n');
        v = nl == std::string::npos ? std::string() : v.substr(nl + 1);
        if (v.size() >= 3 && v.compare(v.size() - 3, 3, "```") == 0) v.resize(v.size() - 3);
        trim(v);
    }
    std::string out;
    out.reserve(v.size() + 16);
    auto ident = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-'; };
    for (size_t i = 0; i < v.size();) {
        const char c = v[i];
        if (c == '"' || c == '\'') {  // a string: copied, single quotes turned into double
            const char q = c;
            out += '"';
            for (i++; i < v.size() && v[i] != q; i++) {
                if (v[i] == '\\' && i + 1 < v.size()) {
                    if (q == '\'' && v[i + 1] == '\'') out += '\'';  // \' needs no escape in a JSON string
                    else (out += v[i]) += v[i + 1];
                    i++;
                    continue;
                }
                if (q == '\'' && v[i] == '"') out += "\\\"";
                else out += v[i];
            }
            out += '"';
            i++;
            continue;
        }
        if (c == ',') {  // trailing comma: dropped when only space comes before } or ]
            size_t j = i + 1;
            while (j < v.size() && std::isspace(static_cast<unsigned char>(v[j]))) j++;
            if (j < v.size() && (v[j] == '}' || v[j] == ']')) {
                i++;
                continue;
            }
        }
        if (ident(c) && !std::isdigit(static_cast<unsigned char>(c)) && c != '-') {  // a bare word
            size_t j = i;
            while (j < v.size() && ident(v[j])) j++;
            const std::string w = v.substr(i, j - i);
            size_t k = j;
            while (k < v.size() && std::isspace(static_cast<unsigned char>(v[k]))) k++;
            if (k < v.size() && v[k] == ':') out += '"' + w + '"';  // an unquoted key
            else if (w == "True") out += "true";
            else if (w == "False") out += "false";
            else if (w == "None") out += "null";
            else out += w;
            i = j;
            continue;
        }
        out += c;
        i++;
    }
    return out;
}

// Tool calls in the Qwen XML format: <tool_call> <function=NAME> <parameter=P> VALUE </parameter> ... </function>
// </tool_call>. The format has no escaping, so a value can itself contain that markup: a file that documents tool
// calls, a test of this parser. llama.cpp's grammar ends a value at the first "\n</parameter>\n", which cuts such a
// file short and turns the markup quoted in it into extra calls (writing the wrong files). This parser reads calls in
// order and takes a "</parameter>" as a value's end only when what follows continues the call:
//   - <parameter=Q> for a parameter of this tool that is not set yet and is not the one being read, or
//   - </function>, then </tool_call>, then the end of the text or the next <tool_call>.
// Any other "</parameter>" is part of the value. It also takes parameters in any order (the model often writes
// str_replace_editor's file_text before path, which the grammar rejects) and types each value by the tool's JSON
// schema (strings as written, anything else parsed as JSON).
// With partial = true (streaming) an unfinished call is returned with its JSON left open, so the arguments only grow:
// a candidate end that cannot be decided yet is held back. Without it, a call missing its closing tags is dropped.
// Search starts after the reasoning (a </think> before the first call), so markup mentioned while thinking is not a
// call; a </think> inside a call's value (a file about reasoning tags) is just text.
// Returns the calls; `content_end` gets where the first call starts (the end of the text before it).
inline std::vector<common_chat_tool_call> xml_tool_calls(const std::string & text, const std::vector<common_chat_tool> & tools,
                                                         bool partial, size_t & content_end) {
    std::vector<common_chat_tool_call> calls;
    auto starts = [&](size_t at, const char * s) { return text.compare(at, strlen(s), s) == 0; };
    // the text from `at` to the end is a proper prefix of s (only more text can decide)
    auto prefix_of = [&](size_t at, const char * s) {
        const size_t n = text.size() - at;
        return n < strlen(s) && text.compare(at, n, s, n) == 0;
    };
    auto skip_ws = [&](size_t at) {
        while (at < text.size() && (text[at] == '\n' || text[at] == ' ' || text[at] == '\r' || text[at] == '\t')) at++;
        return at;
    };
    // the next <tool_call> from `from` that starts a call (<function= follows, or may still follow); a mention of
    // the tag in prose is skipped
    auto next_call = [&](size_t from) {
        for (size_t q = text.find("<tool_call>", from); q != std::string::npos; q = text.find("<tool_call>", q + 11)) {
            const size_t a = skip_ws(q + 11);
            if (a >= text.size() ? partial : starts(a, "<function=") || (partial && prefix_of(a, "<function="))) return q;
        }
        return std::string::npos;
    };
    // reasoning ends at the first </think> when it comes before the first call (a </think> after it is file text)
    size_t       p     = next_call(0);
    const size_t think = text.find("</think>");
    if (think != std::string::npos && (p == std::string::npos || think < p)) p = next_call(think + 8);
    content_end = p;
    auto trim1 = [](std::string v) {
        if (!v.empty() && v.front() == '\n') v.erase(0, 1);
        if (!v.empty() && v.back() == '\n') v.pop_back();
        return v;
    };
    enum Verdict { kEnd, kText, kWait };
    while (p != std::string::npos) {
        size_t pos = skip_ws(p + 11);
        if (pos >= text.size() || prefix_of(pos, "<function=") || !starts(pos, "<function=")) {
            break;  // nothing of the call yet, or not a call
        }
        const size_t fe = text.find('>', pos);
        if (fe == std::string::npos) break;
        const std::string name = text.substr(pos + 10, fe - pos - 10);
        json              schema = json::object();
        for (const auto & t : tools) {
            if (t.name == name) {
                try {
                    schema = json::parse(t.parameters).value("properties", json::object());
                } catch (...) {}
            }
        }
        json                  args = json::object();
        std::set<std::string> set;
        bool                  complete = false, open_value = false, stop = false;
        pos = fe + 1;
        // is the "</parameter>" at c the end of parameter `pn`'s value?
        auto verdict = [&](size_t c, const std::string & pn) {
            const size_t a = skip_ws(c + 12);
            if (a >= text.size()) return partial ? kWait : kText;
            if (starts(a, "<parameter=")) {
                const size_t ne = text.find('>', a);
                if (ne == std::string::npos) return partial ? kWait : kText;
                const std::string q = text.substr(a + 11, ne - a - 11);
                const bool known = schema.empty() || schema.contains(q);
                return q != pn && known && !set.count(q) ? kEnd : kText;
            }
            if (starts(a, "</function>")) {
                const size_t b = skip_ws(a + 11);
                if (b >= text.size()) return partial ? kWait : kText;
                if (!starts(b, "</tool_call>")) return prefix_of(b, "</tool_call>") && partial ? kWait : kText;
                const size_t e = skip_ws(b + 12);
                if (e >= text.size()) return partial ? kWait : kEnd;
                if (starts(e, "<tool_call>")) return kEnd;
                return prefix_of(e, "<tool_call>") && partial ? kWait : kText;
            }
            if (partial && (prefix_of(a, "<parameter=") || prefix_of(a, "</function>"))) return kWait;
            return kText;
        };
        auto set_arg = [&](const std::string & pn, std::string v, bool value_partial) {
            v = trim1(v);
            if (value_partial && !v.empty()) {
                // tokens can split a UTF-8 code point: hold its first bytes until the rest arrives
                size_t lead = v.size() - 1;
                while (lead && (static_cast<unsigned char>(v[lead]) & 0xC0) == 0x80) --lead;
                const auto   byte  = static_cast<unsigned char>(v[lead]);
                const size_t width = byte >= 0xF0 ? 4 : byte >= 0xE0 ? 3 : byte >= 0xC0 ? 2 : 1;
                if (v.size() - lead < width) v.resize(lead);
            }
            const std::string ty = schema.contains(pn) && schema[pn].contains("type") && schema[pn]["type"].is_string()
                                       ? schema[pn]["type"].get<std::string>() : "string";
            if (ty == "string") {
                args[pn]   = v;
                open_value = value_partial;
                return true;
            }
            if (value_partial) return false;  // a number or object is shown once complete
            try {
                args[pn] = json::parse(v);
            } catch (...) {
                try {
                    args[pn] = json::parse(repair_json(v));
                } catch (...) {
                    args[pn] = v;
                }
            }
            return true;
        };
        while (!stop) {
            pos = skip_ws(pos);
            if (pos >= text.size()) {
                stop = true;
                break;
            }
            if (starts(pos, "<parameter=")) {
                const size_t ne = text.find('>', pos);
                if (ne == std::string::npos) {
                    stop = true;
                    break;
                }
                const std::string pn     = text.substr(pos + 11, ne - pos - 11);
                const size_t      vstart = ne + 1;
                size_t            search = vstart;
                for (;;) {
                    const size_t c = text.find("</parameter>", search);
                    if (c == std::string::npos) {
                        if (partial) {
                            // the value goes on; hold a partly generated "</parameter>" at its end
                            std::string v = text.substr(vstart);
                            const std::string closing = "</parameter>";
                            for (size_t n = std::min(v.size(), closing.size() - 1); n > 0; --n) {
                                if (v.compare(v.size() - n, n, closing, 0, n) == 0) {
                                    v.resize(v.size() - n);
                                    break;
                                }
                            }
                            set_arg(pn, v, true);
                        }
                        stop = true;
                        break;
                    }
                    const Verdict vd = verdict(c, pn);
                    if (vd == kEnd) {
                        set_arg(pn, text.substr(vstart, c - vstart), false);
                        set.insert(pn);
                        pos = c + 12;
                        break;
                    }
                    if (vd == kWait) {  // only more text tells: show the value up to here
                        set_arg(pn, text.substr(vstart, c - vstart), true);
                        stop = true;
                        break;
                    }
                    search = c + 12;  // "</parameter>" inside the value
                }
                continue;
            }
            if (starts(pos, "</function>")) {
                const size_t b = skip_ws(pos + 11);
                if (starts(b, "</tool_call>")) {
                    complete = true;
                    pos      = b + 12;
                }
                stop = true;
                break;
            }
            stop = true;  // anything else: the call is malformed or not finished
        }
        if (!complete && !partial) break;  // cut off: never a finished call
        common_chat_tool_call tc;
        tc.name      = name;
        tc.arguments = args.dump();
        if (!complete) {
            tc.arguments.pop_back();  // leave the object open for subsequent parameters
            if (open_value) tc.arguments.pop_back();  // and the last string
        }
        calls.push_back(tc);
        if (!complete) break;
        p = next_call(pos);
    }
    return calls;
}

// Tool calls of the text replace the strict parse's whenever it has a <tool_call> (xml_tool_calls); the content is
// cut where the first call starts. Logged when the result differs from the strict parse (final parses only).
inline void lenient_tool_calls(const std::string & text, const std::vector<common_chat_tool> & tools, common_chat_msg & msg, bool partial = false) {
    if (text.find("<tool_call>") == std::string::npos) return;
    size_t     content_end = std::string::npos;
    const auto calls       = xml_tool_calls(text, tools, partial, content_end);
    // a model that writes its calls another way (JSON inside <tool_call>, as Hermes-style fine-tunes do): the strict
    // parse's calls stand
    if (calls.empty() && !msg.tool_calls.empty() && text.find("<function=") == std::string::npos) return;
    if (!partial && calls.size() != msg.tool_calls.size()) {
        fprintf(stderr, "tool calls: %zu by the XML parser, %zu by the strict grammar (the XML parser's are used)\n",
                calls.size(), msg.tool_calls.size());
    }
    msg.tool_calls = calls;
    const size_t c = msg.content.find("<tool_call>");
    if (c != std::string::npos) msg.content.erase(c);
}

}
