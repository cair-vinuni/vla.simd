// GGUF reader: header/metadata/tensor parsing, element widening, corrupt-file
// rejection, the model-path mount rules and the JSON reader it relies on. The
// per-model adapters are checked against real vla.cpp checkpoints by
// tools/check_gguf.py, since they need full-size tensors.

#include "io/files.h"
#include "io/gguf.h"
#include "io/gguf_models.h"
#include "io/json.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace tcpu;

static int failures = 0;

#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);      \
            std::fprintf(stderr, __VA_ARGS__);                             \
            std::fprintf(stderr, "\n");                                    \
            failures++;                                                    \
        }                                                                  \
    } while (0)

// Minimal GGUF v3 writer, enough to exercise every path the reader takes.
struct Writer {
    std::string kv, info, data;
    uint64_t n_kv = 0, n_t = 0;

    template<class T> static void put(std::string& s, T v) { s.append((const char*)&v, sizeof v); }
    static void str(std::string& s, const std::string& v) { put<uint64_t>(s, v.size()); s += v; }

    void key(const std::string& k, uint32_t type) { str(kv, k); put<uint32_t>(kv, type); n_kv++; }
    void u32(const std::string& k, uint32_t v) { key(k, io::GGUF_U32); put(kv, v); }
    void f32(const std::string& k, float v) { key(k, io::GGUF_F32); put(kv, v); }
    void f64(const std::string& k, double v) { key(k, io::GGUF_F64); put(kv, v); }
    void s(const std::string& k, const std::string& v) { key(k, io::GGUF_STRING); str(kv, v); }
    void u8s(const std::string& k, const std::vector<uint8_t>& v) {
        key(k, io::GGUF_ARRAY); put<uint32_t>(kv, io::GGUF_U8); put<uint64_t>(kv, v.size());
        kv.append((const char*)v.data(), v.size());
    }
    void i32s(const std::string& k, const std::vector<int32_t>& v) {
        key(k, io::GGUF_ARRAY); put<uint32_t>(kv, io::GGUF_I32); put<uint64_t>(kv, v.size());
        for (int32_t x : v) put(kv, x);
    }
    void tensor(const std::string& name, std::vector<uint64_t> ne, uint32_t type, const void* p, size_t n) {
        while (data.size() % 32) data += '\0';
        str(info, name);
        put<uint32_t>(info, (uint32_t)ne.size());
        for (uint64_t d : ne) put(info, d);
        put(info, type);
        put<uint64_t>(info, data.size());
        data.append((const char*)p, n);
        n_t++;
    }
    std::string bytes() const {
        std::string h = "GGUF";
        put<uint32_t>(h, 3);
        put(h, n_t);
        put(h, n_kv);
        h += kv + info;
        while (h.size() % 32) h += '\0';
        return h + data;
    }
};

static std::string tmpdir() {
    const char* t = std::getenv("TMPDIR");
    std::string tmpl = std::string(t && *t ? t : "/tmp") + "/test_gguf_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    return mkdtemp(buf.data()) ? std::string(buf.data()) : std::string();
}

static void write(const std::string& path, const std::string& bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(bytes.data(), (std::streamsize)bytes.size());
}

static void test_reader(const std::string& dir) {
    Writer w;
    w.s("general.architecture", "test");
    w.u32("test.n", 7);
    w.f32("test.eps", 1e-6f);
    w.f64("test.period", 0.004);
    w.u8s("test.blob", {0, 1, 2, 255});
    w.i32s("test.ids", {101, -1, 102});
    const float f[6] = {1.5f, -2.0f, 3.25f, 0.0f, -0.5f, 1e-3f};
    w.tensor("a.f32", {3, 2}, io::GGML_F32, f, sizeof f);
    const uint16_t bf[2] = {0x3FC0, 0xC000};              // 1.5, -2.0
    w.tensor("a.bf16", {2}, io::GGML_BF16, bf, sizeof bf);
    const uint16_t hf[3] = {0x3C00, 0xC000, 0x0001};      // 1.0, -2.0, smallest subnormal
    w.tensor("a.f16", {3}, io::GGML_F16, hf, sizeof hf);
    const std::string path = dir + "/model.gguf";
    write(path, w.bytes());

    io::Gguf g;
    CHECK(g.open(path), "open: %s", g.error().c_str());
    CHECK(g.str("general.architecture") == "test", "architecture");
    CHECK(g.num("test.n", 0) == 7, "u32");
    CHECK((float)g.num("test.eps", 0) == 1e-6f, "f32");
    CHECK(g.num("test.period", 0) == 0.004, "f64");
    CHECK(g.num("missing", -1) == -1, "default");
    const io::GgufValue* blob = g.get("test.blob");
    CHECK(blob && blob->bytes == std::vector<uint8_t>({0, 1, 2, 255}), "u8 array");
    const io::GgufValue* ids = g.get("test.ids");
    CHECK(ids && ids->nums == std::vector<double>({101, -1, 102}), "i32 array");

    const io::GgufTensor* t = g.tensor("a.f32");
    CHECK(t && t->shape() == std::vector<int64_t>({2, 3}), "shape is ne reversed");
    std::vector<float> v;
    std::string err;
    CHECK(t && io::tensor_f32(*t, v, err) && std::memcmp(v.data(), f, sizeof f) == 0, "f32 data");
    CHECK(io::tensor_f32(*g.tensor("a.bf16"), v, err) && v == std::vector<float>({1.5f, -2.0f}), "bf16 widen");
    CHECK(io::tensor_f32(*g.tensor("a.f16"), v, err) && v[0] == 1.0f && v[1] == -2.0f &&
          v[2] == std::ldexp(1.0f, -24), "f16 widen");
    CHECK(g.tensor("nope") == nullptr, "missing tensor");

    // round to nearest even, ties included
    CHECK(io::f32_to_bf16(1.0f) == 0x3F80, "bf16 exact");
    uint32_t bits = 0x3F808000;                           // exactly halfway, even below
    float x;
    std::memcpy(&x, &bits, 4);
    CHECK(io::f32_to_bf16(x) == 0x3F80, "bf16 tie to even");
    bits = 0x3F818000;                                    // halfway, odd below
    std::memcpy(&x, &bits, 4);
    CHECK(io::f32_to_bf16(x) == 0x3F82, "bf16 tie rounds up to even");

    // truncation anywhere must fail cleanly, never read past the file
    const std::string whole = w.bytes();
    for (size_t cut : {size_t(3), size_t(20), whole.size() / 2, whole.size() - 1}) {
        write(dir + "/cut.gguf", whole.substr(0, cut));
        io::Gguf c;
        CHECK(!c.open(dir + "/cut.gguf"), "truncated at %zu accepted", cut);
    }
    std::string bad = whole;
    bad[4] = 9;                                           // version 9
    write(dir + "/bad.gguf", bad);
    io::Gguf b;
    CHECK(!b.open(dir + "/bad.gguf"), "bad version accepted");
    CHECK(g.open(path), "reopen: %s", g.error().c_str());
    Writer empty;
    write(dir + "/bad.gguf", empty.bytes());
    CHECK(g.open(dir + "/bad.gguf") && g.tensors().empty() && g.keys().empty(), "reopen retains state");
    CHECK(!g.open(dir + "/missing.gguf"), "missing file accepted");
    CHECK(g.open(path) && g.error().empty(), "reopen retains error");
    Writer overflow;
    overflow.tensor("overflow", {uint64_t(1) << 32, uint64_t(1) << 32}, io::GGML_F32, f, sizeof f);
    write(dir + "/bad.gguf", overflow.bytes());
    CHECK(!b.open(dir + "/bad.gguf"), "overflowing shape accepted");
    Writer duplicate;
    duplicate.u32("same", 1);
    duplicate.u32("same", 2);
    write(dir + "/bad.gguf", duplicate.bytes());
    CHECK(!b.open(dir + "/bad.gguf"), "duplicate metadata accepted");
    for (double alignment : {-32.0, 0.5, std::numeric_limits<double>::infinity()}) {
        Writer a;
        a.f64("general.alignment", alignment);
        write(dir + "/bad.gguf", a.bytes());
        CHECK(!b.open(dir + "/bad.gguf"), "invalid alignment accepted");
    }
    for (uint32_t nan_bits : {0x7F800001u, 0x7FFFFFFFu, 0xFF800001u, 0xFFFFFFFFu}) {
        std::memcpy(&x, &nan_bits, sizeof x);
        const uint16_t h = io::f32_to_bf16(x);
        CHECK((h & 0x7F80) == 0x7F80 && (h & 0x7F) != 0, "bf16 lost NaN");
    }
    unlink((dir + "/cut.gguf").c_str());
    unlink((dir + "/bad.gguf").c_str());
}

static void test_mount(const std::string& dir) {
    // model.gguf from test_reader: a directory with one .gguf mounts as a GGUF
    CHECK(io::find_gguf(dir) == dir + "/model.gguf", "dir with one gguf");
    CHECK(io::find_gguf(dir + "/model.gguf") == dir + "/model.gguf", "gguf file");
    {
        // ...and an architecture no adapter knows is a load failure, not a plain dir
        io::Mount m(dir);
        CHECK(!m.ok(), "unknown architecture mounted");
    }
    write(dir + "/vit.meta", "hidden 768\n");
    CHECK(io::find_gguf(dir).empty(), "a converted dir (has .meta) is not a GGUF");
    io::Mount plain(dir);
    CHECK(plain.ok() && !plain.gguf(), "plain dir");
    io::InFile f(dir + "/vit.meta");
    std::string k;
    int v = 0;
    CHECK((f >> k >> v) && k == "hidden" && v == 768, "disk read through InFile");
    io::InFile missing(dir + "/nope.meta");
    CHECK(!missing, "missing file tests false");
    unlink((dir + "/vit.meta").c_str());
    Writer w;
    w.s("general.architecture", "vla-simd");
    w.u32("vla_simd.format", 1);
    std::vector<float> weights(1 << 18, 1.5f);
    w.tensor("weights.bin", {weights.size()}, io::GGML_F32, weights.data(), weights.size()*sizeof(float));
    const std::string path = dir + "/own.gguf";
    write(path, w.bytes());
    {
        io::Mount first(path);
        io::InFile one(path + "/weights.bin", std::ios::binary);
        CHECK(first.ok() && one, "first load");
        auto second = std::async(std::launch::async, [&] {
            io::Mount mount(path);
            io::InFile two(path + "/weights.bin", std::ios::binary);
            float value = 0;
            two.read((char*)&value, sizeof value);
            return mount.ok() && two && value == 1.5f;
        });
        CHECK(second.get(), "concurrent load lost consumed weights");
    }
    for (const char* rel : {"../escape.txt", "/absolute.txt", "tok/../../escape"}) {
        Writer bad;
        bad.s("general.architecture", "vla-simd");
        bad.u32("vla_simd.format", 1);
        bad.s(std::string("vla_simd.file.") + rel, "bad");
        write(path, bad.bytes());
        io::Gguf g;
        io::Files files;
        std::string error;
        CHECK(g.open(path) && !io::adapt_gguf(g, io::Sidecar{dir}, files, error), "unsafe embedded path accepted");
    }
    unlink(path.c_str());
}

static void test_json() {
    io::Json j;
    CHECK(io::Json::parse(R"({"a": [1, 2.5e-1, true, null], "s": "x\"é", "o": {"k": -3}})", j), "parse");
    const io::Json* a = j.at("a");
    CHECK(a && a->arr.size() == 4 && a->arr[1].num == 0.25 && a->arr[2].as_num() == 1, "array");
    CHECK(j.at("s") && j.at("s")->str == "x\"\xC3\xA9", "string escapes");
    CHECK(j.at("o") && j.at("o")->at("k") && j.at("o")->at("k")->num == -3, "nested");
    CHECK(io::Json::parse("[NaN, Infinity]", j) && std::isnan(j.arr[0].num) && std::isinf(j.arr[1].num),
          "python non-finite");
    CHECK(!io::Json::parse("{\"a\": }", j), "reject malformed");
    CHECK(!io::Json::parse("[1] x", j), "reject trailing");
    CHECK(io::Json::parse("[1]", j) && j.arr.size() == 1, "reparse retains array");
    CHECK(io::Json::parse(R"({"a": "old", "a": "new"})", j) && j.at("a")->str == "new",
          "duplicate JSON key does not replace value");
    for (const char* s : {R"("\u12zz")", R"("\uD800")", R"("\uDC00")", "\"a\nb\"",
                          "[01]", "[+1]", "[.5]", "[1.]", "[0x1]", "[1e]"})
        CHECK(!io::Json::parse(s, j), "invalid JSON accepted: %s", s);
    CHECK(io::Json::parse(R"("\uD83D\uDE00")", j) && j.str == "\xF0\x9F\x98\x80", "surrogate pair");
}

int main() {
    const std::string dir = tmpdir();
    CHECK(!dir.empty(), "mkdtemp");
    test_reader(dir);
    test_mount(dir);
    test_json();
    unlink((dir + "/model.gguf").c_str());
    rmdir(dir.c_str());
    if (failures) std::fprintf(stderr, "%d check(s) failed\n", failures);
    else std::printf("ok\n");
    return failures ? 1 : 0;
}
