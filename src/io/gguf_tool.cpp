/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// vla-simd-gguf info <model.gguf>
//     architecture, metadata and tensor dtypes of a vla.cpp GGUF
// vla-simd-gguf extract <model.gguf | dir> <out-dir>
//     write the files the GGUF loads as: the converted directory the Python
//     converter would have produced (sidecars are read, not copied)

#include "io/files.h"
#include "io/gguf.h"
#include "io/gguf_models.h"
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <sys/stat.h>

using namespace tcpu::io;

static int usage() {
    std::fprintf(stderr, "usage: vla-simd-gguf info <model.gguf>\n"
                         "       vla-simd-gguf extract <model.gguf | dir> <out-dir>\n");
    return 2;
}

static int info(const std::string& path) {
    Gguf g;
    if (!g.open(path)) { std::fprintf(stderr, "%s\n", g.error().c_str()); return 1; }
    std::printf("architecture  %s\n", g.str("general.architecture", "?").c_str());
    std::map<uint32_t, size_t> types;
    size_t bytes = 0;
    for (const GgufTensor& t : g.tensors()) { types[t.type]++; bytes += t.nbytes; }
    std::printf("tensors       %zu (%.1f MB)", g.tensors().size(), bytes / 1e6);
    for (auto& [ty, n] : types)
        std::printf("  %s x%zu", ty == GGML_F32 ? "F32" : ty == GGML_F16 ? "F16" :
                                 ty == GGML_BF16 ? "BF16" : ("type" + std::to_string(ty)).c_str(), n);
    std::printf("\n");
    return 0;
}

static int extract(const std::string& model, const std::string& out) {
    const std::string file = find_gguf(model);
    if (file.empty()) { std::fprintf(stderr, "%s: not a GGUF model\n", model.c_str()); return 1; }
    Gguf g;
    if (!g.open(file)) { std::fprintf(stderr, "%s\n", g.error().c_str()); return 1; }
    const size_t slash = file.rfind('/');
    const std::string side = file == model ? (slash == std::string::npos ? "." : file.substr(0, slash)) : model;
    Files files;
    std::string err;
    if (!adapt_gguf(g, Sidecar{side}, files, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    mkdir(out.c_str(), 0755);
    for (auto& [rel, bytes] : files) {
        const size_t s = rel.rfind('/');
        if (s != std::string::npos) mkdir((out + "/" + rel.substr(0, s)).c_str(), 0755);
        std::ofstream f(out + "/" + rel, std::ios::binary);
        f.write(bytes.data(), (std::streamsize)bytes.size());
        if (!f) { std::fprintf(stderr, "cannot write %s/%s\n", out.c_str(), rel.c_str()); return 1; }
        std::printf("%-24s %12zu bytes\n", rel.c_str(), bytes.size());
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "info") return info(argv[2]);
    if (argc == 4 && std::string(argv[1]) == "extract") return extract(argv[2], argv[3]);
    return usage();
}
