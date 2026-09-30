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
#include <cerrno>
#include <fcntl.h>
#include <filesystem>
#include <map>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

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

static bool write_file(int root, const std::string& rel, const std::string& bytes) {
    const std::filesystem::path path(rel);
    if (path.is_absolute() || rel.empty() || rel.find('\0') != std::string::npos) return false;
    for (const auto& part : path)
        if (part == "." || part == ".." || part.empty()) return false;
    int dir = dup(root);
    if (dir < 0) return false;
    for (const auto& part : path.parent_path()) {
        if (mkdirat(dir, part.c_str(), 0755) != 0 && errno != EEXIST) { close(dir); return false; }
        const int next = openat(dir, part.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        close(dir);
        if (next < 0) return false;
        dir = next;
    }
    const int fd = openat(dir, path.filename().c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0644);
    close(dir);
    if (fd < 0) return false;
    size_t at = 0;
    while (at < bytes.size()) {
        const ssize_t n = write(fd, bytes.data() + at, bytes.size() - at);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { close(fd); return false; }
        at += (size_t)n;
    }
    return close(fd) == 0;
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
    if (mkdir(out.c_str(), 0755) != 0 && errno != EEXIST) {
        std::fprintf(stderr, "cannot create %s\n", out.c_str());
        return 1;
    }
    const int root = open(out.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (root < 0) { std::fprintf(stderr, "cannot open directory %s\n", out.c_str()); return 1; }
    for (auto& [rel, bytes] : files) {
        if (!write_file(root, rel, bytes)) {
            std::fprintf(stderr, "cannot create %s/%s (existing files and symlinks are refused)\n", out.c_str(), rel.c_str());
            close(root);
            return 1;
        }
        std::printf("%-24s %12zu bytes\n", rel.c_str(), bytes.size());
    }
    close(root);
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "info") return info(argv[2]);
    if (argc == 4 && std::string(argv[1]) == "extract") return extract(argv[2], argv[3]);
    return usage();
}
