/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <istream>
#include <memory>
#include <string>

// Model file access. The loaders read a converted directory: <dir>/<name>.meta
// and .bin files. A vla.cpp GGUF is loaded through the same loaders by mounting
// it: an adapter regenerates, in memory, the exact files the Python converter
// would have written from the same checkpoint, and open() serves them under the
// GGUF's path. A file the GGUF does not carry (a tokenizer, statistics) is read
// from disk beside it, as a sidecar.
//
// A model path mounts as a GGUF when it is a .gguf file, or a directory holding
// exactly one .gguf and no .meta files. Anything else is a plain directory.

namespace tcpu {
namespace io {

// Drop-in for std::ifstream: a mounted GGUF's generated file, else the file
// on disk. A missing file yields a stream that tests false.
class InFile : public std::istream {
public:
    explicit InFile(const std::string& path, std::ios::openmode mode = std::ios::in);
private:
    std::unique_ptr<std::streambuf> buf;
};

// Mounts are refcounted per path, so nested loaders (a C API wrapping a model
// wrapping its sub-modules) share one parse. The generated files are dropped
// when the last Mount goes.
class Mount {
public:
    explicit Mount(const std::string& path);
    ~Mount();
    Mount(const Mount&) = delete;
    Mount& operator=(const Mount&) = delete;

    bool ok() const { return good; }     // false: a GGUF that failed to parse or adapt
    bool gguf() const { return mounted; }
private:
    std::string key;
    bool good = true, mounted = false;
};

// Where a model path's GGUF is, if it is one ("" otherwise).
std::string find_gguf(const std::string& path);

} // namespace io
} // namespace tcpu
