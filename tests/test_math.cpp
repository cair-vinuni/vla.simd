#include "models/diffusion/scheduler.h"
#include "preprocess/image.h"
#include "tokenizer/t5_tokenizer.h"
#include "tokenizer/tokenizer.h"
#include "models/arena.h"
#include "models/act/act_transformer.h"
#include "models/act/resnet_backbone.h"
#include "models/diffusion/diffusion_model.h"
#include "models/smolvla/siglip_vision.h"
#include "models/smolvla/smollm2_lm.h"
#include "models/smolvla/action_expert.h"
#include "models/octo/octo_transformer.h"
#include "models/octo/diffusion_head.h"
#include "models/turbovla/dinov3_vision.h"
#include "models/turbovla/bert_text.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unistd.h>

int main(int argc, char** argv) {
    tcpu::DPConfig c;
    c.horizon = 1;
    c.action_dim = 3;
    tcpu::DPNoiseScheduler s;
    if (argc == 6) {
        c.beta_schedule = argv[1];
        c.num_train_timesteps = std::stoi(argv[2]);
        c.num_inference_steps = std::stoi(argv[3]);
        c.scheduler = std::string(argv[4]) == "DDPM" ? tcpu::DPScheduler::DDPM : tcpu::DPScheduler::DDIM;
        c.clip_sample = std::stoi(argv[5]) != 0;
        if (!s.init(c)) return 1;
        float x[] = {0.2f, -0.4f, 0.7f};
        for (size_t i = 0; i < s.timesteps.size(); i++) {
            const float factor = float(i + 1) / float(s.timesteps.size());
            const float e[] = {0.1f * factor, -0.2f * factor, 0.3f * factor};
            const float z[] = {-0.3f, 0.5f, 0.1f};
            s.step(x, e, z, s.timesteps[i], (int)i);
            std::printf("%d %.9g %.9g %.9g\n", s.timesteps[i], x[0], x[1], x[2]);
        }
        return 0;
    }
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) { std::fprintf(stderr, "%s\n", what); failures++; }
    };
    for (auto scheduler : {tcpu::DPScheduler::DDPM, tcpu::DPScheduler::DDIM}) {
        c.scheduler = scheduler;
        for (const char* schedule : {"linear", "scaled_linear", "squaredcos_cap_v2"}) {
            c.beta_schedule = schedule;
            c.num_train_timesteps = c.num_inference_steps = 1;
            check(s.init(c), "single-step init failed");
            float x[] = {0.2f, -0.4f, 0.7f};
            const float e[] = {0, 0, 0};
            s.step(x, e, nullptr, 0, 0);
            for (float v : x) check(std::isfinite(v), "single-step schedule produced nonfinite output");
            c.num_train_timesteps = 100;
            c.num_inference_steps = 7;
            check(s.init(c), "subsampled init failed");
            check(s.timesteps == std::vector<int>({84, 70, 56, 42, 28, 14, 0}), "incorrect leading spacing");
        }
    }
    c.num_inference_steps = 101;
    check(!s.init(c) && s.timesteps.empty(), "excess steps accepted");
    c.num_inference_steps = 10;
    c.beta_schedule = "linear";
    for (float beta : {0.0f, -1.0f, 1.0f, std::numeric_limits<float>::quiet_NaN()}) {
        c.beta_start = beta;
        check(!s.init(c), "invalid beta accepted");
    }
    float sentinel = 42;
    tcpu::resize_with_pad(nullptr, 1, 1, -1, &sentinel);
    check(sentinel == 42, "negative output dimension writes memory");
    uint8_t pixel[] = {0, 127, 255};
    float image[12];
    tcpu::resize_with_pad(pixel, 1, 1, 2, image);
    for (int i = 0; i < 4; i++) {
        check(image[i] == -1 && image[8+i] == 1, "constant image resize");
        check(std::fabs(image[4+i] - (127.0/255*2-1)) < 1e-7, "image normalization");
    }
    std::string temp = (std::filesystem::temp_directory_path() / "vla_tokenizer_XXXXXX").string();
    if (!mkdtemp(temp.data())) return 1;
    const char* path = temp.c_str();
    const std::filesystem::path root(path);
    check(!tcpu::shape_fits({INT_MAX, 2}) && !tcpu::shape_fits({1, -1}) &&
          tcpu::shape_fits({32, 64}), "invalid shape validation");
    for (double value : {1.5, 1e100, -1e100, std::numeric_limits<double>::infinity()}) {
        bool rejected = false;
        try { tcpu::metadata_int(value); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "invalid integer metadata accepted");
    }
    for (const char* metadata : {"heads 0\n", "n_enc -1\n", "dim 2147483647\n", "chunk invalid\n"}) {
        std::ofstream(root / "act.meta") << metadata;
        std::ofstream(root / "act.bin", std::ios::binary).put(0);
        tcpu::ActTransformer model;
        check(!model.load(path, 512, 0), "invalid ACT metadata accepted");
    }
    {
        std::vector<float> weights(64*7*7*3+64);
        std::ofstream bin(root / "backbone.bin", std::ios::binary);
        bin.write((const char*)weights.data(), weights.size()*sizeof(float));
        bin.close();
        for (const char* stride : {"2", "0", "2.5", "2garbage"}) {
            std::ofstream(root / "backbone.meta") << "stem_stride " << stride << '\n';
            tcpu::ResNetBackbone model;
            check(model.load(path) == (std::string(stride) == "2"), "convolution stride validation");
        }
    }
    for (const char* metadata : {"scheduler DDM\n", "film_scale invalid\n", "down_dims 32 garbage\n",
                                 "n_cams 2147483647\n", "crop_h -1\n"}) {
        std::ofstream(root / "diffusion.meta") << metadata;
        tcpu::DiffusionModel model;
        check(!model.load(path), "invalid diffusion metadata accepted");
    }
    auto rejects = [&](const char* file, const std::string& field, auto load) {
        for (const char* value : {"1e100", "1.5", "invalid"}) {
            std::ofstream(root / file) << field << ' ' << value << '\n';
            bool rejected = false;
            try { rejected = !load(); } catch (const std::invalid_argument&) { rejected = true; }
            check(rejected, "malformed numeric metadata accepted");
        }
    };
    rejects("vit.meta", "hidden", [&] { return tcpu::SiglipVision{}.load(path); });
    rejects("vlm.meta", "hidden", [&] { return tcpu::SmollmVlm{}.load(path); });
    rejects("aex.meta", "expert_h", [&] { return tcpu::ActionExpert{}.load(path); });
    rejects("octo.meta", "d", [&] { return tcpu::OctoTransformer{}.load(path); });
    rejects("head.meta", "hidden", [&] { return tcpu::DiffusionHead{}.load(path); });
    rejects("vision.meta", "hidden", [&] { return tcpu::Dinov3Vision{}.load(path, 256); });
    rejects("text.meta", "hidden", [&] { return tcpu::BertText{}.load(path, 256); });
    {
        std::ofstream(root / "vocab.txt") << "0\ta\n1\tb\n2\tab\n";
        std::ofstream(root / "merges.txt") << "a b\n";
        tcpu::Tokenizer tok;
        check(tok.load(path) && tok.encode("ab") == std::vector<int>({2}), "BPE encode");
        std::ofstream(root / "vocab.txt") << "3\tc\n4\td\n5\tcd\n";
        std::ofstream(root / "merges.txt") << "c d\n";
        check(tok.load(path) && !tok.vocab.count("a") && tok.ranks.size() == 1, "BPE reload retains vocabulary");
        std::ofstream(root / "vocab.txt") << "<pad>\t0\n</s>\t0\n<unk>\t0\n▁a\t-1\n";
        tcpu::T5Tokenizer t5;
        check(t5.load(path) && t5.encode("a", 2) == std::vector<int>({3, 1}), "T5 encode");
        std::ofstream(root / "vocab.txt") << "<pad>\t0\n</s>\t0\n<unk>\t0\n▁b\t-1\n";
        check(t5.load(path) && !t5.piece_id.count("▁a") && t5.scores.size() == 4, "T5 reload retains vocabulary");
    }
    std::filesystem::remove_all(root);
    return failures ? 1 : 0;
}
