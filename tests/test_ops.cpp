#include "hal/arch.h"
#include "hal/common/layout.h"
#include "ops/conv_ops.h"
#include "ops/lm_ops.h"
#include "ops/quant_ops.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

static uint64_t state = 0x243F6A8885A308D3ull;
static float frand(float lo, float hi) {
    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
    return lo + (hi - lo)*(float)((state >> 40) & 0xFFFFFF)/16777216.0f;
}
static std::vector<float> rv(size_t n, float lo = -1.f, float hi = 1.f) {
    std::vector<float> v(n);
    for (float& x : v) x = frand(lo, hi);
    return v;
}
static bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

static void naive_linear(std::vector<double>& y, std::vector<double>& mag, const float* x,
                         const float* W, const float* b, int seq, int N, int K) {
    y.assign((size_t)seq*N, 0.0); mag.assign((size_t)seq*N, 0.0);
    for (int t = 0; t < seq; t++)
        for (int n = 0; n < N; n++) {
            double s = b ? b[n] : 0.0, m = b ? std::fabs(b[n]) : 0.0;
            for (int k = 0; k < K; k++) {
                const double p = (double)x[(size_t)t*K+k]*W[(size_t)n*K+k];
                s += p; m += std::fabs(p);
            }
            y[(size_t)t*N+n] = s; mag[(size_t)t*N+n] = m;
        }
}

static void expect_close(const char* what, const std::vector<float>& got, const std::vector<double>& ref,
                         const std::vector<double>& mag, double rel) {
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); i++) {
        const double e = std::fabs(got[i] - ref[i])/(mag[i] + 1e-30);
        if (e > worst) worst = e;
    }
    CHECK(worst <= rel, "%s: worst scaled error %.3e > %.1e", what, worst, rel);
}

static void test_linear() {
    const int shapes[][3] = {{1, 16, 3}, {7, 48, 64}, {50, 768, 1027}, {13, 64, 17}, {257, 96, 48}};
    for (auto& s : shapes) {
        const int seq = s[0], N = s[1], K = s[2];
        auto x = rv((size_t)seq*K), W = rv((size_t)N*K, -.2f, .2f), b = rv(N);
        std::vector<double> ref, mag;
        naive_linear(ref, mag, x.data(), W.data(), b.data(), seq, N, K);
        std::vector<float> y((size_t)seq*N);
        dense_linear(y.data(), x.data(), W.data(), b.data(), seq, N, K);
        expect_close("dense_linear", y, ref, mag, 1e-5);

        if (N % 16) continue;
        std::vector<float> Wp((size_t)N*K), yp((size_t)seq*N);
        pack_weights16(W.data(), Wp.data(), N, K);
        dense_linear_packed(yp.data(), x.data(), Wp.data(), b.data(), seq, N, K);
        expect_close("dense_linear_packed", yp, ref, mag, 1e-5);

        const int ldo = hal::kt_stride(seq);
        std::vector<float> kt((size_t)N*ldo, 0.f), ktt((size_t)seq*N);
        dense_linear_packed_kt(kt.data(), x.data(), Wp.data(), b.data(), seq, N, K, ldo);
        for (int n = 0; n < N; n++)
            for (int t = 0; t < seq; t++) ktt[(size_t)t*N+n] = kt[(size_t)n*ldo+t];
        CHECK(same_bits(ktt, yp), "dense_linear_packed_kt != dense_linear_packed (%d,%d,%d)", seq, N, K);

#if TCPU_ISA_X86
        std::vector<float> g((size_t)seq*N), gg(yp);
        dense_linear_packed_gelu(g.data(), x.data(), Wp.data(), b.data(), seq, N, K);
        gelu_tanh(gg.data(), seq*N);
        CHECK(same_bits(g, gg), "dense_linear_packed_gelu != gelu_tanh(packed) (%d,%d,%d)", seq, N, K);

        auto acc = rv((size_t)seq*N), sum(acc);
        dense_linear_packed_add(acc.data(), x.data(), Wp.data(), b.data(), seq, N, K);
        for (size_t i = 0; i < sum.size(); i++) sum[i] += yp[i];
        CHECK(same_bits(acc, sum), "dense_linear_packed_add != out + packed (%d,%d,%d)", seq, N, K);
#endif

        std::vector<int8_t> Wq(packed_i8_words(N, K)*4), xq((size_t)seq*i8_kpad(K));
        std::vector<float> ws(N), as(seq), yq((size_t)seq*N), yr((size_t)seq*N);
        pack_weights_i8(W.data(), Wq.data(), ws.data(), N, K);
        quantize_act_i8(x.data(), xq.data(), as.data(), seq, K);
        dense_linear_i8_ref(yr.data(), xq.data(), as.data(), Wq.data(), ws.data(), b.data(), seq, N, K);
        expect_close("dense_linear_i8_ref", yr, ref, mag, 2e-2);
        if (int8_gemm_available()) {
            dense_linear_i8_pre(yq.data(), xq.data(), as.data(), Wq.data(), ws.data(), b.data(), seq, N, K);
            CHECK(same_bits(yq, yr), "dense_linear_i8_pre != dense_linear_i8_ref (%d,%d,%d)", seq, N, K);
        }
    }
}

static void naive_attention(std::vector<double>& out, const float* Q, const float* K, const float* V,
                            const float* mask, int sq, int sk, int nq, int nkv, int hd, float scale) {
    out.assign((size_t)sq*nq*hd, 0.0);
    std::vector<double> s(sk);
    for (int h = 0; h < nq; h++) {
        const int g = h/(nq/nkv);
        for (int i = 0; i < sq; i++) {
            double mx = -INFINITY;
            for (int j = 0; j < sk; j++) {
                double d = 0.0;
                for (int c = 0; c < hd; c++)
                    d += (double)Q[((size_t)i*nq+h)*hd+c]*K[((size_t)j*nkv+g)*hd+c];
                s[j] = d*scale + (mask ? mask[(size_t)i*sk+j] : 0.0);
                if (s[j] > mx) mx = s[j];
            }
            double z = 0.0;
            for (int j = 0; j < sk; j++) { s[j] = std::isinf(s[j]) ? 0.0 : std::exp(s[j]-mx); z += s[j]; }
            for (int c = 0; c < hd; c++) {
                double a = 0.0;
                for (int j = 0; j < sk; j++) a += s[j]*V[((size_t)j*nkv+g)*hd+c];
                out[((size_t)i*nq+h)*hd+c] = a/z;
            }
        }
    }
}

static void test_attention() {
    const int cfgs[][5] = {{1, 50, 8, 2, 64}, {50, 50, 8, 2, 64}, {50, 113, 15, 5, 64},
                           {17, 33, 4, 4, 72}, {9, 20, 2, 1, 256}, {256, 256, 12, 12, 64}};
    const float NINF = -std::numeric_limits<float>::infinity();
    for (auto& c : cfgs) {
        const int sq = c[0], sk = c[1], nq = c[2], nkv = c[3], hd = c[4];
        auto Q = rv((size_t)sq*nq*hd), K = rv((size_t)sk*nkv*hd), V = rv((size_t)sk*nkv*hd);
        const float scale = 1.0f/std::sqrt((float)hd);
        std::vector<float> dense((size_t)sq*nq*hd), masked(dense.size()), zero((size_t)sq*sk, 0.f);
        gqa_attention_dense(dense.data(), Q.data(), K.data(), V.data(), sq, sk, nq, nkv, hd, scale);
        gqa_attention_masked(masked.data(), Q.data(), K.data(), V.data(), sq, sk, nq, nkv, hd, scale, zero.data());
        CHECK(same_bits(dense, masked), "dense != masked(zero) (%d,%d,%d,%d,%d)", sq, sk, nq, nkv, hd);

        std::vector<double> ref;
        naive_attention(ref, Q.data(), K.data(), V.data(), nullptr, sq, sk, nq, nkv, hd, scale);
        std::vector<double> one(ref.size(), 1.0);
        expect_close("gqa_attention_dense", dense, ref, one, 2e-6);

        std::vector<float> causal((size_t)sq*sk);
        for (int i = 0; i < sq; i++)
            for (int j = 0; j < sk; j++) causal[(size_t)i*sk+j] = j > i + (sk - sq) ? NINF : 0.f;
        gqa_attention_masked(masked.data(), Q.data(), K.data(), V.data(), sq, sk, nq, nkv, hd, scale, causal.data());
        naive_attention(ref, Q.data(), K.data(), V.data(), causal.data(), sq, sk, nq, nkv, hd, scale);
        expect_close("gqa_attention_masked causal", masked, ref, one, 2e-6);

        const int ldk = hal::kt_stride(sk);
        std::vector<float> KT((size_t)nkv*hd*ldk, 0.f), pre(masked.size());
        for (int h = 0; h < nkv; h++)
            for (int d = 0; d < hd; d++)
                for (int j = 0; j < sk; j++) KT[((size_t)h*hd+d)*ldk+j] = K[((size_t)j*nkv+h)*hd+d];
        gqa_attention_masked(pre.data(), Q.data(), K.data(), V.data(), sq, sk, nq, nkv, hd, scale,
                             causal.data(), KT.data());
        CHECK(same_bits(pre, masked), "masked with K_pre != masked (%d,%d,%d,%d,%d)", sq, sk, nq, nkv, hd);
    }
}

static void test_eltwise() {
    const int n = 20011;
    auto x = rv(n, -30.f, 30.f);
    auto run = [&](void (*f)(float*, int), double (*r)(double), double tol, const char* name) {
        std::vector<float> y(x);
        f(y.data(), n);
        double worst = 0.0;
        for (int i = 0; i < n; i++) {
            const double e = std::fabs(y[i] - r(x[i]))/std::max(std::fabs(r(x[i])), 1.0);
            if (e > worst) worst = e;
        }
        CHECK(worst <= tol, "%s: worst error %.3e > %.1e", name, worst, tol);
        float nan = std::numeric_limits<float>::quiet_NaN();
        f(&nan, 1);
        CHECK(std::isnan(nan), "%s(NaN) is not NaN", name);
    };
    run(silu, [](double v) { return v/(1.0 + std::exp(-v)); }, 2e-6, "silu");
    run(gelu_tanh, [](double v) { return 0.5*v*(1.0 + std::tanh(0.7978845608028654*(v + 0.044715*v*v*v))); },
        2e-6, "gelu_tanh");
    run(gelu_erf, [](double v) { return 0.5*v*(1.0 + std::erf(v/std::sqrt(2.0))); }, 2e-6, "gelu_erf");
    run(mish, [](double v) { return v > 20.0 ? v : v*std::tanh(std::log1p(std::exp(v))); }, 2e-6, "mish");
}

static void test_norms() {
    const int shapes[][2] = {{1, 768}, {50, 960}, {7, 6}};
    for (auto& s : shapes) {
        const int seq = s[0], H = s[1];
        auto x = rv((size_t)seq*H, -3.f, 5.f), w = rv(H), b = rv(H);
        std::vector<float> y((size_t)seq*H), z((size_t)seq*H);
        rmsnorm(y.data(), x.data(), w.data(), seq, H, 1e-6f);
        layernorm(z.data(), x.data(), w.data(), b.data(), seq, H, 1e-5f);
        double wr = 0.0, wl = 0.0;
        for (int t = 0; t < seq; t++) {
            double ss = 0.0, mu = 0.0, var = 0.0;
            for (int i = 0; i < H; i++) { const double v = x[(size_t)t*H+i]; ss += v*v; mu += v; }
            mu /= H;
            for (int i = 0; i < H; i++) { const double d = x[(size_t)t*H+i] - mu; var += d*d; }
            const double r = 1.0/std::sqrt(ss/H + 1e-6), l = 1.0/std::sqrt(var/H + 1e-5);
            for (int i = 0; i < H; i++) {
                const size_t k = (size_t)t*H+i;
                wr = std::max(wr, std::fabs(y[k] - x[k]*r*w[i]));
                wl = std::max(wl, std::fabs(z[k] - ((x[k] - mu)*l*w[i] + b[i])));
            }
        }
        CHECK(wr <= 1e-5 && wl <= 1e-5, "norms (%d,%d): rms %.3e ln %.3e", seq, H, wr, wl);
    }
}

static void test_conv() {
    const int cfgs[][7] = {{30, 30, 3, 64, 7, 2, 3}, {28, 28, 64, 64, 3, 1, 1}, {15, 15, 64, 128, 3, 2, 1}};
    for (auto& c : cfgs) {
        const int H = c[0], Wd = c[1], Cin = c[2], Cout = c[3], k = c[4], st = c[5], pd = c[6];
        const int Ho = (H + 2*pd - k)/st + 1, Wo = (Wd + 2*pd - k)/st + 1, K = k*k*Cin;
        auto x = rv((size_t)H*Wd*Cin), W = rv((size_t)Cout*K, -.1f, .1f), b = rv(Cout);
        std::vector<float> y((size_t)Ho*Wo*Cout), yp(y.size()), Wp((size_t)Cout*K);
        conv2d(y.data(), x.data(), W.data(), b.data(), H, Wd, Cin, Cout, k, st, pd);
        pack_weights16(W.data(), Wp.data(), Cout, K);
        conv2d_packed(yp.data(), x.data(), Wp.data(), b.data(), H, Wd, Cin, Cout, k, st, pd);
        double worst = 0.0;
        for (int oy = 0; oy < Ho; oy++)
            for (int ox = 0; ox < Wo; ox++)
                for (int co = 0; co < Cout; co++) {
                    double s = b[co], m = std::fabs(b[co]);
                    for (int ky = 0; ky < k; ky++)
                        for (int kx = 0; kx < k; kx++) {
                            const int iy = oy*st - pd + ky, ix = ox*st - pd + kx;
                            if (iy < 0 || iy >= H || ix < 0 || ix >= Wd) continue;
                            for (int ci = 0; ci < Cin; ci++) {
                                const double p = (double)x[((size_t)iy*Wd+ix)*Cin+ci]*
                                                 W[(size_t)co*K + ((size_t)ky*k+kx)*Cin + ci];
                                s += p; m += std::fabs(p);
                            }
                        }
                    const size_t o = ((size_t)oy*Wo+ox)*Cout+co;
                    worst = std::max(worst, std::max(std::fabs(y[o] - s), std::fabs(yp[o] - s))/(m + 1e-30));
                }
        CHECK(worst <= 1e-5, "conv2d (%d,%d,%d,%d): scaled error %.3e", H, Cin, Cout, k, worst);
    }

    const int T = 8, Cin = 64, Cout = 32, k = 4, st = 2, pd = 1, To = (T - 1)*st - 2*pd + k;
    auto x = rv((size_t)T*Cin), W = rv((size_t)Cin*k*Cout, -.1f, .1f), b = rv(Cout);
    std::vector<float> y((size_t)To*Cout);
    conv_transpose1d(y.data(), x.data(), W.data(), b.data(), T, Cin, Cout, k, st, pd);
    double worst = 0.0;
    for (int o = 0; o < To; o++)
        for (int co = 0; co < Cout; co++) {
            double s = b[co], m = std::fabs(b[co]);
            for (int t = 0; t < T; t++) {
                const int kt = o + pd - t*st;
                if (kt < 0 || kt >= k) continue;
                for (int ci = 0; ci < Cin; ci++) {
                    const double p = (double)x[(size_t)t*Cin+ci]*W[((size_t)ci*k+kt)*Cout+co];
                    s += p; m += std::fabs(p);
                }
            }
            worst = std::max(worst, std::fabs(y[(size_t)o*Cout+co] - s)/(m + 1e-30));
        }
    CHECK(worst <= 1e-5, "conv_transpose1d: scaled error %.3e", worst);

    const int P = 64, C = 256, G = 8;
    auto gx = rv((size_t)P*C, -2.f, 3.f), gs = rv(C), gb = rv(C);
    std::vector<float> g1((size_t)P*C), g2((size_t)P*C);
    groupnorm(g1.data(), gx.data(), gs.data(), gb.data(), P, C, G, 1e-5f, false);
    groupnorm(g2.data(), gx.data(), gs.data(), gb.data(), P, C, G, 1e-5f, false);
    CHECK(same_bits(g1, g2), "groupnorm is not deterministic");
    double gw = 0.0;
    const int gc = C/G;
    for (int g = 0; g < G; g++) {
        double mu = 0.0, var = 0.0;
        for (int p = 0; p < P; p++) for (int c = g*gc; c < (g+1)*gc; c++) mu += gx[(size_t)p*C+c];
        mu /= (double)P*gc;
        for (int p = 0; p < P; p++) for (int c = g*gc; c < (g+1)*gc; c++) {
            const double d = gx[(size_t)p*C+c] - mu; var += d*d;
        }
        const double inv = 1.0/std::sqrt(var/((double)P*gc) + 1e-5);
        for (int p = 0; p < P; p++) for (int c = g*gc; c < (g+1)*gc; c++)
            gw = std::max(gw, std::fabs(g1[(size_t)p*C+c] - ((gx[(size_t)p*C+c] - mu)*inv*gs[c] + gb[c])));
    }
    CHECK(gw <= 1e-5, "groupnorm error %.3e", gw);
}

int main() {
    test_linear();
    test_attention();
    test_eltwise();
    test_norms();
    test_conv();
    if (failures) std::fprintf(stderr, "%d check(s) failed on %s\n", failures, hal::backend_name());
    else std::printf("ok (%s)\n", hal::backend_name());
    return failures ? 1 : 0;
}
