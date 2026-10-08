// Copyright FunASR (https://github.com/modelscope/FunASR).
// Licensed under the MIT License.
//
// The frontend and FSMN graph in this file are derived from FunASR's
// runtime/llama.cpp/funasr-common/funasr_vad.h (MIT), pinned by EchoASR's
// FSMN-VAD governance manifest. audio.cpp owns model loading and inference;
// callers own product endpoint and segmentation policy.

#include "engine/models/fsmn_vad/session.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::fsmn_vad {
namespace {

constexpr int kSampleRate = 16000;
constexpr std::size_t kWindowFrames = 400;
constexpr std::size_t kShiftFrames = 160;
constexpr int kFftSize = 512;
constexpr int kMelBins = 80;
constexpr float kPreemphasis = 0.97f;
constexpr float kLowFrequency = 20.0f;
constexpr float kHighFrequency = 8000.0f;
constexpr float kPi = 3.14159265358979323846f;
constexpr std::size_t kWeightContextBytes = 4U * 1024U * 1024U;

struct OperationControl {
    std::shared_ptr<const runtime::TaskExecutionControl> execution_control;
};

struct VadOptions {
    double threshold = 0.5;
    int min_speech_duration_ms = 250;
    int min_silence_duration_ms = 100;
    int speech_padding_ms = 30;
    double max_speech_duration_seconds = 0.0;
};

struct FrameRange {
    std::size_t begin_frame = 0;
    std::size_t end_frame = 0;
};

struct RealtimeVadObservation {
    std::size_t begin_frame = 0;
    std::size_t end_frame = 0;
    bool speech = false;
};

void throw_if_cancelled(const OperationControl &control) {
    runtime::throw_if_task_cancelled(control.execution_control);
}

float mel_frequency(float frequency) {
    return 1127.0f * std::log1p(frequency / 700.0f);
}

void fft(std::vector<float> &real, std::vector<float> &imaginary) {
    for (int index = 1, reversed = 0; index < kFftSize; ++index) {
        int bit = kFftSize >> 1;
        for (; reversed & bit; bit >>= 1) reversed ^= bit;
        reversed ^= bit;
        if (index < reversed) {
            std::swap(real[index], real[reversed]);
            std::swap(imaginary[index], imaginary[reversed]);
        }
    }
    for (int length = 2; length <= kFftSize; length <<= 1) {
        const float angle = -2.0f * kPi / static_cast<float>(length);
        const float step_real = std::cos(angle);
        const float step_imaginary = std::sin(angle);
        for (int begin = 0; begin < kFftSize; begin += length) {
            float current_real = 1.0f;
            float current_imaginary = 0.0f;
            for (int offset = 0; offset < length / 2; ++offset) {
                const int left = begin + offset;
                const int right = left + length / 2;
                const float left_real = real[left];
                const float left_imaginary = imaginary[left];
                const float right_real =
                    real[right] * current_real -
                    imaginary[right] * current_imaginary;
                const float right_imaginary =
                    real[right] * current_imaginary +
                    imaginary[right] * current_real;
                real[left] = left_real + right_real;
                imaginary[left] = left_imaginary + right_imaginary;
                real[right] = left_real - right_real;
                imaginary[right] = left_imaginary - right_imaginary;
                const float next_real =
                    current_real * step_real -
                    current_imaginary * step_imaginary;
                current_imaginary =
                    current_real * step_imaginary +
                    current_imaginary * step_real;
                current_real = next_real;
            }
        }
    }
}

class FbankFrontend {
public:
    FbankFrontend() {
        window_.resize(kWindowFrames);
        for (std::size_t index = 0; index < kWindowFrames; ++index) {
            window_[index] = 0.54f - 0.46f * std::cos(
                2.0f * kPi * static_cast<float>(index) /
                static_cast<float>(kWindowFrames - 1));
        }
        constexpr int spectrum_bins = kFftSize / 2 + 1;
        filters_.assign(kMelBins, std::vector<float>(spectrum_bins));
        const float bin_width =
            static_cast<float>(kSampleRate) / static_cast<float>(kFftSize);
        const float low_mel = mel_frequency(kLowFrequency);
        const float high_mel = mel_frequency(kHighFrequency);
        const float mel_step = (high_mel - low_mel) / (kMelBins + 1);
        for (int mel = 0; mel < kMelBins; ++mel) {
            const float left = low_mel + mel * mel_step;
            const float center = low_mel + (mel + 1) * mel_step;
            const float right = low_mel + (mel + 2) * mel_step;
            for (int bin = 0; bin < spectrum_bins; ++bin) {
                const float value = mel_frequency(bin_width * bin);
                if (value > left && value < right) {
                    filters_[mel][bin] = value <= center
                        ? (value - left) / (center - left)
                        : (right - value) / (right - center);
                }
            }
        }
    }

    std::vector<float> compute(const float *samples) const {
        std::vector<float> frame(kWindowFrames);
        double mean = 0.0;
        for (std::size_t index = 0; index < kWindowFrames; ++index) {
            frame[index] = samples[index] * 32768.0f;
            mean += frame[index];
        }
        mean /= static_cast<double>(kWindowFrames);
        for (auto &value : frame) value -= static_cast<float>(mean);
        for (std::size_t index = kWindowFrames - 1; index > 0; --index) {
            frame[index] -= kPreemphasis * frame[index - 1];
        }
        frame[0] -= kPreemphasis * frame[0];

        std::vector<float> real(kFftSize);
        std::vector<float> imaginary(kFftSize);
        for (std::size_t index = 0; index < kWindowFrames; ++index) {
            real[index] = frame[index] * window_[index];
        }
        fft(real, imaginary);

        std::vector<float> features(kMelBins);
        constexpr float floor = 1.1920929e-07f;
        for (int mel = 0; mel < kMelBins; ++mel) {
            float energy = 0.0f;
            for (int bin = 0; bin <= kFftSize / 2; ++bin) {
                const float weight = filters_[mel][bin];
                if (weight > 0.0f) {
                    energy += weight *
                        (real[bin] * real[bin] +
                         imaginary[bin] * imaginary[bin]);
                }
            }
            features[mel] = std::log(std::max(energy, floor));
        }
        return features;
    }

private:
    std::vector<float> window_;
    std::vector<std::vector<float>> filters_;
};

struct ModelAssets {
    explicit ModelAssets(const std::filesystem::path &path) {
        ggml_context *metadata_context = nullptr;
        gguf_init_params parameters{true, &metadata_context};
        gguf_context *gguf = gguf_init_from_file(path.c_str(), parameters);
        if (gguf == nullptr || metadata_context == nullptr) {
            if (gguf != nullptr) gguf_free(gguf);
            if (metadata_context != nullptr) ggml_free(metadata_context);
            throw std::runtime_error("failed to load FSMN-VAD GGUF model");
        }
        auto read_u32 = [&](const char *key, int fallback) {
            const int index = gguf_find_key(gguf, key);
            return index < 0
                ? fallback
                : static_cast<int>(gguf_get_val_u32(gguf, index));
        };
        input_dim = read_u32("vad.input_dim", 400);
        projection_dim = read_u32("vad.proj_dim", 128);
        layer_count = read_u32("vad.fsmn_layers", 4);
        left_order = read_u32("vad.lorder", 20);
        output_dim = read_u32("vad.output_dim", 248);
        lfr_m = read_u32("vad.lfr_m", 5);
        lfr_n = read_u32("vad.lfr_n", 1);
        gguf_free(gguf);
        ggml_free(metadata_context);
        if (input_dim != 400 || projection_dim != 128 || layer_count != 4 ||
            left_order != 20 || output_dim != 248 || lfr_m != 5 || lfr_n != 1) {
            throw std::runtime_error(
                "unsupported FSMN-VAD GGUF architecture metadata");
        }
        source = engine::assets::open_tensor_source(path);
        cmvn_shift = source->require_f32("cmvn.shift", {input_dim});
        cmvn_scale = source->require_f32("cmvn.scale", {input_dim});
    }

    std::shared_ptr<const engine::assets::TensorSource> source;
    std::vector<float> cmvn_shift;
    std::vector<float> cmvn_scale;
    int input_dim = 0;
    int projection_dim = 0;
    int layer_count = 0;
    int left_order = 0;
    int output_dim = 0;
    int lfr_m = 0;
    int lfr_n = 0;
};

struct ModelData {
    ModelData(
        std::shared_ptr<const ModelAssets> assets,
        engine::core::ExecutionContext &execution_context)
        : assets(std::move(assets)),
          store(std::make_shared<engine::core::BackendWeightStore>(
              execution_context.backend(),
              execution_context.backend_type(),
              "FSMN-VAD weights",
              kWeightContextBytes)),
          input_dim(this->assets->input_dim),
          projection_dim(this->assets->projection_dim),
          layer_count(this->assets->layer_count),
          left_order(this->assets->left_order),
          output_dim(this->assets->output_dim),
          lfr_m(this->assets->lfr_m),
          lfr_n(this->assets->lfr_n) {
        const auto &source = *this->assets->source;
        auto load = [&](const std::string &name,
                        std::initializer_list<int64_t> shape) {
            tensors.emplace(
                name,
                store->load_f32_tensor(source, name, shape).tensor);
        };
        load("encoder.in_linear1.linear.weight", {140, input_dim});
        load("encoder.in_linear1.linear.bias", {140});
        load("encoder.in_linear2.linear.weight", {250, 140});
        load("encoder.in_linear2.linear.bias", {250});
        for (int layer = 0; layer < layer_count; ++layer) {
            const std::string prefix =
                "encoder.fsmn." + std::to_string(layer) + ".";
            load(prefix + "linear.linear.weight", {projection_dim, 250});
            load(prefix + "fsmn_block.conv_left.weight",
                 {left_order, projection_dim});
            load(prefix + "affine.linear.weight", {250, projection_dim});
            load(prefix + "affine.linear.bias", {250});
        }
        load("encoder.out_linear1.linear.weight", {140, 250});
        load("encoder.out_linear1.linear.bias", {140});
        load("encoder.out_linear2.linear.weight", {output_dim, 140});
        load("encoder.out_linear2.linear.bias", {output_dim});
        store->upload();
    }

    ggml_tensor *get(const std::string &name) const {
        const auto found = tensors.find(name);
        if (found == tensors.end() || found->second == nullptr) {
            throw std::runtime_error("FSMN-VAD GGUF tensor is unavailable: " + name);
        }
        return found->second;
    }

    std::shared_ptr<const ModelAssets> assets;
    std::shared_ptr<engine::core::BackendWeightStore> store;
    std::map<std::string, ggml_tensor *> tensors;
    int input_dim = 0;
    int projection_dim = 0;
    int layer_count = 0;
    int left_order = 0;
    int output_dim = 0;
    int lfr_m = 0;
    int lfr_n = 0;
};

ggml_tensor *linear(
    ggml_context *context,
    ggml_tensor *weight,
    ggml_tensor *bias,
    ggml_tensor *input) {
    auto *output = ggml_mul_mat(context, weight, input);
    return bias == nullptr ? output : ggml_add(context, output, bias);
}

struct InferenceResult {
    std::vector<float> speech_scores;
    std::vector<std::vector<float>> caches;
};

InferenceResult infer(
    const ModelData &model,
    const std::vector<float> &features,
    const std::vector<std::vector<float>> &input_caches,
    ggml_backend_t backend,
    const OperationControl &control) {
    throw_if_cancelled(control);
    if (features.empty() || features.size() % model.input_dim != 0) {
        throw std::invalid_argument("FSMN-VAD inference features are invalid");
    }
    const std::size_t frames = features.size() / model.input_dim;
    const std::size_t cache_frames = model.left_order - 1;
    const std::size_t cache_values = cache_frames * model.projection_dim;
    if (input_caches.size() != static_cast<std::size_t>(model.layer_count) ||
        std::any_of(
            input_caches.begin(), input_caches.end(),
            [cache_values](const auto &cache) {
                return cache.size() != cache_values;
            })) {
        throw std::invalid_argument("FSMN-VAD layer caches are invalid");
    }

    ggml_init_params parameters{32U * 1024U * 1024U, nullptr, true};
    ggml_context *context = ggml_init(parameters);
    if (context == nullptr) {
        throw std::bad_alloc();
    }
    ggml_gallocr_t allocator = nullptr;

    try {
        auto *input = ggml_new_tensor_2d(
            context, GGML_TYPE_F32, model.input_dim, frames);
        ggml_set_input(input);
        auto *hidden = linear(
            context,
            model.get("encoder.in_linear1.linear.weight"),
            model.get("encoder.in_linear1.linear.bias"),
            input);
        hidden = linear(
            context,
            model.get("encoder.in_linear2.linear.weight"),
            model.get("encoder.in_linear2.linear.bias"),
            hidden);
        hidden = ggml_relu(context, hidden);

        std::vector<ggml_tensor *> cache_inputs;
        std::vector<ggml_tensor *> cache_outputs;
        for (int layer = 0; layer < model.layer_count; ++layer) {
            const std::string prefix =
                "encoder.fsmn." + std::to_string(layer) + ".";
            auto *projected = ggml_mul_mat(
                context, model.get(prefix + "linear.linear.weight"), hidden);
            auto *cache = ggml_new_tensor_2d(
                context,
                GGML_TYPE_F32,
                model.projection_dim,
                cache_frames);
            ggml_set_input(cache);
            cache_inputs.push_back(cache);
            auto *history = ggml_concat(context, cache, projected, 1);
            auto *accumulated = projected;
            auto *kernel = model.get(prefix + "fsmn_block.conv_left.weight");
            for (int order = 0; order < model.left_order; ++order) {
                auto *slice = ggml_view_2d(
                    context,
                    history,
                    model.projection_dim,
                    frames,
                    history->nb[1],
                    static_cast<std::size_t>(order) * history->nb[1]);
                auto *weight = ggml_view_1d(
                    context,
                    kernel,
                    model.projection_dim,
                    static_cast<std::size_t>(order) * kernel->nb[1]);
                accumulated = ggml_add(
                    context,
                    accumulated,
                    ggml_mul(context, slice, weight));
            }
            hidden = linear(
                context,
                model.get(prefix + "affine.linear.weight"),
                model.get(prefix + "affine.linear.bias"),
                accumulated);
            hidden = ggml_relu(context, hidden);
            auto *next_cache = ggml_cont(
                context,
                ggml_view_2d(
                    context,
                    history,
                    model.projection_dim,
                    cache_frames,
                    history->nb[1],
                    frames * history->nb[1]));
            ggml_set_output(next_cache);
            cache_outputs.push_back(next_cache);
        }
        hidden = linear(
            context,
            model.get("encoder.out_linear1.linear.weight"),
            model.get("encoder.out_linear1.linear.bias"),
            hidden);
        hidden = linear(
            context,
            model.get("encoder.out_linear2.linear.weight"),
            model.get("encoder.out_linear2.linear.bias"),
            hidden);
        auto *probabilities = ggml_soft_max(context, hidden);
        ggml_set_output(probabilities);

        ggml_cgraph *graph = ggml_new_graph(context);
        ggml_build_forward_expand(graph, probabilities);
        for (auto *cache_output : cache_outputs) {
            ggml_build_forward_expand(graph, cache_output);
        }
        allocator = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(allocator, graph)) {
            ggml_gallocr_free(allocator);
            throw std::runtime_error("failed to allocate FSMN-VAD graph");
        }
        ggml_backend_tensor_set(input, features.data(), 0, ggml_nbytes(input));
        for (std::size_t layer = 0; layer < cache_inputs.size(); ++layer) {
            ggml_backend_tensor_set(
                cache_inputs[layer],
                input_caches[layer].data(),
                0,
                ggml_nbytes(cache_inputs[layer]));
        }
        throw_if_cancelled(control);
        const auto status = engine::core::compute_backend_graph(
            backend, graph, nullptr, "fsmn_vad");
        throw_if_cancelled(control);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_gallocr_free(allocator);
            throw std::runtime_error("FSMN-VAD graph execution failed");
        }

        std::vector<float> values(frames * model.output_dim);
        ggml_backend_tensor_get(
            probabilities, values.data(), 0, ggml_nbytes(probabilities));
        InferenceResult result;
        result.speech_scores.resize(frames);
        for (std::size_t frame = 0; frame < frames; ++frame) {
            result.speech_scores[frame] =
                1.0f - values[frame * model.output_dim];
        }
        result.caches.assign(
            model.layer_count, std::vector<float>(cache_values));
        for (std::size_t layer = 0; layer < cache_outputs.size(); ++layer) {
            ggml_backend_tensor_get(
                cache_outputs[layer],
                result.caches[layer].data(),
                0,
                ggml_nbytes(cache_outputs[layer]));
        }
        ggml_gallocr_free(allocator);
        allocator = nullptr;
        ggml_free(context);
        return result;
    } catch (...) {
        if (allocator != nullptr) ggml_gallocr_free(allocator);
        ggml_free(context);
        throw;
    }
}

std::vector<float> offline_features(
    const ModelData &model,
    const float *samples,
    std::size_t frame_count,
    const OperationControl &control) {
    if (frame_count < kWindowFrames) return {};
    const FbankFrontend frontend;
    const std::size_t fbank_count =
        (frame_count - kWindowFrames) / kShiftFrames + 1;
    std::vector<std::vector<float>> fbank;
    fbank.reserve(fbank_count);
    for (std::size_t frame = 0; frame < fbank_count; ++frame) {
        if ((frame & 63U) == 0) throw_if_cancelled(control);
        fbank.push_back(frontend.compute(samples + frame * kShiftFrames));
    }
    std::vector<float> features(fbank_count * model.input_dim);
    for (std::size_t frame = 0; frame < fbank_count; ++frame) {
        for (int splice = -2; splice <= 2; ++splice) {
            const auto source = static_cast<std::size_t>(std::clamp<std::int64_t>(
                static_cast<std::int64_t>(frame) + splice,
                0,
                static_cast<std::int64_t>(fbank_count - 1)));
            std::copy(
                fbank[source].begin(),
                fbank[source].end(),
                features.begin() +
                    static_cast<std::ptrdiff_t>(
                        frame * model.input_dim + (splice + 2) * kMelBins));
        }
    }
    const float *shift = model.assets->cmvn_shift.data();
    const float *scale = model.assets->cmvn_scale.data();
    for (std::size_t frame = 0; frame < fbank_count; ++frame) {
        for (int dimension = 0; dimension < model.input_dim; ++dimension) {
            auto &value = features[frame * model.input_dim + dimension];
            value = (value + shift[dimension]) * scale[dimension];
        }
    }
    return features;
}

std::vector<FrameRange> decode_ranges(
    const std::vector<float> &scores,
    std::size_t source_frames,
    const VadOptions &options) {
    const std::size_t min_speech = static_cast<std::size_t>(
        options.min_speech_duration_ms) * kSampleRate / 1000;
    const std::size_t min_silence = static_cast<std::size_t>(
        options.min_silence_duration_ms) * kSampleRate / 1000;
    const std::size_t padding = static_cast<std::size_t>(
        options.speech_padding_ms) * kSampleRate / 1000;
    const std::size_t maximum_speech = options.max_speech_duration_seconds > 0.0
        ? static_cast<std::size_t>(
              options.max_speech_duration_seconds * kSampleRate)
        : 0;
    std::vector<FrameRange> ranges;
    bool active = false;
    std::size_t begin = 0;
    std::size_t silence_begin = 0;
    for (std::size_t index = 0; index < scores.size(); ++index) {
        const std::size_t position = index * kShiftFrames;
        const float silence = 1.0f - scores[index];
        const bool speech = scores[index] >=
            silence + static_cast<float>(options.threshold);
        if (speech) {
            silence_begin = 0;
            if (!active) {
                active = true;
                begin = position;
            }
        } else if (active) {
            if (silence_begin == 0) silence_begin = position;
            if (position - silence_begin >= min_silence) {
                if (silence_begin > begin && silence_begin - begin >= min_speech) {
                    ranges.push_back({begin, silence_begin});
                }
                active = false;
                begin = 0;
                silence_begin = 0;
            }
        }
        if (active && maximum_speech != 0 && position - begin >= maximum_speech) {
            ranges.push_back({begin, position});
            active = false;
            begin = 0;
            silence_begin = 0;
        }
    }
    if (active && source_frames - begin >= min_speech) {
        ranges.push_back({begin, source_frames});
    }
    for (std::size_t index = 0; index < ranges.size(); ++index) {
        const std::size_t padded_begin = ranges[index].begin_frame > padding
            ? ranges[index].begin_frame - padding
            : 0;
        const std::size_t padded_end = std::min(
            source_frames, ranges[index].end_frame + padding);
        if (!ranges.empty() && index > 0 && padded_begin < ranges[index - 1].end_frame) {
            const std::size_t midpoint =
                (ranges[index - 1].end_frame + ranges[index].begin_frame) / 2;
            ranges[index - 1].end_frame = midpoint;
            ranges[index].begin_frame = midpoint;
        } else {
            ranges[index].begin_frame = padded_begin;
        }
        ranges[index].end_frame = padded_end;
    }
    return ranges;
}

class OfflineDetector final {
public:
    struct Result {
        std::vector<FrameRange> ranges;
        std::vector<float> speech_scores;
    };

    OfflineDetector(
        std::shared_ptr<const ModelData> model,
        ggml_backend_t backend)
        : model_(std::move(model)), backend_(backend) {}

    Result detect_mono_f32(
        const float *samples,
        std::size_t frame_count,
        int sample_rate,
        const VadOptions &options,
        const OperationControl &control) {
        if (samples == nullptr || frame_count == 0 || sample_rate != kSampleRate) {
            throw std::invalid_argument(
                "FSMN-VAD requires non-empty mono 16 kHz PCM");
        }
        auto features = offline_features(*model_, samples, frame_count, control);
        if (features.empty()) return {};
        const std::size_t cache_values =
            (model_->left_order - 1) * model_->projection_dim;
        std::vector<std::vector<float>> caches(
            model_->layer_count, std::vector<float>(cache_values));
        auto result = infer(
            *model_, features, caches, backend_, control);
        runtime::report_task_progress(
            control.execution_control, 1.0, "fsmn.infer", 1, 1);
        Result output;
        output.ranges = decode_ranges(
            result.speech_scores, frame_count, options);
        output.speech_scores = std::move(result.speech_scores);
        return output;
    }

private:
    std::shared_ptr<const ModelData> model_;
    ggml_backend_t backend_ = nullptr;
};

class IncrementalDetector final {
public:
    IncrementalDetector(
        std::shared_ptr<const ModelData> model,
        ggml_backend_t backend)
        : model_(std::move(model)), backend_(backend),
          caches_(model_->layer_count,
              std::vector<float>(
                  (model_->left_order - 1) * model_->projection_dim)) {}

    void start(
        const VadOptions &options,
        const OperationControl &control) {
        if (started_) {
            throw std::logic_error("FSMN-VAD incremental session is already active");
        }
        if (options.min_speech_duration_ms != 0 ||
            options.min_silence_duration_ms != 0 ||
            options.speech_padding_ms != 0) {
            throw std::invalid_argument(
                "caller must own incremental VAD duration and padding policy");
        }
        threshold_ = options.threshold;
        control_ = control;
        throw_if_cancelled(control_);
        started_ = true;
    }

    std::vector<RealtimeVadObservation> analyze_mono_f32(
        const float *samples,
        std::size_t frame_count,
        int sample_rate,
        bool final) {
        if (!started_) {
            throw std::logic_error("FSMN-VAD incremental session is not active");
        }
        if (samples == nullptr || frame_count == 0 || sample_rate != kSampleRate) {
            throw std::invalid_argument(
                "FSMN-VAD incremental input must be non-empty mono 16 kHz PCM");
        }
        throw_if_cancelled(control_);
        pcm_.insert(pcm_.end(), samples, samples + frame_count);
        while (pcm_.size() - pcm_offset_ >= kWindowFrames) {
            fbank_.push_back(frontend_.compute(pcm_.data() + pcm_offset_));
            pcm_offset_ += kShiftFrames;
        }
        if (pcm_offset_ > 4096) {
            pcm_.erase(
                pcm_.begin(),
                pcm_.begin() + static_cast<std::ptrdiff_t>(pcm_offset_));
            pcm_offset_ = 0;
        }
        auto features = take_ready_features(final);
        std::vector<float> scores;
        if (!features.empty()) {
            auto result = infer(
                *model_, features, caches_, backend_, control_);
            caches_ = std::move(result.caches);
            scores = std::move(result.speech_scores);
        }

        std::vector<bool> decisions;
        decisions.reserve(scores.size());
        for (float score : scores) {
            raw_speech_ = score >=
                (1.0f - score) + static_cast<float>(threshold_);
            decisions.push_back(raw_speech_);
        }
        std::vector<RealtimeVadObservation> observations;
        std::size_t begin = 0;
        std::size_t decision_index = 0;
        bool state = delayed_speech_;
        while (begin < frame_count) {
            const std::size_t end = std::min(frame_count, begin + kShiftFrames);
            if (decision_index < decisions.size()) {
                state = decisions[decision_index++];
            }
            if (!observations.empty() && observations.back().speech == state) {
                observations.back().end_frame = end;
            } else {
                observations.push_back({begin, end, state});
            }
            begin = end;
        }
        delayed_speech_ = state;
        runtime::report_task_progress(
            control_.execution_control,
            1.0,
            "fsmn.chunk",
            static_cast<std::int64_t>(frame_count),
            static_cast<std::int64_t>(frame_count));
        return observations;
    }

    void finish() {
        if (!started_) {
            throw std::logic_error("FSMN-VAD incremental session is not active");
        }
        throw_if_cancelled(control_);
        started_ = false;
    }

    void reset() {
        control_ = {};
        pcm_.clear();
        pcm_offset_ = 0;
        fbank_.clear();
        fbank_base_ = 0;
        next_lfr_frame_ = 0;
        for (auto &cache : caches_) std::fill(cache.begin(), cache.end(), 0.0f);
        threshold_ = 0.5;
        raw_speech_ = false;
        delayed_speech_ = false;
        started_ = false;
    }

private:
    const std::vector<float> &fbank_at(std::size_t absolute_index) const {
        return fbank_.at(absolute_index - fbank_base_);
    }

    std::vector<float> take_ready_features(bool final) {
        const std::size_t total = fbank_base_ + fbank_.size();
        std::size_t ready_end = total > 2 ? total - 2 : 0;
        if (final) ready_end = total;
        if (ready_end <= next_lfr_frame_) return {};
        std::vector<float> features(
            (ready_end - next_lfr_frame_) * model_->input_dim);
        std::size_t output = 0;
        for (; next_lfr_frame_ < ready_end; ++next_lfr_frame_, ++output) {
            for (int splice = -2; splice <= 2; ++splice) {
                const auto source = static_cast<std::size_t>(
                    std::clamp<std::int64_t>(
                        static_cast<std::int64_t>(next_lfr_frame_) + splice,
                        0,
                        static_cast<std::int64_t>(total - 1)));
                std::copy(
                    fbank_at(source).begin(),
                    fbank_at(source).end(),
                    features.begin() + static_cast<std::ptrdiff_t>(
                        output * model_->input_dim +
                        (splice + 2) * kMelBins));
            }
        }
        const float *shift = model_->assets->cmvn_shift.data();
        const float *scale = model_->assets->cmvn_scale.data();
        for (std::size_t frame = 0; frame < output; ++frame) {
            for (int dimension = 0; dimension < model_->input_dim; ++dimension) {
                auto &value = features[frame * model_->input_dim + dimension];
                value = (value + shift[dimension]) * scale[dimension];
            }
        }
        const std::size_t keep_from = next_lfr_frame_ > 2
            ? next_lfr_frame_ - 2
            : 0;
        while (fbank_base_ < keep_from && !fbank_.empty()) {
            fbank_.pop_front();
            ++fbank_base_;
        }
        return features;
    }

    std::shared_ptr<const ModelData> model_;
    ggml_backend_t backend_ = nullptr;
    FbankFrontend frontend_;
    OperationControl control_;
    std::vector<float> pcm_;
    std::size_t pcm_offset_ = 0;
    std::deque<std::vector<float>> fbank_;
    std::size_t fbank_base_ = 0;
    std::size_t next_lfr_frame_ = 0;
    std::vector<std::vector<float>> caches_;
    double threshold_ = 0.5;
    bool raw_speech_ = false;
    bool delayed_speech_ = false;
    bool started_ = false;
};

VadOptions vad_options_from(
    const std::unordered_map<std::string, std::string> & options,
    const VadOptions * fallback = nullptr) {
    VadOptions config = fallback != nullptr ? *fallback : VadOptions{};
    if (const auto value = runtime::parse_float_option(options, {"threshold"})) {
        config.threshold = *value;
    }
    if (const auto value = runtime::parse_int_option(
            options, {"min_speech_duration_ms"})) {
        config.min_speech_duration_ms = *value;
    }
    if (const auto value = runtime::parse_int_option(
            options, {"min_silence_duration_ms"})) {
        config.min_silence_duration_ms = *value;
    }
    if (const auto value = runtime::parse_int_option(
            options, {"speech_pad_ms"})) {
        config.speech_padding_ms = *value;
    }
    if (const auto value = runtime::parse_float_option(
            options, {"max_speech_duration_s"})) {
        config.max_speech_duration_seconds = *value;
    }
    return config;
}

runtime::SessionOptions fsmn_session_options(runtime::SessionOptions options) {
    if (options.backend.type == engine::core::BackendType::BestAvailable) {
        options.backend.type = engine::core::BackendType::Cpu;
    }
    if (options.backend.type != engine::core::BackendType::Cpu &&
        options.backend.type != engine::core::BackendType::Metal) {
        throw std::invalid_argument(
            "FSMN-VAD explicit validation supports CPU and Metal only");
    }
    if (options.backend.type == engine::core::BackendType::Metal) {
        const auto validation = runtime::find_option(
            options.options, {"fsmn_metal_validation"});
        if (!validation.has_value() ||
            !runtime::parse_bool_option(*validation, "fsmn_metal_validation")) {
            throw std::invalid_argument(
                "FSMN-VAD Metal is rejected outside the explicit validation probe");
        }
    }
    return options;
}

std::filesystem::path resolve_weight_path(
    const std::filesystem::path & model_path) {
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_directory(model_path)) {
        const auto candidate = model_path / "fsmn-vad.gguf";
        if (engine::io::is_existing_file(candidate)) {
            return std::filesystem::weakly_canonical(candidate);
        }
    }
    throw std::runtime_error(
        "FSMN-VAD model path must be a GGUF file or contain fsmn-vad.gguf");
}

class FsmnVadSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::IStreamingVoiceTaskSession {
public:
    FsmnVadSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const ModelAssets> assets)
        : RuntimeSessionBase(fsmn_session_options(options)),
          task_(std::move(task)),
          default_config_(vad_options_from(options.options)),
          model_(std::make_shared<ModelData>(
              std::move(assets), execution_context())),
          offline_(model_, execution_context().backend()),
          incremental_(model_, execution_context().backend()) {
        if (task_.task != runtime::VoiceTaskKind::Vad) {
            throw std::invalid_argument("FSMN-VAD only supports VoiceTaskKind::Vad");
        }
    }

    std::string family() const override { return "fsmn_vad"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        if (request.audio.has_value() &&
            (request.audio->sample_rate != kSampleRate ||
             request.audio->channels != 1)) {
            throw std::invalid_argument(
                "FSMN-VAD requires mono 16 kHz preparation input");
        }
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("FSMN-VAD run()");
        if (task_.mode != runtime::RunMode::Offline) {
            throw std::logic_error(
                "FSMN-VAD offline run called on non-offline session");
        }
        if (!request.audio_input.has_value()) {
            throw std::invalid_argument("FSMN-VAD offline run requires audio_input");
        }
        const auto & audio = *request.audio_input;
        if (audio.sample_rate != kSampleRate || audio.channels != 1 ||
            audio.samples.empty()) {
            throw std::invalid_argument(
                "FSMN-VAD requires non-empty mono 16 kHz PCM");
        }
        const auto config = vad_options_from(request.options, &default_config_);
        runtime::ScopedTaskExecutionControl execution_scope(
            execution_context(), request.execution_control);
        auto detection = offline_.detect_mono_f32(
            audio.samples.data(),
            audio.samples.size(),
            audio.sample_rate,
            config,
            {request.execution_control});
        runtime::TaskResult result;
        result.speech_segments.reserve(detection.ranges.size());
        for (const auto & range : detection.ranges) {
            runtime::SpeechSegment segment;
            segment.span.start_sample = static_cast<std::int64_t>(
                range.begin_frame);
            segment.span.end_sample = static_cast<std::int64_t>(
                range.end_frame);
            result.speech_segments.push_back(std::move(segment));
        }
        const auto return_frame_scores = runtime::find_option(
            request.options, {"return_frame_scores"});
        if (return_frame_scores.has_value() &&
            runtime::parse_bool_option(
                *return_frame_scores, "return_frame_scores")) {
            runtime::VadFrameScoresOutput frame_scores;
            frame_scores.speech_probabilities =
                std::move(detection.speech_scores);
            frame_scores.sample_rate = kSampleRate;
            frame_scores.frame_shift_samples = kShiftFrames;
            result.vad_frame_scores = std::move(frame_scores);
        }
        return result;
    }

    runtime::StreamingPolicy streaming_policy() const override {
        runtime::StreamingPolicy policy;
        policy.input = runtime::StreamingInputKind::AudioChunks;
        policy.output = runtime::StreamingOutputKind::FinalResult;
        policy.preferred_audio_chunk_samples = 512;
        return policy;
    }

    void start_stream(const runtime::TaskRequest & request) override {
        require_prepared("FSMN-VAD start_stream()");
        if (task_.mode != runtime::RunMode::Streaming) {
            throw std::logic_error(
                "FSMN-VAD stream called on non-streaming session");
        }
        incremental_.reset();
        execution_control_ = request.execution_control;
        incremental_.start(default_config_, {request.execution_control});
        next_sample_ = 0;
        emitted_speech_ = false;
        saw_final_chunk_ = false;
        streaming_ = true;
    }

    runtime::StreamEvent process_audio_chunk(
        const runtime::AudioChunk & chunk) override {
        require_prepared("FSMN-VAD process_audio_chunk()");
        if (!streaming_) {
            throw std::logic_error("FSMN-VAD incremental session is not active");
        }
        if (chunk.sample_rate != kSampleRate || chunk.channels != 1 ||
            chunk.samples.empty() || chunk.start_sample != next_sample_) {
            throw std::invalid_argument(
                "FSMN-VAD streaming input must be contiguous mono 16 kHz PCM");
        }
        if (saw_final_chunk_) {
            throw std::logic_error(
                "FSMN-VAD received audio after the final chunk");
        }
        runtime::ScopedTaskExecutionControl execution_scope(
            execution_context(), execution_control_);
        const auto observations = incremental_.analyze_mono_f32(
            chunk.samples.data(),
            chunk.samples.size(),
            chunk.sample_rate,
            chunk.is_final);
        runtime::StreamEvent event;
        for (const auto & observation : observations) {
            if (observation.speech == emitted_speech_) {
                continue;
            }
            runtime::VoiceActivityEvent activity;
            activity.kind = observation.speech
                ? runtime::VoiceActivityEvent::Kind::SpeechStart
                : runtime::VoiceActivityEvent::Kind::SpeechEnd;
            activity.sample = chunk.start_sample +
                static_cast<std::int64_t>(observation.begin_frame);
            event.voice_activity.push_back(std::move(activity));
            emitted_speech_ = observation.speech;
        }
        next_sample_ += static_cast<std::int64_t>(chunk.samples.size());
        saw_final_chunk_ = chunk.is_final;
        return event;
    }

    runtime::TaskResult finalize() override {
        require_prepared("FSMN-VAD finalize()");
        if (!streaming_) {
            throw std::logic_error("FSMN-VAD incremental session is not active");
        }
        if (!saw_final_chunk_ && next_sample_ != 0) {
            throw std::logic_error(
                "FSMN-VAD finalize requires a final AudioChunk");
        }
        incremental_.finish();
        streaming_ = false;
        return {};
    }

    void reset() override {
        incremental_.reset();
        execution_control_.reset();
        next_sample_ = 0;
        emitted_speech_ = false;
        saw_final_chunk_ = false;
        streaming_ = false;
    }

private:
    runtime::TaskSpec task_;
    VadOptions default_config_;
    std::shared_ptr<const ModelData> model_;
    OfflineDetector offline_;
    IncrementalDetector incremental_;
    std::shared_ptr<const runtime::TaskExecutionControl> execution_control_;
    std::int64_t next_sample_ = 0;
    bool emitted_speech_ = false;
    bool saw_final_chunk_ = false;
    bool streaming_ = false;
};

class LoadedModel final : public runtime::ILoadedVoiceModel {
public:
    explicit LoadedModel(std::shared_ptr<const ModelAssets> assets)
        : assets_(std::move(assets)) {
        metadata_.family = "fsmn_vad";
        metadata_.variant = "16k";
        metadata_.description = "FSMN-VAD loaded from GGUF weights.";
        metadata_.weight_candidates = {"fsmn-vad.gguf"};
        capabilities_.supported_tasks = {{
            runtime::VoiceTaskKind::Vad,
            {runtime::RunMode::Offline, runtime::RunMode::Streaming},
        }};
        capabilities_.supports_timestamps = true;
    }

    const runtime::ModelMetadata & metadata() const noexcept override {
        return metadata_;
    }

    const runtime::CapabilitySet & capabilities() const noexcept override {
        return capabilities_;
    }

    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override {
        return std::make_unique<FsmnVadSession>(task, options, assets_);
    }

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const ModelAssets> assets_;
};

class Loader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override { return "fsmn_vad"; }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet capabilities;
        capabilities.supported_tasks = {{
            runtime::VoiceTaskKind::Vad,
            {runtime::RunMode::Offline, runtime::RunMode::Streaming},
        }};
        capabilities.supports_timestamps = true;
        return capabilities;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() &&
            *request.family_hint != family()) {
            return false;
        }
        try {
            (void)resolve_weight_path(request.model_path);
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(
        const runtime::ModelLoadRequest & request) const override {
        if (request.config_id.has_value()) {
            throw std::invalid_argument(
                "FSMN-VAD does not expose selectable config assets");
        }
        const auto path = resolve_weight_path(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = engine::io::is_existing_directory(
            request.model_path)
            ? std::filesystem::weakly_canonical(request.model_path)
            : path.parent_path();
        inspection.metadata.family = family();
        inspection.metadata.variant = "16k";
        inspection.metadata.description =
            "FSMN-VAD loaded from GGUF weights.";
        inspection.metadata.weight_candidates = {"fsmn-vad.gguf"};
        inspection.capabilities = advertised_capabilities();
        inspection.discovered_weights = {{"default", path}};
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(
        const runtime::ModelLoadRequest & request) const override {
        return load_fsmn_vad_model(request);
    }
};

}  // namespace

std::unique_ptr<runtime::ILoadedVoiceModel> load_fsmn_vad_model(
    const runtime::ModelLoadRequest & request) {
    if (request.config_id.has_value()) {
        throw std::invalid_argument(
            "FSMN-VAD does not expose selectable config assets");
    }
    if (request.weight_id.has_value() && *request.weight_id != "default") {
        throw std::invalid_argument("unknown FSMN-VAD weight id");
    }
    const auto path = resolve_weight_path(request.model_path);
    return std::make_unique<LoadedModel>(
        std::make_shared<ModelAssets>(path));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_fsmn_vad_loader() {
    return std::make_shared<Loader>();
}

}  // namespace engine::models::fsmn_vad
