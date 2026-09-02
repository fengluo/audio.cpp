#include "engine/models/campplus_sv/session.h"

#include "engine/framework/audio/campplus_fbank.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/modules/speech_encoders/campplus_encoder.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::campplus_sv {
namespace {

constexpr int kSampleRate = 16000;
constexpr int64_t kEmbeddingSize = 192;

bool contains_campplus(std::string value) {
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value.find("campplus") != std::string::npos ||
        value.find("camp_plus") != std::string::npos ||
        value.find("camp-plus") != std::string::npos;
}

std::filesystem::path resolve_weight_path(
    const std::filesystem::path & model_path) {
    if (engine::io::is_existing_file(model_path)) {
        if (model_path.extension() != ".safetensors") {
            throw std::runtime_error(
                "CAMPPlus speaker model must use safetensors weights");
        }
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_directory(model_path)) {
        static const std::vector<std::filesystem::path> candidates = {
            "campplus-sv-zh-en-16k-f32.safetensors",
            "campplus.safetensors",
            "campplus_sv.safetensors",
        };
        for (const auto & relative : candidates) {
            const auto candidate = model_path / relative;
            if (engine::io::is_existing_file(candidate)) {
                return std::filesystem::weakly_canonical(candidate);
            }
        }
    }
    throw std::runtime_error(
        "CAMPPlus model path must be a safetensors file or contain a "
        "recognized CAMPPlus weight filename");
}

engine::core::BackendConfig cpu_backend(engine::core::BackendConfig backend) {
    if (backend.type == engine::core::BackendType::BestAvailable) {
        backend.type = engine::core::BackendType::Cpu;
    }
    if (backend.type != engine::core::BackendType::Cpu) {
        throw std::invalid_argument(
            "CAMPPlus speaker embedding currently supports CPU only");
    }
    return backend;
}

engine::core::BackendConfig model_backend(
    const runtime::ModelLoadRequest & request) {
    engine::core::BackendConfig backend;
    backend.type = engine::core::BackendType::Cpu;
    if (const auto threads = runtime::parse_int_option(
            request.options, {"threads", "thread_count"})) {
        if (*threads <= 0 || *threads > 256) {
            throw std::invalid_argument(
                "CAMPPlus load option threads must be in [1, 256]");
        }
        backend.threads = *threads;
    }
    return backend;
}

class CampplusSpeakerSession final
    : public runtime::IOfflineVoiceTaskSession {
public:
    CampplusSpeakerSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const modules::CampplusEncoderWeights> weights)
        : task_(std::move(task)),
          backend_(weights->execution_context->config()),
          encoder_(std::move(weights), backend_) {
        (void)cpu_backend(options.backend);
        if (task_.task != runtime::VoiceTaskKind::SpeakerRecognition) {
            throw std::invalid_argument(
                "CAMPPlus only supports VoiceTaskKind::SpeakerRecognition");
        }
        if (task_.mode != runtime::RunMode::Offline) {
            throw std::invalid_argument(
                "CAMPPlus speaker embedding only supports offline execution");
        }
    }

    std::string family() const override { return "campplus_sv"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        if (request.audio.has_value() &&
            (request.audio->sample_rate != kSampleRate ||
             request.audio->channels != 1)) {
            throw std::invalid_argument(
                "CAMPPlus preparation requires mono 16 kHz PCM");
        }
        prepared_ = true;
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        if (!prepared_) {
            throw std::logic_error(
                "CAMPPlus session prepare() must be called before run()");
        }
        if (!request.audio_input.has_value()) {
            throw std::invalid_argument(
                "CAMPPlus speaker embedding requires audio_input");
        }
        const auto & audio = *request.audio_input;
        if (audio.sample_rate != kSampleRate || audio.channels != 1 ||
            audio.samples.empty()) {
            throw std::invalid_argument(
                "CAMPPlus requires non-empty mono 16 kHz PCM");
        }

        runtime::throw_if_task_cancelled(request.execution_control);
        runtime::report_task_progress(
            request.execution_control, 0.0, "speaker.frontend", 0,
            static_cast<int64_t>(audio.samples.size()));
        auto features = audio::extract_campplus_fbank_16k(audio.samples);
        runtime::throw_if_task_cancelled(request.execution_control);
        runtime::report_task_progress(
            request.execution_control, 0.35, "speaker.encoder", 0,
            features.frames);
        auto output = encoder_.embed_from_features(
            features.values, features.frames, features.dims);
        // The shared CAMPPlus runner currently reuses ggml scratch tensors
        // whose contents are not stable across a second host execution.
        // Rebuild the runtime graph per request while retaining uploaded,
        // immutable weights; lifecycle determinism is part of this provider's
        // contract and graph reuse can be restored after the runner is fixed.
        encoder_.release_runtime_graph();
        runtime::throw_if_task_cancelled(request.execution_control);
        if (output.embedding_size != kEmbeddingSize ||
            output.embedding.size() != static_cast<size_t>(kEmbeddingSize)) {
            throw std::runtime_error(
                "CAMPPlus returned an unexpected embedding dimension");
        }
        if (!std::all_of(
                output.embedding.begin(), output.embedding.end(),
                [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error(
                "CAMPPlus returned a non-finite speaker embedding");
        }

        runtime::TaskResult result;
        runtime::SpeakerEmbeddingOutput embedding;
        embedding.values = std::move(output.embedding);
        embedding.feature_frame_count = features.frames;
        embedding.l2_normalized = false;
        result.speaker_embedding = std::move(embedding);
        runtime::report_task_progress(
            request.execution_control, 1.0, "speaker.complete",
            features.frames, features.frames);
        return result;
    }

private:
    runtime::TaskSpec task_;
    engine::core::BackendConfig backend_;
    modules::CampplusEncoderComponent encoder_;
    bool prepared_ = false;
};

class LoadedModel final : public runtime::ILoadedVoiceModel {
public:
    explicit LoadedModel(
        std::shared_ptr<const modules::CampplusEncoderWeights> weights)
        : weights_(std::move(weights)) {
        metadata_.family = "campplus_sv";
        metadata_.variant = "zh-en-16k-192";
        metadata_.description =
            "FunASR CAMPPlus speaker embedding model loaded from safetensors.";
        metadata_.weight_candidates = {
            "campplus-sv-zh-en-16k-f32.safetensors",
            "campplus.safetensors",
            "campplus_sv.safetensors",
        };
        capabilities_.supported_tasks = {{
            runtime::VoiceTaskKind::SpeakerRecognition,
            {runtime::RunMode::Offline},
        }};
        capabilities_.languages = {"zh", "en"};
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
        return std::make_unique<CampplusSpeakerSession>(
            task, options, weights_);
    }

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const modules::CampplusEncoderWeights> weights_;
};

class Loader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override { return "campplus_sv"; }
    std::vector<std::string> family_aliases() const override {
        return {"campplus"};
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet capabilities;
        capabilities.supported_tasks = {{
            runtime::VoiceTaskKind::SpeakerRecognition,
            {runtime::RunMode::Offline},
        }};
        capabilities.languages = {"zh", "en"};
        return capabilities;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() &&
            *request.family_hint != family() &&
            *request.family_hint != "campplus") {
            return false;
        }
        if (!request.family_hint.has_value() &&
            !contains_campplus(request.model_path.filename().string())) {
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
                "CAMPPlus does not expose selectable config assets");
        }
        if (request.weight_id.has_value() &&
            *request.weight_id != "default") {
            throw std::invalid_argument("unknown CAMPPlus weight id");
        }
        const auto path = resolve_weight_path(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = engine::io::is_existing_directory(
            request.model_path)
            ? std::filesystem::weakly_canonical(request.model_path)
            : path.parent_path();
        inspection.metadata.family = family();
        inspection.metadata.variant = "zh-en-16k-192";
        inspection.metadata.description =
            "FunASR CAMPPlus speaker embedding model loaded from safetensors.";
        inspection.metadata.weight_candidates = {
            "campplus-sv-zh-en-16k-f32.safetensors",
            "campplus.safetensors",
            "campplus_sv.safetensors",
        };
        inspection.capabilities = advertised_capabilities();
        inspection.discovered_weights = {{"default", path}};
        inspection.cli.load_options = {{
            "threads", "COUNT",
            "CPU inference thread count fixed when the model is loaded.",
            false, "1", "1", "256",
        }};
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(
        const runtime::ModelLoadRequest & request) const override {
        return load_campplus_sv_model(request);
    }
};

}  // namespace

std::unique_ptr<runtime::ILoadedVoiceModel> load_campplus_sv_model(
    const runtime::ModelLoadRequest & request) {
    if (request.config_id.has_value()) {
        throw std::invalid_argument(
            "CAMPPlus does not expose selectable config assets");
    }
    if (request.weight_id.has_value() && *request.weight_id != "default") {
        throw std::invalid_argument("unknown CAMPPlus weight id");
    }
    const auto path = resolve_weight_path(request.model_path);
    modules::CampplusEncoderConfig config;
    config.feat_dim = 80;
    config.embedding_size = kEmbeddingSize;
    config.tensor_prefix = "speaker_encoder";
    auto encoder = modules::CampplusEncoderComponent::load_from_safetensors(
        path, model_backend(request), config);
    return std::make_unique<LoadedModel>(encoder.weights());
}

std::shared_ptr<runtime::IVoiceModelLoader> make_campplus_sv_loader() {
    return std::make_shared<Loader>();
}

}  // namespace engine::models::campplus_sv
