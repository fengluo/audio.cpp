#include "engine/framework/audio/campplus_fbank.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/campplus_sv/session.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<float> deterministic_pcm() {
    constexpr int kSampleRate = 16000;
    constexpr int kSamples = kSampleRate * 2;
    constexpr double kPi = 3.14159265358979323846;
    std::vector<float> samples(kSamples);
    for (int index = 0; index < kSamples; ++index) {
        const double seconds = static_cast<double>(index) / kSampleRate;
        const double envelope = std::min(1.0, seconds * 8.0) *
            std::min(1.0, (2.0 - seconds) * 8.0);
        samples[static_cast<size_t>(index)] = static_cast<float>(
            envelope * (0.18 * std::sin(2.0 * kPi * 173.0 * seconds) +
                        0.07 * std::sin(2.0 * kPi * 347.0 * seconds) +
                        0.03 * std::sin(2.0 * kPi * 701.0 * seconds)));
    }
    return samples;
}

}  // namespace

int main(int argc, char ** argv) try {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: campplus_sv_probe <campplus.safetensors> [mono-16k.wav]\n";
        return 2;
    }

    auto loader = engine::models::campplus_sv::make_campplus_sv_loader();
    engine::runtime::ModelLoadRequest load_request;
    load_request.model_path = argv[1];
    load_request.family_hint = "campplus_sv";
    load_request.options["threads"] = "4";
    require(loader->can_load(load_request), "CAMPPlus loader rejected its governed weights");
    const auto inspection = loader->inspect(load_request);
    require(inspection.metadata.family == "campplus_sv", "CAMPPlus inspection family mismatch");
    require(inspection.capabilities.supported_tasks.size() == 1,
            "CAMPPlus inspection task count mismatch");

    auto model = loader->load(load_request);
    bool rejected_metal = false;
    try {
        engine::runtime::SessionOptions metal_options;
        metal_options.backend.type = engine::core::BackendType::Metal;
        (void)model->create_task_session(
            {engine::runtime::VoiceTaskKind::SpeakerRecognition,
             engine::runtime::RunMode::Offline},
            metal_options);
    } catch (const std::invalid_argument &) {
        rejected_metal = true;
    }
    require(rejected_metal,
            "CAMPPlus accepted an unvalidated explicit Metal session");

    engine::runtime::SessionOptions session_options;
    session_options.backend.type = engine::core::BackendType::BestAvailable;
    auto session_base = model->create_task_session(
        {engine::runtime::VoiceTaskKind::SpeakerRecognition,
         engine::runtime::RunMode::Offline},
        session_options);
    auto * session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(
        session_base.get());
    require(session != nullptr, "CAMPPlus did not create an offline session");

    // A session owns the immutable uploaded weights it needs.
    model.reset();
    engine::runtime::SessionPreparationRequest preparation;
    preparation.audio = engine::runtime::AudioPreparationContract{16000, 1, 0};
    session->prepare(preparation);

    std::vector<float> pcm = deterministic_pcm();
    std::string input_kind = "deterministic_pcm";
    if (argc == 3) {
        const auto wav = engine::audio::read_wav_f32(
            std::filesystem::path(argv[2]));
        require(wav.sample_rate == 16000 && wav.channels == 1,
                "CAMPPlus probe WAV must be mono 16 kHz");
        pcm = wav.samples;
        input_kind = "wav";
    }
    const auto native_features =
        engine::audio::extract_campplus_fbank_16k(pcm);
    const auto one_frame = engine::audio::extract_campplus_fbank_16k(
        std::vector<float>(559, 0.25F));
    require(one_frame.frames == 1 &&
            std::all_of(one_frame.values.begin(), one_frame.values.end(),
                        [](float value) { return value == 0.0F; }),
            "CAMPPlus short-tail/CMN contract mismatch");
    const auto two_frames = engine::audio::extract_campplus_fbank_16k(
        std::vector<float>(560, 0.25F));
    require(two_frames.frames == 2,
            "CAMPPlus complete second frame was not retained");
    bool rejected_non_finite = false;
    try {
        auto invalid = std::vector<float>(400, 0.0F);
        invalid[123] = std::numeric_limits<float>::quiet_NaN();
        (void)engine::audio::extract_campplus_fbank_16k(invalid);
    } catch (const std::invalid_argument &) {
        rejected_non_finite = true;
    }
    require(rejected_non_finite,
            "CAMPPlus frontend accepted non-finite PCM");

    engine::runtime::TaskRequest request;
    request.audio_input = engine::runtime::AudioBuffer{16000, 1, pcm};
    std::vector<double> progress;
    auto control = std::make_shared<engine::runtime::TaskExecutionControl>();
    control->progress = [&](const engine::runtime::TaskProgress & value) {
        require(progress.empty() || value.fraction >= progress.back(),
                "CAMPPlus progress regressed");
        progress.push_back(value.fraction);
    };
    request.execution_control = control;
    const auto first = session->run(request);
    require(first.speaker_embedding.has_value(),
            "CAMPPlus returned no typed speaker embedding");
    require(first.speaker_embedding->values.size() == 192,
            "CAMPPlus embedding dimension mismatch");
    require(first.speaker_embedding->feature_frame_count == native_features.frames,
            "CAMPPlus snip-edges feature frame count mismatch");
    require(!first.speaker_embedding->l2_normalized,
            "CAMPPlus unexpectedly marked raw output L2-normalized");
    require(!progress.empty() && progress.front() == 0.0 && progress.back() == 1.0,
            "CAMPPlus progress lifecycle mismatch");

    bool rejected_empty = false;
    try {
        auto empty_request = request;
        empty_request.audio_input->samples.clear();
        (void)session->run(empty_request);
    } catch (const std::invalid_argument &) {
        rejected_empty = true;
    }
    require(rejected_empty, "CAMPPlus session accepted empty PCM");

    request.execution_control.reset();
    const auto second = session->run(request);
    require(second.speaker_embedding.has_value(),
            "CAMPPlus repeated inference returned no embedding");
    float repeat_max_abs = 0.0F;
    for (size_t index = 0;
         index < first.speaker_embedding->values.size(); ++index) {
        repeat_max_abs = std::max(
            repeat_max_abs,
            std::abs(first.speaker_embedding->values[index] -
                     second.speaker_embedding->values[index]));
    }
    if (repeat_max_abs > 1.0e-5F) {
        std::cerr << "CAMPPlus repeat max abs: " << repeat_max_abs << '\n';
    }
    require(repeat_max_abs <= 1.0e-5F,
            "CAMPPlus repeated inference exceeded numeric tolerance");

    bool cancelled = false;
    auto cancel_control = std::make_shared<engine::runtime::TaskExecutionControl>();
    cancel_control->cancellation_requested = [] { return true; };
    request.execution_control = cancel_control;
    try {
        (void)session->run(request);
    } catch (const engine::runtime::TaskCancelled &) {
        cancelled = true;
    }
    require(cancelled, "CAMPPlus ignored pre-requested cancellation");

    std::cout << std::setprecision(9);
    std::cout << "{\"input_kind\":\"" << input_kind
              << "\",\"frames\":"
              << first.speaker_embedding->feature_frame_count
              << ",\"dims\":" << native_features.dims
              << ",\"features\":[";
    for (size_t index = 0; index < native_features.values.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << native_features.values[index];
    }
    std::cout << "],\"embedding\":[";
    for (size_t index = 0;
         index < first.speaker_embedding->values.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << first.speaker_embedding->values[index];
    }
    std::cout << "]}\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "campplus_sv_probe failed: " << error.what() << '\n';
    return 1;
}
