#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/fsmn_vad/session.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

struct RunResult {
    engine::runtime::TaskResult result;
    double session_ms = 0.0;
    double inference_ms = 0.0;
};

struct StreamObservation {
    engine::runtime::VoiceActivityEvent::Kind kind =
        engine::runtime::VoiceActivityEvent::Kind::SpeechSegment;
    int64_t sample = 0;
};

struct StreamRunResult {
    std::vector<StreamObservation> observations;
    double session_ms = 0.0;
    double processing_ms = 0.0;
    double max_chunk_ms = 0.0;
};

RunResult run_offline(
    const engine::runtime::ILoadedVoiceModel &model,
    engine::core::BackendType backend,
    const std::vector<float> &samples) {
    engine::runtime::SessionOptions options;
    options.backend.type = backend;
    options.backend.threads = 4;
    if (backend == engine::core::BackendType::Metal) {
        options.options["fsmn_metal_validation"] = "true";
    }
    const auto session_begin = Clock::now();
    auto base = model.create_task_session(
        {engine::runtime::VoiceTaskKind::Vad,
         engine::runtime::RunMode::Offline},
        options);
    const auto session_end = Clock::now();
    auto *session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(
        base.get());
    require(session != nullptr, "FSMN-VAD did not create an offline session");
    engine::runtime::SessionPreparationRequest preparation;
    preparation.audio = engine::runtime::AudioPreparationContract{16000, 1, 0};
    session->prepare(preparation);

    engine::runtime::TaskRequest request;
    request.audio_input = engine::runtime::AudioBuffer{16000, 1, samples};
    request.options["min_speech_duration_ms"] = "0";
    request.options["min_silence_duration_ms"] = "0";
    request.options["speech_pad_ms"] = "0";
    request.options["return_frame_scores"] = "true";
    const auto inference_begin = Clock::now();
    auto result = session->run(request);
    const auto inference_end = Clock::now();
    require(result.vad_frame_scores.has_value(),
            "FSMN-VAD returned no frame scores");
    return {
        std::move(result),
        std::chrono::duration<double, std::milli>(
            session_end - session_begin).count(),
        std::chrono::duration<double, std::milli>(
            inference_end - inference_begin).count(),
    };
}

StreamRunResult run_streaming(
    const engine::runtime::ILoadedVoiceModel &model,
    engine::core::BackendType backend,
    const std::vector<float> &samples) {
    engine::runtime::SessionOptions options;
    options.backend.type = backend;
    options.backend.threads = 4;
    if (backend == engine::core::BackendType::Metal) {
        options.options["fsmn_metal_validation"] = "true";
    }
    options.options["min_speech_duration_ms"] = "0";
    options.options["min_silence_duration_ms"] = "0";
    options.options["speech_pad_ms"] = "0";
    const auto session_begin = Clock::now();
    auto base = model.create_task_session(
        {engine::runtime::VoiceTaskKind::Vad,
         engine::runtime::RunMode::Streaming},
        options);
    const auto session_end = Clock::now();
    auto *session = dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(
        base.get());
    require(session != nullptr, "FSMN-VAD did not create a streaming session");
    engine::runtime::SessionPreparationRequest preparation;
    preparation.audio = engine::runtime::AudioPreparationContract{16000, 1, 0};
    session->prepare(preparation);
    engine::runtime::TaskRequest start_request;
    session->start_stream(start_request);

    StreamRunResult output;
    output.session_ms = std::chrono::duration<double, std::milli>(
        session_end - session_begin).count();
    constexpr std::size_t kChunkFrames = 512;
    const auto processing_begin = Clock::now();
    std::size_t offset = 0;
    while (offset < samples.size()) {
        const std::size_t count = std::min(
            kChunkFrames, samples.size() - offset);
        engine::runtime::AudioChunk chunk;
        chunk.samples.assign(
            samples.begin() + static_cast<std::ptrdiff_t>(offset),
            samples.begin() + static_cast<std::ptrdiff_t>(offset + count));
        chunk.sample_rate = 16000;
        chunk.channels = 1;
        chunk.start_sample = static_cast<int64_t>(offset);
        chunk.is_final = offset + count == samples.size();
        const auto chunk_begin = Clock::now();
        const auto event = session->process_audio_chunk(chunk);
        const auto chunk_end = Clock::now();
        output.max_chunk_ms = std::max(
            output.max_chunk_ms,
            std::chrono::duration<double, std::milli>(
                chunk_end - chunk_begin).count());
        for (const auto &activity : event.voice_activity) {
            output.observations.push_back({activity.kind, activity.sample});
        }
        offset += count;
    }
    (void)session->finalize();
    const auto processing_end = Clock::now();
    output.processing_ms = std::chrono::duration<double, std::milli>(
        processing_end - processing_begin).count();
    session->reset();
    return output;
}

double repeat_max_abs(
    const engine::runtime::TaskResult &left,
    const engine::runtime::TaskResult &right) {
    require(left.vad_frame_scores.has_value() &&
            right.vad_frame_scores.has_value(),
            "FSMN-VAD repeat result has no frame scores");
    const auto &left_scores =
        left.vad_frame_scores->speech_probabilities;
    const auto &right_scores =
        right.vad_frame_scores->speech_probabilities;
    require(left_scores.size() == right_scores.size(),
            "FSMN-VAD repeat frame counts differ");
    double maximum = 0.0;
    for (std::size_t index = 0; index < left_scores.size(); ++index) {
        maximum = std::max(
            maximum,
            std::abs(static_cast<double>(left_scores[index]) -
                     right_scores[index]));
    }
    return maximum;
}

struct CancellationResult {
    bool cancelled = false;
    double elapsed_ms = 0.0;
    double reuse_max_abs_error = 0.0;
};

CancellationResult verify_metal_cancellation(
    const engine::runtime::ILoadedVoiceModel &model,
    const std::vector<float> &samples,
    const engine::runtime::TaskResult &expected) {
    engine::runtime::SessionOptions options;
    options.backend.type = engine::core::BackendType::Metal;
    options.options["fsmn_metal_validation"] = "true";
    auto base = model.create_task_session(
        {engine::runtime::VoiceTaskKind::Vad,
         engine::runtime::RunMode::Offline},
        options);
    auto *session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(
        base.get());
    require(session != nullptr,
            "FSMN-VAD did not create a Metal cancellation session");
    engine::runtime::SessionPreparationRequest preparation;
    preparation.audio = engine::runtime::AudioPreparationContract{16000, 1, 0};
    session->prepare(preparation);

    engine::runtime::TaskRequest request;
    request.audio_input = engine::runtime::AudioBuffer{16000, 1, samples};
    request.options["min_speech_duration_ms"] = "0";
    request.options["min_silence_duration_ms"] = "0";
    request.options["speech_pad_ms"] = "0";
    request.options["return_frame_scores"] = "true";
    auto control = std::make_shared<engine::runtime::TaskExecutionControl>();
    std::atomic<bool> cancellation_requested{false};
    control->cancellation_requested = [&] {
        return cancellation_requested.load(std::memory_order_relaxed);
    };
    request.execution_control = control;
    std::thread canceller([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cancellation_requested.store(true, std::memory_order_relaxed);
    });
    CancellationResult output;
    const auto begin = Clock::now();
    try {
        (void)session->run(request);
    } catch (const engine::runtime::TaskCancelled &) {
        output.cancelled = true;
    }
    const auto end = Clock::now();
    canceller.join();
    output.elapsed_ms = std::chrono::duration<double, std::milli>(
        end - begin).count();
    require(output.cancelled,
            "FSMN-VAD Metal in-flight cancellation was ignored");

    request.execution_control.reset();
    const auto reuse = session->run(request);
    output.reuse_max_abs_error = repeat_max_abs(expected, reuse);
    return output;
}

}  // namespace

int main(int argc, char **argv) try {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: fsmn_vad_metal_probe <fsmn-vad.gguf> "
                     "<mono-16k.wav> [cpu|metal]\n";
        return 2;
    }
    const auto wav = engine::audio::read_wav_f32(
        std::filesystem::path(argv[2]));
    require(wav.sample_rate == 16000 && wav.channels == 1,
            "FSMN-VAD probe WAV must be mono 16 kHz");

    const auto load_begin = Clock::now();
    engine::runtime::ModelLoadRequest load_request;
    load_request.model_path = argv[1];
    load_request.family_hint = "fsmn_vad";
    auto model = engine::models::fsmn_vad::load_fsmn_vad_model(load_request);
    const auto load_end = Clock::now();
    if (argc == 4) {
        const std::string backend_name = argv[3];
        const auto backend = backend_name == "cpu"
            ? engine::core::BackendType::Cpu
            : backend_name == "metal"
                ? engine::core::BackendType::Metal
                : throw std::invalid_argument(
                      "single-backend mode requires cpu or metal");
        auto first = run_offline(*model, backend, wav.samples);
        auto second = run_offline(*model, backend, wav.samples);
        auto stream = run_streaming(*model, backend, wav.samples);
        std::cout << std::setprecision(12)
                  << "{\"backend\":\"" << backend_name
                  << "\",\"frames\":"
                  << first.result.vad_frame_scores->speech_probabilities.size()
                  << ",\"ranges\":"
                  << first.result.speech_segments.size()
                  << ",\"repeat_max_abs_error\":"
                  << repeat_max_abs(first.result, second.result)
                  << ",\"load_ms\":"
                  << std::chrono::duration<double, std::milli>(
                         load_end - load_begin).count()
                  << ",\"first_session_ms\":" << first.session_ms
                  << ",\"second_session_ms\":" << second.session_ms
                  << ",\"first_inference_ms\":" << first.inference_ms
                  << ",\"second_inference_ms\":" << second.inference_ms
                  << ",\"stream_events\":"
                  << stream.observations.size()
                  << ",\"stream_processing_ms\":"
                  << stream.processing_ms
                  << ",\"stream_max_chunk_ms\":"
                  << stream.max_chunk_ms << "}\n";
        return 0;
    }
    auto cpu = run_offline(
        *model, engine::core::BackendType::Cpu, wav.samples);
    auto metal = run_offline(
        *model, engine::core::BackendType::Metal, wav.samples);
    auto cpu_repeat = run_offline(
        *model, engine::core::BackendType::Cpu, wav.samples);
    auto metal_repeat = run_offline(
        *model, engine::core::BackendType::Metal, wav.samples);
    auto cpu_stream = run_streaming(
        *model, engine::core::BackendType::Cpu, wav.samples);
    auto metal_stream = run_streaming(
        *model, engine::core::BackendType::Metal, wav.samples);
    const auto metal_cancellation = verify_metal_cancellation(
        *model, wav.samples, metal.result);

    const auto &cpu_scores =
        cpu.result.vad_frame_scores->speech_probabilities;
    const auto &metal_scores =
        metal.result.vad_frame_scores->speech_probabilities;
    require(cpu_scores.size() == metal_scores.size(),
            "FSMN-VAD CPU and Metal frame counts differ");
    double max_abs = 0.0;
    double mean_abs = 0.0;
    std::size_t decision_mismatches = 0;
    std::vector<std::size_t> decision_mismatch_frames;
    for (std::size_t index = 0; index < cpu_scores.size(); ++index) {
        const double difference = std::abs(
            static_cast<double>(cpu_scores[index]) - metal_scores[index]);
        max_abs = std::max(max_abs, difference);
        mean_abs += difference;
        const bool cpu_speech = cpu_scores[index] >= 0.75F;
        const bool metal_speech = metal_scores[index] >= 0.75F;
        if (cpu_speech != metal_speech) {
            ++decision_mismatches;
            decision_mismatch_frames.push_back(index);
        }
    }
    if (!cpu_scores.empty()) mean_abs /= cpu_scores.size();
    const double cpu_repeat_error = repeat_max_abs(
        cpu.result, cpu_repeat.result);
    const double metal_repeat_error = repeat_max_abs(
        metal.result, metal_repeat.result);

    std::size_t stream_mismatches =
        cpu_stream.observations.size() == metal_stream.observations.size()
        ? 0U
        : std::max(
              cpu_stream.observations.size(),
              metal_stream.observations.size());
    if (stream_mismatches == 0) {
        for (std::size_t index = 0;
             index < cpu_stream.observations.size(); ++index) {
            const auto &left = cpu_stream.observations[index];
            const auto &right = metal_stream.observations[index];
            stream_mismatches +=
                left.kind != right.kind || left.sample != right.sample
                ? 1U : 0U;
        }
    }

    std::size_t range_mismatches =
        cpu.result.speech_segments.size() == metal.result.speech_segments.size()
        ? 0U
        : std::max(
              cpu.result.speech_segments.size(),
              metal.result.speech_segments.size());
    if (range_mismatches == 0) {
        for (std::size_t index = 0;
             index < cpu.result.speech_segments.size(); ++index) {
            const auto &left = cpu.result.speech_segments[index].span;
            const auto &right = metal.result.speech_segments[index].span;
            range_mismatches +=
                left.start_sample != right.start_sample ||
                left.end_sample != right.end_sample ? 1U : 0U;
        }
    }

    std::cout << std::setprecision(12)
              << "{\"frames\":" << cpu_scores.size()
              << ",\"cpu_ranges\":"
              << cpu.result.speech_segments.size()
              << ",\"metal_ranges\":"
              << metal.result.speech_segments.size()
              << ",\"range_mismatches\":" << range_mismatches
              << ",\"decision_mismatches\":" << decision_mismatches
              << ",\"decision_mismatch_details\":[";
    for (std::size_t index = 0;
         index < decision_mismatch_frames.size(); ++index) {
        if (index != 0) std::cout << ',';
        const auto frame = decision_mismatch_frames[index];
        std::cout << "{\"frame\":" << frame
                  << ",\"cpu\":" << cpu_scores[frame]
                  << ",\"metal\":" << metal_scores[frame] << '}';
    }
    std::cout << ']'
              << ",\"max_abs_error\":" << max_abs
              << ",\"mean_abs_error\":" << mean_abs
              << ",\"cpu_repeat_max_abs_error\":" << cpu_repeat_error
              << ",\"metal_repeat_max_abs_error\":" << metal_repeat_error
              << ",\"load_ms\":"
              << std::chrono::duration<double, std::milli>(
                     load_end - load_begin).count()
              << ",\"cpu_session_ms\":" << cpu.session_ms
              << ",\"metal_session_ms\":" << metal.session_ms
              << ",\"cpu_inference_ms\":" << cpu.inference_ms
              << ",\"metal_inference_ms\":" << metal.inference_ms
              << ",\"cpu_repeat_inference_ms\":"
              << cpu_repeat.inference_ms
              << ",\"metal_repeat_inference_ms\":"
              << metal_repeat.inference_ms
              << ",\"cpu_stream_events\":"
              << cpu_stream.observations.size()
              << ",\"metal_stream_events\":"
              << metal_stream.observations.size()
              << ",\"stream_mismatches\":" << stream_mismatches
              << ",\"cpu_stream_processing_ms\":"
              << cpu_stream.processing_ms
              << ",\"metal_stream_processing_ms\":"
              << metal_stream.processing_ms
              << ",\"cpu_stream_max_chunk_ms\":"
              << cpu_stream.max_chunk_ms
              << ",\"metal_stream_max_chunk_ms\":"
              << metal_stream.max_chunk_ms
              << ",\"metal_cancelled\":"
              << (metal_cancellation.cancelled ? "true" : "false")
              << ",\"metal_cancellation_ms\":"
              << metal_cancellation.elapsed_ms
              << ",\"metal_cancel_reuse_max_abs_error\":"
              << metal_cancellation.reuse_max_abs_error
              << ",\"range_mismatch_details\":[";
    const auto comparable_ranges = std::min(
        cpu.result.speech_segments.size(),
        metal.result.speech_segments.size());
    bool wrote_range = false;
    for (std::size_t index = 0; index < comparable_ranges; ++index) {
        const auto &left = cpu.result.speech_segments[index].span;
        const auto &right = metal.result.speech_segments[index].span;
        if (left.start_sample == right.start_sample &&
            left.end_sample == right.end_sample) {
            continue;
        }
        if (wrote_range) std::cout << ',';
        wrote_range = true;
        std::cout << "{\"index\":" << index
                  << ",\"cpu_begin\":" << left.start_sample
                  << ",\"cpu_end\":" << left.end_sample
                  << ",\"metal_begin\":" << right.start_sample
                  << ",\"metal_end\":" << right.end_sample << '}';
    }
    std::cout << "]}\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << "fsmn_vad_metal_probe failed: " << error.what() << '\n';
    return 1;
}
