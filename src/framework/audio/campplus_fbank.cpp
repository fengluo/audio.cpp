#include "engine/framework/audio/campplus_fbank.h"

#include "engine/framework/audio/kaldi_fbank.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::audio {

CampplusFbankFeatures extract_campplus_fbank_16k(
    const std::vector<float> & waveform_16k) {
    if (waveform_16k.empty()) {
        throw std::invalid_argument(
            "CAMPPlus fbank requires non-empty mono 16 kHz PCM");
    }
    for (const float sample : waveform_16k) {
        if (!std::isfinite(sample)) {
            throw std::invalid_argument(
                "CAMPPlus fbank PCM contains a non-finite sample");
        }
    }

    KaldiFbankOptions options;
    options.sample_rate = 16000;
    options.num_mels = 80;
    options.frame_length_ms = 25.0F;
    options.frame_shift_ms = 10.0F;
    options.window_type = KaldiFbankWindowType::Povey;
    options.lfr_m = 1;
    options.lfr_n = 1;
    options.preemphasis = 0.97F;
    options.low_frequency = 20.0F;
    options.high_frequency = 0.0F;
    options.remove_dc_offset = true;
    options.upscale_samples = false;
    options.apply_cmvn = false;

    auto fbank = extract_kaldi_fbank(waveform_16k, options);
    if (fbank.frames <= 0) {
        throw std::invalid_argument(
            "CAMPPlus fbank requires at least one complete 25 ms frame");
    }
    if (fbank.feature_dim != 80) {
        throw std::runtime_error("CAMPPlus fbank feature dimension mismatch");
    }

    for (int mel = 0; mel < fbank.feature_dim; ++mel) {
        double sum = 0.0;
        for (int frame = 0; frame < fbank.frames; ++frame) {
            sum += fbank.values[
                static_cast<size_t>(frame * fbank.feature_dim + mel)];
        }
        const float mean = static_cast<float>(sum / fbank.frames);
        for (int frame = 0; frame < fbank.frames; ++frame) {
            fbank.values[
                static_cast<size_t>(frame * fbank.feature_dim + mel)] -= mean;
        }
    }

    CampplusFbankFeatures result;
    result.values = std::move(fbank.values);
    result.frames = fbank.frames;
    result.dims = fbank.feature_dim;
    return result;
}

}  // namespace engine::audio
