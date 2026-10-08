#pragma once

#include <cstdint>
#include <vector>

namespace engine::audio {

struct CampplusFbankFeatures {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t dims = 0;
};

// FunASR-compatible, snip-edges 16 kHz/80-bin Kaldi fbank followed by
// utterance-level cepstral mean normalization. The input is normalized mono
// float PCM; a final tail shorter than one complete frame is intentionally
// ignored.
CampplusFbankFeatures extract_campplus_fbank_16k(
    const std::vector<float> & waveform_16k);

}  // namespace engine::audio
