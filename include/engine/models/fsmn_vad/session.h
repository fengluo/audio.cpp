#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::fsmn_vad {

std::unique_ptr<runtime::ILoadedVoiceModel> load_fsmn_vad_model(
    const runtime::ModelLoadRequest & request);
std::shared_ptr<runtime::IVoiceModelLoader> make_fsmn_vad_loader();

}  // namespace engine::models::fsmn_vad
