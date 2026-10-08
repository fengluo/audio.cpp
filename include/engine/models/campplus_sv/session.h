#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::campplus_sv {

std::unique_ptr<runtime::ILoadedVoiceModel> load_campplus_sv_model(
    const runtime::ModelLoadRequest & request);
std::shared_ptr<runtime::IVoiceModelLoader> make_campplus_sv_loader();

}  // namespace engine::models::campplus_sv
