#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"

#include "test_assert.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace {

using engine::test::require;
using engine::test::require_eq;

struct ContextDeleter {
    void operator()(ggml_context * context) const {
        if (context != nullptr) ggml_free(context);
    }
};

struct BufferDeleter {
    void operator()(ggml_backend_buffer_t buffer) const {
        if (buffer != nullptr) ggml_backend_buffer_free(buffer);
    }
};

void test_progress_and_cancellation_helpers() {
    std::vector<engine::runtime::TaskProgress> events;
    auto control = std::make_shared<engine::runtime::TaskExecutionControl>();
    control->cancellation_requested = [] { return false; };
    control->progress = [&](const engine::runtime::TaskProgress & progress) {
        events.push_back(progress);
    };

    engine::runtime::report_task_progress(control, -0.5, "starting");
    engine::runtime::report_task_progress(control, 1.5, "complete", 3, 3);
    require_eq(events.size(), std::size_t{2}, "progress event count");
    require(events[0].fraction == 0.0, "progress lower clamp");
    require(events[1].fraction == 1.0, "progress upper clamp");
    require(events[1].stage == "complete", "progress stage");
    require_eq(events[1].completed, int64_t{3}, "progress completed units");
    require_eq(events[1].total, int64_t{3}, "progress total units");
    require(!engine::runtime::task_cancellation_requested(control), "unexpected cancellation");

    control->cancellation_requested = []() -> bool {
        throw std::runtime_error("consumer callback failure");
    };
    require(
        engine::runtime::task_cancellation_requested(control),
        "throwing cancellation callback must fail closed");
}

void test_graph_abort_scope(
    engine::core::BackendType backend_type,
    int device,
    const std::string &label) {
    engine::core::BackendConfig config;
    config.type = backend_type;
    config.device = device;
    config.threads = 2;
    engine::core::ExecutionContext execution(config);

    ggml_init_params params{8 * 1024 * 1024, nullptr, true};
    std::unique_ptr<ggml_context, ContextDeleter> context(ggml_init(params));
    require(context != nullptr, label + " ggml context creation");
    ggml_tensor * input = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4096);
    ggml_tensor * output = input;
    for (int node = 0; node < 256; ++node) {
        output = ggml_sqr(context.get(), output);
    }
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(context.get(), 512, false);
    ggml_build_forward_expand(graph, output);
    std::unique_ptr<std::remove_pointer_t<ggml_backend_buffer_t>, BufferDeleter> buffer(
        ggml_backend_alloc_ctx_tensors(context.get(), execution.backend()));
    require(buffer != nullptr, label + " backend tensor allocation");

    std::atomic_int cancellation_checks{0};
    auto control = std::make_shared<engine::runtime::TaskExecutionControl>();
    control->cancellation_requested = [&] {
        return cancellation_checks.fetch_add(1, std::memory_order_acq_rel) > 0;
    };
    {
        engine::runtime::ScopedTaskExecutionControl scope(execution, control);
        require(
            scope.backend_cancellation_installed(),
            label + " abort callback installation");
        const ggml_status status =
            engine::core::compute_backend_graph(execution.backend(), graph);
        require_eq(status, GGML_STATUS_ABORTED, label + " cancelled graph status");
    }

    const ggml_status status =
        engine::core::compute_backend_graph(execution.backend(), graph);
    require_eq(status, GGML_STATUS_SUCCESS, label + " graph status after scope cleanup");
    ggml_backend_synchronize(execution.backend());
}

void test_graph_abort_scopes() {
    test_graph_abort_scope(engine::core::BackendType::Cpu, 0, "CPU");
    for (const auto &device : engine::core::list_backend_devices()) {
        if (device.backend == "MTL") {
            test_graph_abort_scope(
                engine::core::BackendType::Metal,
                device.index,
                "Metal");
            break;
        }
    }
}

}  // namespace

int main() {
    test_progress_and_cancellation_helpers();
    test_graph_abort_scopes();
    return 0;
}
