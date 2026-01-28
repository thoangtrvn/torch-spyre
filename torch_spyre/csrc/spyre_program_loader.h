/************************************************************
* Copyright 2025 The Torch-Spyre Authors.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* Utility to extract programs from a sendnn::Graph and load them
* to device memory using direct DMA (bypassing graph execution).
************************************************************/

#ifndef TORCH_SPYRE_CSRC_SPYRE_PROGRAM_LOADER_HPP_
#define TORCH_SPYRE_CSRC_SPYRE_PROGRAM_LOADER_HPP_

#include <string>
#include <vector>

#include <common/logging.hpp>
#include <sendnn/graph.hpp>
#include <sendnn/graph/graph_deserializer.hpp>
#include <sendnn/graph/senparms.hpp>
#include <sendnn/ops.hpp>

#include "module.h"
// #include "direct_program_dma.hpp"

namespace spyre {

/**
 * Information about a program extracted from a graph
 */
struct ProgramInfo {
  std::string name;           // Program name
  const void* data;           // Pointer to program data (in Const node)
  uint64_t size_bytes;        // Size of program in bytes
  uint64_t device_dmva;       // Destination DMVA on device
};

/**
 * Result of loading programs to device
 */
struct ProgramLoadResult {
  bool success = false;
  std::string error_message;
  std::vector<ProgramInfo> programs_loaded;
  double total_duration_ms = 0.0;
};

/**
 * Extract program information from a DeviceInit SuperNode's execution graph
 *
 * The DeviceInit SuperNode contains:
 *   - Const nodes with program binary data
 *   - SenDataTransfer nodes that specify where to copy them
 *
 * @param exec_graph The execution_graph_ from SenSuperNodeV2 (DeviceInit)
 * @return Vector of ProgramInfo structures
 */
inline auto ExtractProgramsFromExecGraph(const sendnn::SubGraph& exec_graph)
    -> std::vector<ProgramInfo> {

  std::vector<ProgramInfo> programs;

  // Map Const node names to their data
  std::unordered_map<std::string, std::pair<const void*, uint64_t>> const_data_map;

  // First pass: collect all Const nodes
  for (const auto* node : exec_graph.compute_ops_) {
    if (node->Fn() == sendnn::opcodes::Const) {
      auto* const_attrs = node->Attrs<sendnn::attributes::Const>();
      if (const_attrs && const_attrs->data_.NumBytes() > 0) {
        std::pair<const void*, long unsigned int> p(const_attrs->data_.Data(), const_attrs->data_.NumBytes());
        const_data_map[node->Name()] = p;
        LOG_DEBUG << "Found Const node: " << node->Name()
                  << " with " << const_attrs->data_.NumBytes() << " bytes";
      }
    }
  }

  // Second pass: find SenDataTransfer nodes that reference Const nodes
  for (const auto* node : exec_graph.compute_ops_) {
    if (node->Fn() == sendnn::opcodes::SenDataTransfer) {
      auto* transfer_attrs = node->Attrs<sendnn::attributes::SenDataTransfer>();
      if (!transfer_attrs) continue;

      // Check if input comes from a Const node
      if (node->NumInputs() > 0) {
        auto* pred_node = node->PredecessorAt(0);
        if (pred_node) {
          if (pred_node && pred_node->Fn() == sendnn::opcodes::Const) {
            // This is a program transfer
            auto it = const_data_map.find(pred_node->Name());
            if (it != const_data_map.end()) {
              // Get destination DMVA from output tensor
              uint64_t dmva = 0;
              if (node->NumOutputs() > 0) {
                auto output_tensor = node->OutputAt(0);
                dmva = reinterpret_cast<uint64_t>(output_tensor.Data());
              }

              ProgramInfo info;
              info.name = pred_node->Name();
              info.data = it->second.first;
              info.size_bytes = transfer_attrs->size_bytes_;
              info.device_dmva = dmva;

              programs.push_back(info);

              LOG_INFO << "Found program transfer: " << info.name
                       << " size=" << info.size_bytes
                       << " -> DMVA=0x" << std::hex << info.device_dmva << std::dec;
            }
          }
        }
      }
    }
  }

  return programs;
}

/**
 * Extract all programs from a g2 graph's DeviceInit SuperNode
 *
 * @param g2 The compiled g2 graph
 * @return Vector of ProgramInfo structures
 */
inline auto ExtractProgramsFromGraph(const sendnn::Graph& g2)
    -> std::vector<ProgramInfo> {

  std::vector<ProgramInfo> all_programs;

  for (const auto* super_node : g2.compute_ops_) {
    if (super_node->Name() == "DeviceInit") {
      auto* sn_attrs = super_node->Attrs<sendnn::attributes::SenSuperNodeV2>();
      if (sn_attrs) {
        auto programs = ExtractProgramsFromExecGraph(sn_attrs->execution_graph_);
        all_programs.insert(all_programs.end(), programs.begin(), programs.end());
      }
    }
  }

  return all_programs;
}

// /**
//  * Load programs from a graph to device using direct DMA
//  *
//  * This extracts program binaries from the graph's DeviceInit SuperNode
//  * and transfers them to device memory using direct DMA, bypassing
//  * the normal graph execution flow.
//  *
//  * @param g2 The compiled g2 graph containing DeviceInit
//  * @param device_idx Target device index (default 0)
//  * @return ProgramLoadResult with status and info about loaded programs
//  *
//  * Example usage:
//  *   sendnn::Graph g2;
//  *   sendnn::Deserialize(&g2, "/path/to/g2");
//  *   auto result = spyre::LoadProgramsFromGraphDirect(g2);
//  *   if (!result.success) {
//  *       std::cerr << "Failed: " << result.error_message << std::endl;
//  *   }
//  */
// inline auto LoadProgramsFromGraphDirect(const sendnn::Graph& g2,
//                                         int64_t device_idx = 0)
//     -> ProgramLoadResult {

//   ProgramLoadResult result;

//   // Ensure runtime is started
//   auto runtime = GlobalRuntime::get();
//   if (!runtime) {
//     result.error_message = "Runtime not initialized. Call startRuntime() first.";
//     return result;
//   }

//   // Extract programs from graph
//   auto programs = ExtractProgramsFromGraph(g2);
//   if (programs.empty()) {
//     result.error_message = "No programs found in graph's DeviceInit SuperNode";
//     return result;
//   }

//   LOG_INFO << "Found " << programs.size() << " programs to load";

//   // Transfer each program using direct DMA
//   for (const auto& prog : programs) {
//     LOG_INFO << "Loading program: " << prog.name
//              << " (" << prog.size_bytes << " bytes)"
//              << " to DMVA 0x" << std::hex << prog.device_dmva << std::dec;

//     auto dma_result = TransferProgramToDevice(
//         prog.data,
//         prog.device_dmva,
//         prog.size_bytes,
//         device_idx
//     );

//     if (!dma_result.success) {
//       result.error_message = "Failed to load program " + prog.name +
//                             ": " + dma_result.error_message;
//       return result;
//     }

//     result.programs_loaded.push_back(prog);
//     result.total_duration_ms += dma_result.duration_ms;
//   }

//   result.success = true;
//   LOG_INFO << "Successfully loaded " << result.programs_loaded.size()
//            << " programs in " << result.total_duration_ms << " ms";

//   return result;
// }

// /**
//  * Load programs from a serialized g2 file to device using direct DMA
//  *
//  * @param g2_path Path to serialized g2 file
//  * @param device_idx Target device index (default 0)
//  * @return ProgramLoadResult with status and info about loaded programs
//  *
//  * Example usage:
//  *   auto result = spyre::LoadProgramsFromFileDirect("/path/to/g2");
//  */
// inline auto LoadProgramsFromFileDirect(const std::string& g2_path,
//                                        int64_t device_idx = 0)
//     -> ProgramLoadResult {

//   ProgramLoadResult result;

//   // Load the g2 graph
//   sendnn::Graph g2;
//   try {
//     sendnn::Deserialize(&g2, g2_path);
//   } catch (const std::exception& e) {
//     result.error_message = "Failed to deserialize g2 from " + g2_path +
//                           ": " + e.what();
//     return result;
//   }

//   return LoadProgramsFromGraphDirect(g2, device_idx);
// }

// /**
//  * Example: Modify launchKernel to use direct program loading
//  *
//  * This demonstrates how to use the direct DMA approach within
//  * the existing launchKernel workflow.
//  */
// inline void launchKernelWithDirectProgramLoad(const std::string& g2_path,
//                                               std::vector<at::Tensor>& args) {
//   // Start runtime if not already started
//   startRuntime();

//   // Load and deserialize the graph
//   sendnn::Graph g2;
//   sendnn::Deserialize(&g2, g2_path);

//   // Option 1: Load programs using direct DMA (bypassing graph execution)
//   auto program_result = LoadProgramsFromGraphDirect(g2);
//   if (!program_result.success) {
//     throw std::runtime_error("Program loading failed: " +
//                             program_result.error_message);
//   }

//   LOG_INFO << "Loaded " << program_result.programs_loaded.size()
//            << " programs via direct DMA";

//   // At this point, programs are loaded to device.
//   // The rest of the workflow (partition init, compute, etc.) would
//   // still need to be handled, but this demonstrates how to extract
//   // and directly load the program binaries.

//   // Option 2: For a complete example, you could also extract model weights
//   // and load them separately using the same direct DMA approach.

//   // Note: In a full implementation, you would need to also:
//   // 1. Create/initialize the partition (segment table setup)
//   // 2. Allocate frames for compute
//   // 3. Execute compute operations
//   // These still require the graph-based approach or additional
//   // standalone utilities.
// }

// /**
//  * Get program info from a graph without loading (for inspection)
//  *
//  * @param g2_path Path to serialized g2 file
//  * @return Vector of ProgramInfo structures
//  */
// inline auto InspectProgramsInGraph(const std::string& g2_path)
//     -> std::vector<ProgramInfo> {

//   sendnn::Graph g2;
//   sendnn::Deserialize(&g2, g2_path);
//   return ExtractProgramsFromGraph(g2);
// }

}  // namespace spyre

#endif  // TORCH_SPYRE_CSRC_SPYRE_PROGRAM_LOADER_HPP_
