/*
 * Copyright 2026 The Torch-Spyre Authors.
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
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace spyre {

// PCI vendor/device IDs for IBM Spyre Accelerator
constexpr uint16_t kSpyreVendorId = 0x1014;  // IBM
constexpr uint16_t kSpyreDeviceId = 0x06a7;  // Spyre Accelerator

// PCI bus ID, e.g. "0000:29:00.0"
struct SpyreDeviceInfo {
  std::string pci_bus_id;
  int index;  // logical index (0-based)
};

// Returns the list of Spyre devices visible to this process.
//
// Discovery priority (full device list):
//   1. AIU_WORLD_RANK_* env vars — set by login scripts in all environments,
//      scanned as _0, _1, ... until a gap
//   2. PCIDEVICE_IBM_COM_AIU_PF env var — set by K8s device plugin,
//      comma-separated PCI bus IDs
//   3. Full PCI bus scan via /sys/bus/pci/devices/
//
// Filter (applied on top of the discovered list):
//   SPYRE_VISIBLE_DEVICES env var — comma-separated PCI bus IDs or
//   0-based indices (e.g. "0,1" or "0000:29:00.0,0000:2a:00.0")
//
// The result is cached after the first call.
const std::vector<SpyreDeviceInfo>& getVisibleDevices();

// Convenience: returns the number of visible Spyre devices.
int getVisibleDeviceCount();

// Ensure AIU_WORLD_RANK_* and SPYRE_DEVICES env vars are set correctly
// for flex, based on the visible device list from getVisibleDevices().
//
// If SPYRE_DEVICES is already set, this is a no-op.  Otherwise it
// overwrites AIU_WORLD_RANK_<i> to match the (potentially filtered)
// visible device list and sets SPYRE_DEVICES to sequential indices.
//
// Must be called before flex::initializeRuntime().
void ensureSpyreDevicesEnv();

}  // namespace spyre
