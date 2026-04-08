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

#include "spyre_device_enum.h"

#include <dirent.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "logging.h"

namespace spyre {

namespace detail {

// Read a hex value from a sysfs file, e.g. /sys/bus/pci/devices/XXX/vendor
// Returns -1 on failure.
int readSysfsHex(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) return -1;
  int val = -1;
  f >> std::hex >> val;
  return val;
}

// Scan AIU_WORLD_RANK_* env vars for device discovery.
// These are set by login scripts in all environments (OpenShift & bare metal).
// Returns PCI bus IDs in rank order, stopping at the first gap.
std::vector<std::string> scanEnvRanks() {
  std::vector<std::string> bus_ids;
  for (int i = 0;; ++i) {
    std::string key = "AIU_WORLD_RANK_" + std::to_string(i);
    const char* val = std::getenv(key.c_str());
    if (!val || val[0] == '\0') break;
    bus_ids.push_back(val);
  }
  return bus_ids;
}

// Scan /sys/bus/pci/devices/ for all Spyre accelerators.
// Returns PCI bus IDs sorted lexicographically.
std::vector<std::string> scanPciBus() {
  std::vector<std::string> bus_ids;
  const std::string sysfs_path = "/sys/bus/pci/devices";
  DIR* dir = opendir(sysfs_path.c_str());
  if (!dir) {
    DEBUGINFO("spyre_device_enum: cannot open", sysfs_path);
    return bus_ids;
  }

  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name(entry->d_name);
    if (name == "." || name == "..") continue;

    std::string dev_dir = sysfs_path + "/" + name;
    int vendor = readSysfsHex(dev_dir + "/vendor");
    int device = readSysfsHex(dev_dir + "/device");

    if (vendor == kSpyreVendorId && device == kSpyreDeviceId) {
      bus_ids.push_back(name);
    }
  }
  closedir(dir);

  std::sort(bus_ids.begin(), bus_ids.end());
  return bus_ids;
}

// Parse SPYRE_VISIBLE_DEVICES env var.
// Accepts comma-separated PCI bus IDs or 0-based integer indices.
// Returns resolved PCI bus IDs.
std::vector<std::string> parseVisibleDevices(
    const std::string& env_val, const std::vector<std::string>& all_bus_ids) {
  std::vector<std::string> result;
  std::istringstream ss(env_val);
  std::string token;

  while (std::getline(ss, token, ',')) {
    // Trim whitespace
    while (!token.empty() && token.front() == ' ') token.erase(0, 1);
    while (!token.empty() && token.back() == ' ') token.pop_back();
    if (token.empty()) continue;

    // Check if token is a plain integer (index) or a PCI bus ID
    bool is_index =
        !token.empty() && std::all_of(token.begin(), token.end(), ::isdigit);

    if (is_index) {
      int idx = std::stoi(token);
      if (idx >= 0 && idx < static_cast<int>(all_bus_ids.size())) {
        result.push_back(all_bus_ids[idx]);
      } else {
        DEBUGINFO("spyre_device_enum: SPYRE_VISIBLE_DEVICES index", idx,
                  "out of range (", all_bus_ids.size(), "devices found)");
      }
    } else {
      // Assume it's a PCI bus ID — validate it exists
      if (std::find(all_bus_ids.begin(), all_bus_ids.end(), token) !=
          all_bus_ids.end()) {
        result.push_back(token);
      } else {
        DEBUGINFO("spyre_device_enum: SPYRE_VISIBLE_DEVICES bus ID", token,
                  "not found among Spyre devices");
      }
    }
  }
  return result;
}

std::vector<SpyreDeviceInfo> buildDeviceList() {
  // Step 1: Discover all available devices.
  // Priority: AIU_WORLD_RANK_* > PCIDEVICE_IBM_COM_AIU_PF > PCI bus scan
  std::vector<std::string> all_bus_ids = scanEnvRanks();
  const char* discovery_source = nullptr;

  if (!all_bus_ids.empty()) {
    discovery_source = "AIU_WORLD_RANK_*";
  } else {
    const char* k8s_env = std::getenv("PCIDEVICE_IBM_COM_AIU_PF");
    if (k8s_env && k8s_env[0] != '\0') {
      // Validate K8s bus IDs against PCI scan — filters out any IDs
      // that don't match real hardware on this host.
      std::vector<std::string> pci_bus_ids = scanPciBus();
      all_bus_ids = parseVisibleDevices(k8s_env, pci_bus_ids);
      discovery_source = "PCIDEVICE_IBM_COM_AIU_PF";
    } else {
      all_bus_ids = scanPciBus();
      discovery_source = "PCI bus scan";
    }
  }

  DEBUGINFO("spyre_device_enum:", discovery_source, "found", all_bus_ids.size(),
            "devices");

  // Step 2: Apply SPYRE_VISIBLE_DEVICES filter.
  std::vector<std::string> visible_bus_ids;
  const char* env = std::getenv("SPYRE_VISIBLE_DEVICES");

  if (env && env[0] != '\0') {
    visible_bus_ids = parseVisibleDevices(env, all_bus_ids);
    DEBUGINFO("spyre_device_enum: SPYRE_VISIBLE_DEVICES =", env, "->",
              visible_bus_ids.size(), "devices");
  } else {
    visible_bus_ids = all_bus_ids;
  }

  // Step 3: Build device info list.
  std::vector<SpyreDeviceInfo> devices;
  devices.reserve(visible_bus_ids.size());
  for (int i = 0; i < static_cast<int>(visible_bus_ids.size()); ++i) {
    devices.push_back({visible_bus_ids[i], i});
  }
  return devices;
}

}  // namespace detail

const std::vector<SpyreDeviceInfo>& getVisibleDevices() {
  static std::once_flag flag;
  static std::vector<SpyreDeviceInfo> devices;
  std::call_once(flag, [&]() { devices = detail::buildDeviceList(); });
  return devices;
}

int getVisibleDeviceCount() {
  return static_cast<int>(getVisibleDevices().size());
}

void ensureSpyreDevicesEnv() {
  // If SPYRE_DEVICES is already set, flex will use it directly — nothing to do.
  const char* existing = std::getenv("SPYRE_DEVICES");
  if (existing && existing[0] != '\0') {
    DEBUGINFO("spyre_device_enum: SPYRE_DEVICES already set =", existing);
    return;
  }

  const auto& visible = getVisibleDevices();
  if (visible.empty()) return;

  // Set AIU_WORLD_RANK_<i> env vars with the PCI bus IDs of visible devices.
  // flex's CreatePciId reads RANK (set by torchrun), then calls
  // RdmaGetPCIeAddress(rank) which checks AIU_WORLD_RANK_<id> and passes
  // the PCI bus ID directly to senlib::SenPci::pf(pci_address).
  //
  // overwrite=1: SPYRE_VISIBLE_DEVICES may have re-indexed the list, so
  // the mapping from index to PCI bus ID may differ from what the login
  // script originally set.
  for (const auto& dev : visible) {
    std::string env_name = "AIU_WORLD_RANK_" + std::to_string(dev.index);
    setenv(env_name.c_str(), dev.pci_bus_id.c_str(), /*overwrite=*/1);
    DEBUGINFO("spyre_device_enum: set", env_name, "=", dev.pci_bus_id);
  }

  // Set SPYRE_DEVICES as sequential indices so flex's
  // RdmaGetPCIeAddress uses the correct rank-to-index mapping.
  std::string spyre_devices;
  for (int i = 0; i < static_cast<int>(visible.size()); ++i) {
    if (!spyre_devices.empty()) spyre_devices += ',';
    spyre_devices += std::to_string(i);
  }
  setenv("SPYRE_DEVICES", spyre_devices.c_str(), /*overwrite=*/0);
  DEBUGINFO("spyre_device_enum: synthesized SPYRE_DEVICES =", spyre_devices);
}

}  // namespace spyre
