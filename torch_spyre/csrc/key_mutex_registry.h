// Lifetime-safe registry of per-key mutexes.
//
// A kernel-artifact cache serializes the loading of each key behind its own
// mutex so that only one thread copies a given program to the device. The
// per-key mutexes must outlive any thread currently holding them, even while a
// concurrent cache clear erases the registry: otherwise a thread could lock a
// mutex that another thread's clear has already destroyed (use-after-free).
//
// Each key maps to a std::shared_ptr<std::mutex>. acquire() inserts (if absent)
// and returns the handle under a unique lock; the caller keeps that handle for
// the whole per-key critical section, so a concurrent clear() that drops the
// map's reference cannot destroy the mutex out from under the holder. The
// registry's own map is guarded by an internal shared_mutex, so acquire()/clear()
// are themselves thread-safe.
#ifndef TORCH_SPYRE_CSRC_KEY_MUTEX_REGISTRY_H_
#define TORCH_SPYRE_CSRC_KEY_MUTEX_REGISTRY_H_

#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace spyre {

class KeyMutexRegistry {
 public:
  // Return a shared handle to the per-key mutex, creating it if necessary.
  // Hold the returned shared_ptr for the entire per-key critical section: it
  // keeps the mutex alive even if clear() concurrently erases the map entry.
  std::shared_ptr<std::mutex> acquire(const std::string& key) {
    std::unique_lock<std::shared_mutex> lock(map_mtx_);
    std::shared_ptr<std::mutex>& slot = mutexes_[key];
    if (!slot) {
      slot = std::make_shared<std::mutex>();
    }
    return slot;
  }

  // Drop every per-key mutex map entry. Any handle a caller already holds stays
  // valid through shared ownership; only the registry's reference is released.
  void clear() {
    std::unique_lock<std::shared_mutex> lock(map_mtx_);
    mutexes_.clear();
  }

  // Number of keys currently registered (test/diagnostic aid).
  std::size_t size() const {
    std::shared_lock<std::shared_mutex> lock(map_mtx_);
    return mutexes_.size();
  }

 private:
  mutable std::shared_mutex map_mtx_;
  std::unordered_map<std::string, std::shared_ptr<std::mutex>> mutexes_;
};

}  // namespace spyre

#endif  // TORCH_SPYRE_CSRC_KEY_MUTEX_REGISTRY_H_
