// Deterministic, torch-free lifetime/concurrency tests for KeyMutexRegistry.
//
// The property under test: a handle returned by acquire() stays valid and
// lockable even after clear() has erased the registry's own reference, and a
// clear() concurrent with a held lock can neither destroy the live mutex nor
// deadlock. This is the invariant that keeps clearArtifactCache() from causing a
// use-after-free on a per-key mutex a loading thread is still holding.
#include "key_mutex_registry.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <thread>  // NOLINT(build/c++11)

namespace {

// A retained handle survives clear() and remains lockable: clear() only drops
// the registry's reference, never the shared mutex a caller still owns.
void RetainedHandleSurvivesClear() {
  spyre::KeyMutexRegistry registry;

  std::shared_ptr<std::mutex> held = registry.acquire("program-a");
  assert(held != nullptr);
  assert(held.use_count() == 2);  // registry slot + local handle
  assert(registry.size() == 1);

  registry.clear();  // drops the registry's reference while we still hold one
  assert(registry.size() == 0);
  assert(held.use_count() == 1);  // only the local handle remains

  // The mutex is still alive and usable after the registry forgot it.
  {
    std::lock_guard<std::mutex> lock(*held);
  }
  std::cout << "RetainedHandleSurvivesClear: ok\n";
}

// Acquiring the same key twice returns the same underlying mutex; distinct keys
// return distinct mutexes.
void SameKeySameMutex() {
  spyre::KeyMutexRegistry registry;

  std::shared_ptr<std::mutex> a1 = registry.acquire("k1");
  std::shared_ptr<std::mutex> a2 = registry.acquire("k1");
  std::shared_ptr<std::mutex> b = registry.acquire("k2");

  assert(a1.get() == a2.get());
  assert(a1.get() != b.get());
  assert(registry.size() == 2);
  std::cout << "SameKeySameMutex: ok\n";
}

// A clear() racing a thread that holds a per-key lock must not destroy the live
// mutex nor deadlock: the holder keeps shared ownership for the whole critical
// section, exactly as a loading thread does against clearArtifactCache().
void ClearRacesHeldLock() {
  spyre::KeyMutexRegistry registry;
  std::atomic<bool> locked{false};
  std::atomic<bool> may_release{false};

  std::shared_ptr<std::mutex> handle = registry.acquire("busy");

  // The holder owns its own shared_ptr copy, exactly as a loading thread does.
  std::thread holder([handle, &locked, &may_release]() {
    std::lock_guard<std::mutex> lock(*handle);
    locked.store(true);
    while (!may_release.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });

  while (!locked.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // Clear while the holder is mid-critical-section. The registry drops its
  // reference, but both this handle and the holder's copy keep the mutex alive.
  registry.clear();
  assert(registry.size() == 0);
  assert(handle.use_count() == 2);  // local handle + holder thread's copy

  may_release.store(true);
  holder.join();

  // After the holder releases, the mutex (still owned by `handle`) is lockable.
  {
    std::lock_guard<std::mutex> lock(*handle);
  }
  std::cout << "ClearRacesHeldLock: ok\n";
}

// Many threads hammering acquire()/clear() concurrently must not crash or
// deadlock, and every acquired handle must be independently lockable.
void ConcurrentAcquireAndClear() {
  spyre::KeyMutexRegistry registry;
  std::atomic<bool> stop{false};

  std::thread clearer([&]() {
    while (!stop.load()) {
      registry.clear();
      std::this_thread::yield();
    }
  });

  std::thread workers[8];
  for (int t = 0; t < 8; ++t) {
    workers[t] = std::thread([&registry, t]() {
      const std::string key = "key-" + std::to_string(t % 3);
      for (int i = 0; i < 2000; ++i) {
        std::shared_ptr<std::mutex> m = registry.acquire(key);
        std::lock_guard<std::mutex> lock(*m);  // never a freed mutex
      }
    });
  }

  for (auto& w : workers) {
    w.join();
  }
  stop.store(true);
  clearer.join();
  std::cout << "ConcurrentAcquireAndClear: ok\n";
}

}  // namespace

int main() {
  RetainedHandleSurvivesClear();
  SameKeySameMutex();
  ClearRacesHeldLock();
  ConcurrentAcquireAndClear();
  std::cout << "key_mutex_registry_test: all lifetime/concurrency checks passed\n";
  return 0;
}
