"""Bounded session memoization with shared in-flight work and epoch invalidation."""
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, CancelledError
from copy import deepcopy
import hashlib
import json
from threading import Lock


def cache_key(namespace, value):
    return namespace, hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class SessionMemo:
    def __init__(self, capacity=128, workers=2):
        self.capacity, self.workers = capacity, workers
        self.entries = OrderedDict()
        self.lock = Lock()
        self.epoch = 0
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='book-retrieval')

    def clear(self):
        with self.lock:
            self.epoch += 1
            self.entries.clear()

    def _entry(self, key, background=False):
        with self.lock:
            if key in self.entries:
                future = self.entries[key]
                self.entries.move_to_end(key)
                return future, False, self.epoch, 'hit' if future.done() else 'inflight'
            if background and sum(not f.done() for f in self.entries.values()) >= self.workers:
                return None, False, self.epoch, 'busy'
            future = Future()
            self.entries[key] = future
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
            return future, True, self.epoch, 'miss'

    def _compute(self, key, future, epoch, factory, cache_if):
        try:
            with self.lock:
                if epoch != self.epoch:
                    raise CancelledError('Session changed before prefetched work started')
            result = factory()
            reusable = cache_if(result) if cache_if else True
        except BaseException as error:
            with self.lock:
                if self.entries.get(key) is future:
                    del self.entries[key]
            future.set_exception(error)
        else:
            with self.lock:
                if (not reusable or epoch != self.epoch) and self.entries.get(key) is future:
                    del self.entries[key]
            future.set_result(result)

    def get(self, key, factory, cache_if=None):
        future, owner, epoch, status = self._entry(key)
        if owner:
            # Synchronous leaders avoid nested worker-pool deadlocks.
            self._compute(key, future, epoch, factory, cache_if)
        return deepcopy(future.result()), status

    def prefetch(self, key, factory, cache_if=None):
        future, owner, epoch, status = self._entry(key, background=True)
        if owner:
            self.executor.submit(self._compute, key, future, epoch, factory, cache_if)
            return 'scheduled'
        return status

    def close(self):
        self.clear()
        self.executor.shutdown(wait=True)


MEMO = SessionMemo()
