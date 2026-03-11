#!/usr/bin/env python3
"""MVCC — Multi-Version Concurrency Control database engine.

Implements snapshot isolation with: versioned rows, transaction timestamps,
read snapshots, write-write conflict detection, garbage collection.

Usage: python mvcc2.py [--test]
"""

import sys, threading
from collections import defaultdict

class Version:
    def __init__(self, value, created_by, deleted_by=None):
        self.value = value
        self.created_by = created_by  # txn_id
        self.deleted_by = deleted_by  # txn_id or None

class Transaction:
    _next_id = 0
    _lock = threading.Lock()
    
    def __init__(self, engine):
        with Transaction._lock:
            Transaction._next_id += 1
            self.txn_id = Transaction._next_id
        self.engine = engine
        self.snapshot = set(engine.committed_txns)  # visible txns at start
        self.writes = {}  # key -> value
        self.deletes = set()
        self.reads = set()  # keys read
        self.committed = False
        self.aborted = False
    
    def get(self, key):
        """Read a key — returns latest visible version."""
        self.reads.add(key)
        # Check own writes first
        if key in self.deletes:
            return None
        if key in self.writes:
            return self.writes[key]
        # Find latest visible version
        versions = self.engine.store.get(key, [])
        for v in reversed(versions):
            if v.created_by in self.snapshot or v.created_by == self.txn_id:
                if v.deleted_by is None or (v.deleted_by not in self.snapshot and v.deleted_by != self.txn_id):
                    return v.value
        return None
    
    def put(self, key, value):
        """Write a key."""
        if self.committed or self.aborted:
            raise RuntimeError("Transaction already ended")
        self.writes[key] = value
        self.deletes.discard(key)
    
    def delete(self, key):
        """Delete a key."""
        self.deletes.add(key)
        self.writes.pop(key, None)
    
    def commit(self):
        """Commit transaction with write-write conflict check."""
        with self.engine.lock:
            # Check for write-write conflicts
            for key in self.writes:
                for other_txn in self.engine.committed_txns - self.snapshot:
                    if key in self.engine.txn_writes.get(other_txn, set()):
                        self.aborted = True
                        raise RuntimeError(f"Write-write conflict on key '{key}' with txn {other_txn}")
            for key in self.deletes:
                for other_txn in self.engine.committed_txns - self.snapshot:
                    if key in self.engine.txn_writes.get(other_txn, set()):
                        self.aborted = True
                        raise RuntimeError(f"Write-write conflict on key '{key}'")
            
            # Apply writes
            for key, value in self.writes.items():
                self.engine.store.setdefault(key, [])
                # Mark old versions as deleted
                for v in self.engine.store[key]:
                    if v.deleted_by is None and v.created_by in (self.snapshot | {self.txn_id}):
                        v.deleted_by = self.txn_id
                self.engine.store[key].append(Version(value, self.txn_id))
            
            for key in self.deletes:
                if key in self.engine.store:
                    for v in self.engine.store[key]:
                        if v.deleted_by is None:
                            v.deleted_by = self.txn_id
            
            self.engine.committed_txns.add(self.txn_id)
            self.engine.txn_writes[self.txn_id] = set(self.writes.keys()) | self.deletes
            self.committed = True
    
    def rollback(self):
        self.aborted = True
        self.writes.clear()
        self.deletes.clear()

class MVCCEngine:
    def __init__(self):
        self.store = defaultdict(list)  # key -> [Version]
        self.committed_txns = set()
        self.txn_writes = {}  # txn_id -> set of keys written
        self.lock = threading.Lock()
    
    def begin(self):
        return Transaction(self)
    
    def gc(self, oldest_active_txn=None):
        """Garbage collect old versions no longer visible to any transaction."""
        if oldest_active_txn is None:
            oldest_active_txn = max(self.committed_txns) if self.committed_txns else 0
        
        removed = 0
        for key in list(self.store.keys()):
            versions = self.store[key]
            # Keep latest version and any version visible to active transactions
            new_versions = []
            for v in versions:
                if v.deleted_by is not None and v.deleted_by <= oldest_active_txn:
                    removed += 1
                    continue
                new_versions.append(v)
            self.store[key] = new_versions
            if not new_versions:
                del self.store[key]
        return removed
    
    def snapshot_read(self, key):
        """Read latest committed value (no transaction needed)."""
        txn = self.begin()
        txn.snapshot = set(self.committed_txns)
        return txn.get(key)

# --- Tests ---

def test_basic_crud():
    db = MVCCEngine()
    t1 = db.begin()
    t1.put("name", "Alice")
    t1.put("age", 30)
    t1.commit()
    
    t2 = db.begin()
    assert t2.get("name") == "Alice"
    assert t2.get("age") == 30
    assert t2.get("missing") is None

def test_snapshot_isolation():
    db = MVCCEngine()
    t1 = db.begin()
    t1.put("x", 1)
    t1.commit()
    
    # t2 starts before t3 commits
    t2 = db.begin()
    
    t3 = db.begin()
    t3.put("x", 2)
    t3.commit()
    
    # t2 should still see old value (snapshot isolation)
    assert t2.get("x") == 1
    
    # New transaction sees updated value
    t4 = db.begin()
    assert t4.get("x") == 2

def test_write_write_conflict():
    db = MVCCEngine()
    t1 = db.begin()
    t1.put("key", "v1")
    t1.commit()
    
    t2 = db.begin()
    t3 = db.begin()
    
    t2.put("key", "v2")
    t2.commit()
    
    t3.put("key", "v3")
    try:
        t3.commit()
        assert False, "Should have detected conflict"
    except RuntimeError as e:
        assert "conflict" in str(e).lower()

def test_delete():
    db = MVCCEngine()
    t1 = db.begin()
    t1.put("x", 100)
    t1.commit()
    
    t2 = db.begin()
    t2.delete("x")
    t2.commit()
    
    t3 = db.begin()
    assert t3.get("x") is None

def test_rollback():
    db = MVCCEngine()
    t1 = db.begin()
    t1.put("x", 1)
    t1.commit()
    
    t2 = db.begin()
    t2.put("x", 999)
    t2.rollback()
    
    t3 = db.begin()
    assert t3.get("x") == 1

def test_read_own_writes():
    db = MVCCEngine()
    t1 = db.begin()
    t1.put("a", 1)
    assert t1.get("a") == 1  # see own write before commit
    t1.put("a", 2)
    assert t1.get("a") == 2
    t1.commit()

def test_gc():
    db = MVCCEngine()
    for i in range(10):
        t = db.begin()
        t.put("counter", i)
        t.commit()
    
    assert len(db.store["counter"]) == 10
    removed = db.gc()
    assert removed >= 8  # old versions cleaned up

def test_concurrent_reads():
    db = MVCCEngine()
    t1 = db.begin()
    t1.put("shared", "initial")
    t1.commit()
    
    readers = []
    for _ in range(5):
        t = db.begin()
        readers.append(t)
    
    # Writer commits while readers are active
    tw = db.begin()
    tw.put("shared", "updated")
    tw.commit()
    
    # All readers still see old value
    for r in readers:
        assert r.get("shared") == "initial"

if __name__ == "__main__":
    if "--test" in sys.argv or len(sys.argv) == 1:
        test_basic_crud()
        test_snapshot_isolation()
        test_write_write_conflict()
        test_delete()
        test_rollback()
        test_read_own_writes()
        test_gc()
        test_concurrent_reads()
        print("All tests passed!")
