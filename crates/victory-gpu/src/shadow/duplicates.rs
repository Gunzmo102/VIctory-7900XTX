//! Diagnostics for redundant UMA shadows and repeated vertex/index data.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use super::cover;

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UMA_DUPLICATES").is_some_and(|v| v == "1"))
}

static REUSED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

pub(super) fn alias_reused(bytes: u64) {
    if enabled() {
        REUSED[0].fetch_add(1, Ordering::Relaxed);
        REUSED[1].fetch_add(bytes, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum Kind {
    Vertex(u32),
    Index(u32),
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct View {
    at: u64,
    bytes: u32,
    kind: Kind,
}

const MAX_SAMPLES: usize = 256;
const MAX_VIEW: u32 = 2 << 20;
const BUDGET: u64 = 32 << 20;
static VIEWS: Mutex<(BTreeSet<View>, u64)> = Mutex::new((BTreeSet::new(), 0));

/// Opt-in samples of the title ranges used as ordinary vertex/index views.
/// Repeated draws of the same view consume no additional budget.
pub(crate) fn mesh(at: u64, bytes: u32, kind: Kind) {
    if !enabled() || at == 0 || bytes == 0 || bytes > MAX_VIEW {
        return;
    }
    let mut views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    if views.0.len() < MAX_SAMPLES
        && views.1 + u64::from(bytes) <= BUDGET
        && views.0.insert(View { at, bytes, kind })
    {
        views.1 += u64::from(bytes);
    }
}

#[derive(Default)]
struct MeshSummary {
    matches: u64,
    bytes: u64,
    example: Option<[u64; 2]>,
}

fn summarize(samples: impl Iterator<Item = (View, blake3::Hash)>) -> MeshSummary {
    let mut first = HashMap::new();
    let mut out = MeshSummary::default();
    for (v, hash) in samples {
        let at = *first.entry((v.kind, v.bytes, hash)).or_insert(v.at);
        if at != v.at {
            out.matches += 1;
            out.bytes += u64::from(v.bytes);
            out.example.get_or_insert([at, v.at]);
        }
    }
    out
}

// A shadow can outlive its cache entry while a list still holds it.
static COPIES: Mutex<BTreeMap<u64, (u64, u64)>> = Mutex::new(BTreeMap::new());

pub(super) fn made(id: u64, at: u64, bytes: u64) {
    if enabled() {
        COPIES
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, (at, bytes));
    }
}

pub(super) fn dropped(id: u64) {
    if enabled() {
        COPIES.lock().unwrap_or_else(|e| e.into_inner()).remove(&id);
    }
}

fn overlapping_bytes(spans: &mut [(u64, u64)]) -> u64 {
    spans.sort_unstable();
    let total: u64 = spans.iter().map(|s| s.1).sum();
    total - cover(spans.iter().copied())
}

pub(crate) fn report() {
    if !enabled() {
        return;
    }
    let views = {
        let mut views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *views).0
    };
    let mut scratch = Vec::new();
    let mut sampled = 0;
    let mesh = summarize(views.into_iter().filter_map(|v| {
        scratch.resize(v.bytes as usize, 0);
        // Read via the kernel: a title allocation can be unmapped between
        // recording and sampling. No GPU readback or title write is needed.
        if !crate::host::read_into(v.at as *const u8, &mut scratch)
            || scratch.iter().all(|&b| b == 0)
        {
            return None;
        }
        sampled += 1;
        Some((v, blake3::hash(&scratch)))
    }));
    let mut spans: Vec<_> = COPIES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .values()
        .copied()
        .collect();
    let shadow_copies = spans.len();
    let overlapping_bytes = overlapping_bytes(&mut spans);
    tracing::info!(
        shadow_copies,
        overlapping_bytes,
        aliases_reused = REUSED[0].swap(0, Ordering::Relaxed),
        alias_bytes_reused = REUSED[1].swap(0, Ordering::Relaxed),
        mesh_views_sampled = sampled,
        matching_mesh_views = mesh.matches,
        matching_mesh_view_bytes = mesh.bytes,
        example_title_addresses = ?mesh.example,
        "UMA duplicate check"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_overlapping_bytes_count_as_extra_backing() {
        let mut spans = [(0x100, 0x80), (0x110, 0x20), (0x140, 0x80), (0x300, 0x80)];
        assert_eq!(overlapping_bytes(&mut spans), 0x60);
    }

    #[test]
    fn matching_mesh_views_require_distinct_addresses_and_the_same_layout() {
        let hash = blake3::hash(b"mesh");
        let mesh = View {
            at: 0x1000,
            bytes: 4,
            kind: Kind::Vertex(4),
        };
        let same = View { at: 0x2000, ..mesh };
        let indices = View {
            at: 0x3000,
            kind: Kind::Index(16),
            ..mesh
        };
        let other_stride = View {
            at: 0x4000,
            kind: Kind::Vertex(8),
            ..mesh
        };
        let changed = View { at: 0x5000, ..mesh };
        let stats = summarize(
            [
                (mesh, hash),
                (mesh, hash),
                (indices, hash),
                (other_stride, hash),
                (changed, blake3::hash(b"different")),
                (same, hash),
            ]
            .into_iter(),
        );
        assert_eq!(stats.matches, 1);
        assert_eq!(stats.bytes, 4);
        assert_eq!(stats.example, Some([mesh.at, same.at]));
    }
}
