//! What goes up to copies in the card's memory goes up as it was when its
//! submission was prepared (`VICTORY_SEND_OWN`, on by default; `0`: as
//! before).
//!
//! The sends were copies out of the copies' CPU sides, which the card makes
//! when it comes to the submission. By then later submissions are prepared
//! (two frames of them with `VICTORY_RELAY`), and their refreshes have
//! written the same CPU sides: the earlier submission's shaders then read a
//! later one's bytes. For what the title's CPU writes that is as on the
//! console (it does not write what the card still reads). But the title's
//! own buffer copies are commands in its lists, and those are done by us
//! when a submission is prepared: "a value to a buffer, a draw that reads
//! it" in one submission and the same with another value in the next is
//! ordinary, and both draws read the second value.
//!
//! So each submission's bytes are taken out of the CPU sides into memory of
//! the submission's own when they are taken from the dirty lists, and the
//! card copies them from there. Runs of a few words, which a refresh that
//! sends only what changed makes many of, share a dispatch per buffer.
//! Groups of four words share one destination offset in the upload;
//! short tails use offset/value pairs. Other runs use buffer copies.

use super::*;
use crate::shadow::Shadow;

pub(super) fn own() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SEND_OWN").is_none_or(|v| v != "0"))
}

/// Runs up to this many bytes go through the scatter shader.
const SHORT: usize = 256;

/// Whether a run of `len` bytes at offset `at` of a copy uses scatter.
fn short(at: usize, len: usize) -> bool {
    len <= SHORT && at % 4 == 0 && len % 4 == 0
}

/// Whether a run of `len` bytes at offset `at` of a copy in the card's memory
/// is no copy command of its own, but pairs of one dispatch.
pub(crate) fn goes_as_words(at: usize, len: usize) -> bool {
    own() && short(at, len)
}

/// Whether the title's bytes that go in under flights' writes
/// (`Shadow::merge_under_writes`) go in by whole words
/// (`VICTORY_UNDER_WORDS=0`: by the bytes that changed, as before).
///
/// By bytes nearly every run (a float's three low bytes) was a copy command
/// of its own, and a submission with more than 4,096 of them stood until the
/// flights had landed: 4 to 14 ms of a steady frame (session play-1125-0203:
/// "more changed runs than a submission sends up"), for some 20 KB. Whole
/// words are pairs of one dispatch, and are not counted.
///
/// A word is taken to be the title's or the shaders', not shared: where the
/// title changed a byte of a word of which a flight's shader writes another
/// byte, that byte is as it was before the flight. The landing rests on the
/// same ([`crate::shadow::exact_back`]).
pub(crate) fn words_under() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| own() && std::env::var_os("VICTORY_UNDER_WORDS").is_none_or(|v| v != "0"))
}

const PAIR_BYTES: u64 = 8;
const QUAD_BYTES: u64 = 20;

fn packed() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SEND_PACKED").is_none_or(|v| v != "0"))
}

const SCATTER: &str = r#"
cbuffer Part : register(b0) { uint pairs; uint pair_count; uint quads; uint first; uint count; };
ByteAddressBuffer packets : register(t0);
RWByteAddressBuffer card : register(u0);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= count) return;
    uint i = first + id.x;
    if (i < pair_count) {
        uint2 p = packets.Load2(pairs + i * 8u);
        card.Store(p.x, p.y);
    } else {
        uint at = quads + (i - pair_count) * 20u;
        card.Store4(packets.Load(at), packets.Load4(at + 4u));
    }
}
"#;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

fn pass(host: &Host) -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let code = compile(SCATTER, windows::core::s!("main"), windows::core::s!("cs_5_1")).ok()?;
        let root = raw_compute_root(host, 5)?;
        // SAFETY: the code and root signature live through creation.
        unsafe {
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .ok()?;
            name(&pso, || "victory: changed words sent to the card".into());
            Some(Pass { root, pso })
        }
    })
    .as_ref()
}

/// Memory a submission's sends come from, mapped.
pub(super) struct Memory {
    buffer: ID3D12Resource,
    data: usize,
    size: u64,
}

// SAFETY: a D3D12 object (free-threaded) and its mapping, with one owner.
unsafe impl Send for Memory {}

/// What submissions that have run gave back.
static SPARE: Mutex<Vec<(u64, Memory)>> = Mutex::new(Vec::new());

/// At most this much is kept (of the card's memory, with [`on_card`]).
fn kept() -> u64 {
    if on_card() { 256 << 20 } else { 768 << 20 }
}

/// Whether the memory lies in the card's memory, written by the CPU through
/// the bus (`VICTORY_SEND_BAR=1`): the copies on the card's drawing queue, which
/// nothing overlaps, then cross no bus. In a captured frame they took 5.4 ms of
/// the card's 30 (342 copies), and take 0.4 this way; but filling it takes the
/// submitting thread 8-9 ms a frame instead of 3 (73-93 MB, run card-3 against
/// card-1), so it is a loss while the card waits for that thread: off by
/// default.
fn on_card() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SEND_BAR").is_some_and(|v| v == "1"))
}

fn memory(host: &Host, size: u64) -> Option<Memory> {
    {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        let fit = spare.iter().enumerate().filter(|(_, (_, m))| m.size >= size).min_by_key(|(_, (since, m))| (m.size, std::cmp::Reverse(*since))).map(|(i, _)| i);
        if let Some(i) = fit {
            return Some(spare.swap_remove(i).1);
        }
    }
    // By doubles, so that the next of about this size finds it: what
    // submissions send differs from frame to frame.
    let size = size.max(8 << 20).next_power_of_two();
    let mapped = if on_card() { host.mapped_card_buffer(size, false) } else { None };
    let (buffer, data) = match mapped {
        Some(made) => made,
        None => host.upload_buffer(size)?,
    };
    name(&buffer, || "bytes a submission sends up".into());
    Some(Memory {
        buffer,
        data: data as usize,
        size,
    })
}

/// The submission has run.
pub(super) fn give_back(memory: Memory) {
    let evicted = {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        spare.push((now_ms(), memory));
        // Keep the smaller buffers when over budget, and evict the oldest
        // first among equally sized buffers.
        spare.sort_unstable_by_key(|(since, m)| (m.size, std::cmp::Reverse(*since)));
        let mut bytes = 0u64;
        let n = spare.iter().take_while(|(_, m)| {
            bytes += m.size;
            bytes <= kept()
        }).count();
        spare.split_off(n)
    };
    drop(evicted);
}

/// Releases unused send buffers after the RAM pressure-adjusted idle period.
pub(super) fn trim_idle(now: u64, idle: u64) {
    let expired = {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        spare.extract_if(.., |(since, _)| now.saturating_sub(*since) >= idle).collect::<Vec<_>>()
    };
    drop(expired);
}

/// Bytes copied, runs copied, words stored, CPU preparation time in
/// microseconds, and the actual scatter packet bytes.
pub(super) static STATS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

#[derive(Clone, Copy, Default)]
struct Counts { pairs: u32, quads: u32 }

impl Counts {
    fn add(&mut self, bytes: usize, packed: bool) {
        let words = (bytes / 4) as u32;
        let quads = if packed { words / 4 } else { 0 };
        self.quads += quads;
        self.pairs += words - quads * 4;
    }

    fn packets(self) -> u32 {
        self.pairs + self.quads
    }

    fn bytes(self) -> u64 {
        u64::from(self.pairs) * PAIR_BYTES + u64::from(self.quads) * QUAD_BYTES
    }
}

#[derive(Clone, Copy)]
struct Scatter {
    pairs: u64,
    quads: u64,
    counts: Counts,
}

fn short_counts(ranges: &[(usize, usize)], packed: bool) -> Counts {
    let mut counts = Counts::default();
    for &(a, b) in ranges {
        if short(a, b - a) {
            counts.add(b - a, packed);
        }
    }
    counts
}

/// # Safety
/// Each source run is valid. The pair and quad regions are writable,
/// aligned to four bytes, disjoint and sized using `short_counts`.
unsafe fn pack_scatter(
    src: *const u8, ranges: &[(usize, usize)], packed: bool,
    mut pairs: *mut [u32; 2], mut quads: *mut [u32; 5],
) {
    for &(a, b) in ranges.iter().filter(|&&(a, b)| short(a, b - a)) {
        let mut offset = a;
        if packed {
            while b - offset >= 16 {
                // SAFETY: this packet stores four consecutive changed
                // words; none of the neighboring bytes are included.
                unsafe {
                    let [a, b, c, d] = src.add(offset).cast::<[u32; 4]>().read_unaligned();
                    quads.write([offset as u32, a, b, c, d]);
                    quads = quads.add(1);
                }
                offset += 16;
            }
        }
        while offset < b {
            // SAFETY: one changed word and an output pair assigned to this worker.
            unsafe {
                pairs.write([offset as u32, src.add(offset).cast::<u32>().read_unaligned()]);
                pairs = pairs.add(1);
            }
            offset += 4;
        }
    }
}

/// One copy's sends: ranges copied (offset in the copy, in the memory,
/// bytes), and the packed scatter packets.
struct Copy {
    runs: Vec<(usize, u64, usize)>,
    scatter: Scatter,
}

/// Where a submission's bytes are taken to.
pub(super) struct Own {
    memory: Memory,
    copies: Vec<Copy>,
}

/// Runs whose pairs are one piece of work.
const RUNS_AT_A_TIME: usize = 4096;

/// Takes what `dirty` sends up out of the CPU sides.
pub(super) fn take(host: &Host, dirty: &[(std::sync::Arc<Shadow>, Vec<(usize, usize)>)]) -> Option<Own> {
    let began = std::time::Instant::now();
    pass(host)?;
    // A byte beside a short run's whole words may belong to the GPU.
    // Pack those runs without widening any destination range.
    let packed = packed();
    let mut at = 0u64;
    let mut copies = Vec::with_capacity(dirty.len());
    for (_, ranges) in dirty {
        let mut runs = Vec::new();
        let mut counts = Counts::default();
        for &(a, b) in ranges {
            if short(a, b - a) {
                // HLSL uses 32-bit byte addresses. The regular buffer-copy
                // fallback still supports the full range.
                if b as u64 > (1u64 << 32) {
                    return None;
                }
                counts.add(b - a, packed);
            } else {
                runs.push((a, at, b - a));
                at += ((b - a) as u64).next_multiple_of(16);
            }
        }
        let pairs = at;
        let quads = pairs + u64::from(counts.pairs) * PAIR_BYTES;
        at = (quads + u64::from(counts.quads) * QUAD_BYTES).next_multiple_of(16);
        copies.push(Copy { runs, scatter: Scatter { pairs, quads, counts } });
    }
    if at == 0 || at > (1u64 << 32) {
        return None;
    }
    let memory = memory(host, at)?;
    let mut moves: Vec<(usize, usize, usize)> = Vec::new();
    let mut work = Vec::new();
    for ((shadow, ranges), copy) in dirty.iter().zip(&copies) {
        let from = shadow.card_data() as usize;
        moves.extend(copy.runs.iter().map(|&(a, to, n)| (from + a, memory.data + to as usize, n)));
        let mut pairs = memory.data + copy.scatter.pairs as usize;
        let mut quads = memory.data + copy.scatter.quads as usize;
        for ranges in ranges.chunks(RUNS_AT_A_TIME) {
            let counts = short_counts(ranges, packed);
            if counts.packets() > 0 {
                work.push((from, ranges, pairs, quads));
                pairs += counts.pairs as usize * PAIR_BYTES as usize;
                quads += counts.quads as usize * QUAD_BYTES as usize;
            }
        }
    }
    crate::land::each_under(work.len(), &|i| {
        let (from, ranges, pairs, quads) = work[i];
        // SAFETY: each job owns its packet region until this call returns;
        // the live CPU-side copies are stable through packing.
        unsafe { pack_scatter(from as *const u8, ranges, packed, pairs as *mut _, quads as *mut _) };
    });
    STATS[0].fetch_add(moves.iter().map(|m| m.2 as u64).sum(), Ordering::Relaxed);
    STATS[1].fetch_add(moves.len() as u64, Ordering::Relaxed);
    STATS[2].fetch_add(copies.iter().map(|c| u64::from(c.scatter.counts.pairs) + 4 * u64::from(c.scatter.counts.quads)).sum(), Ordering::Relaxed);
    STATS[4].fetch_add(copies.iter().map(|c| c.scatter.counts.bytes()).sum(), Ordering::Relaxed);
    crate::shadow::copy_all(moves);
    STATS[3].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
    Some(Own { memory, copies })
}

impl Own {
    /// Records the sends into `list`; the copies that `looked` says are in
    /// `COPY_DEST`, the others in the common state. Those with a "before" are
    /// left readable by compute shaders.
    ///
    /// # Safety
    /// An open list of ours, whose compute state is set anew behind this.
    pub(super) unsafe fn record(
        &self,
        host: &Host,
        list: &ID3D12GraphicsCommandList,
        dirty: &[(std::sync::Arc<Shadow>, Vec<(usize, usize)>)],
        looked: impl Fn(&Shadow) -> bool,
    ) {
        let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        let write = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let dest = D3D12_RESOURCE_STATE_COPY_DEST;
        let (mut first, mut last) = (Vec::new(), Vec::new());
        for ((shadow, _), copy) in dirty.iter().zip(&self.copies) {
            for &(a, from, n) in &copy.runs {
                // SAFETY: the caller's; live buffers, ranges inside them.
                unsafe { list.CopyBufferRegion(&shadow.buffer, a as u64, &self.memory.buffer, from, n as u64) };
            }
            let mut state = (looked(shadow) || !copy.runs.is_empty()).then_some(dest);
            if copy.scatter.counts.packets() > 0 {
                if let Some(was) = state {
                    first.push(transition(&shadow.buffer, was, write));
                }
                state = Some(write);
            }
            let hashed = shadow.card_sides().is_some_and(|(_, before, _)| before.is_some());
            if let Some(was) = state
                && hashed
                && (!copy.runs.is_empty() || copy.scatter.counts.packets() > 0)
            {
                last.push(transition(&shadow.buffer, was, read));
            }
        }
        // SAFETY: the caller's; live objects.
        unsafe {
            if !first.is_empty() {
                list.ResourceBarrier(&first);
            }
            if let Some(pass) = pass(host)
                && self.copies.iter().any(|c| c.scatter.counts.packets() > 0)
            {
                list.SetComputeRootSignature(&pass.root);
                list.SetPipelineState(&pass.pso);
                list.SetComputeRootShaderResourceView(1, self.memory.buffer.GetGPUVirtualAddress());
                for ((shadow, _), copy) in dirty.iter().zip(&self.copies).filter(|(_, c)| c.scatter.counts.packets() > 0) {
                    list.SetComputeRootUnorderedAccessView(2, shadow.buffer.GetGPUVirtualAddress());
                    const MOST: u32 = 64 * D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;
                    let scatter = copy.scatter;
                    let packets = scatter.counts.packets();
                    let mut done = 0;
                    while done < packets {
                        let n = (packets - done).min(MOST);
                        let constants = [scatter.pairs as u32, scatter.counts.pairs, scatter.quads as u32, done, n];
                        list.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
                        list.Dispatch(n.div_ceil(64), 1, 1);
                        done += n;
                    }
                }
            }
            if !last.is_empty() {
                list.ResourceBarrier(&last);
            }
        }
    }

    /// The memory, to be kept until the submission has run.
    pub(super) fn memory(self) -> Memory {
        self.memory
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_packing_reduces_bytes_without_touching_unchanged_words() {
        let input: Vec<u32> = (0..96).map(|i| i ^ 0xabba).collect();
        // Disjoint runs: a byte copy, a quad with a tail, a gap, another
        // quad, a singleton, and a large run that stays a direct copy.
        let ranges = [(1, 3), (4, 28), (32, 48), (60, 64), (80, 356)];
        for packed in [false, true] {
            let counts = short_counts(&ranges, packed);
            assert_eq!(counts.bytes(), if packed { 64 } else { 88 });
            let mut pairs = vec![[0u32; 2]; counts.pairs as usize];
            let mut quads = vec![[0u32; 5]; counts.quads as usize];
            // SAFETY: valid source and packet arrays of exactly the
            // computed size, including the zero-quad fallback.
            unsafe {
                pack_scatter(input.as_ptr().cast(), &ranges, packed, pairs.as_mut_ptr(), quads.as_mut_ptr());
            }
            let mut actual = vec![0; input.len()];
            for [offset, word] in pairs {
                actual[offset as usize / 4] = word;
            }
            for [offset, a, b, c, d] in quads {
                let at = offset as usize / 4;
                actual[at..at + 4].copy_from_slice(&[a, b, c, d]);
            }
            let mut expected = vec![0; input.len()];
            for (from, to) in [(1, 7), (8, 12), (15, 16)] {
                expected[from..to].copy_from_slice(&input[from..to]);
            }
            assert_eq!(actual, expected);
        }
    }
}
