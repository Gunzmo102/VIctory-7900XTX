//! GPU detiling for first uploads of fully mapped 2D textures. The GPU
//! copies elements as bytes, so native BC blocks stay compressed for sampling.

use std::sync::OnceLock;

use victory_formats::texture::{Dimension, Layout, Swizzle};
use windows::Win32::Graphics::Direct3D12::*;

use super::Staging;
use crate::host::{Host, compile, raw_compute_root, transition};

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_DETILE").is_some_and(|v| v == "1"))
}

const SHADER: &str = r#"
cbuffer Params : register(b0) {
    uint width, height, words, columns, rows, dst, pitch, unused;
};
ByteAddressBuffer tiled : register(t0);
RWByteAddressBuffer output_bytes : register(u0);
[numthreads(16, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= width || id.y >= height) return;
    uint from = tiled.Load(columns + 4u * id.x) + tiled.Load(rows + 4u * id.y);
    uint to = dst + id.y * pitch + id.x * words * 4u;
    for (uint k = 0; k < words; ++k)
        output_bytes.Store(to + k * 4u, tiled.Load(from + k * 4u));
}
"#;

pub(super) struct Pass {
    root: ID3D12RootSignature,
    pipeline: ID3D12PipelineState,
}

pub(super) fn pass(host: &Host) -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let code = compile(
            SHADER,
            windows::core::s!("main"),
            windows::core::s!("cs_5_1"),
        )
        .ok()?;
        let root = raw_compute_root(host, 8)?;
        // SAFETY: the shader bytecode and root signature outlive creation.
        unsafe {
            let pipeline = host
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
            Some(Pass { root, pipeline })
        }
    })
    .as_ref()
}

/// Bound each worker's additional upload and device memory, and leave
/// small copies with the CPU where a dispatch is unlikely to pay off.
const MIN_BYTES: usize = 256 << 10;
const MAX_BYTES: u64 = 8 << 20;

pub(super) struct Scratch {
    input: ID3D12Resource,
    mapped: usize,
    input_size: usize,
    pub(super) output: ID3D12Resource,
    output_size: u64,
    parts: Vec<[u32; 8]>,
}

// SAFETY: a worker owns its scratch, and waits for its fence before reuse.
unsafe impl Send for Scratch {}

impl Scratch {
    pub(super) fn prepare(slot: &mut Option<Self>, staging: &Staging) -> Option<()> {
        let layout = &staging.layout;
        if layout.desc.dimension != Dimension::Tex2d
            || layout.slices != 1
            || layout.desc.swizzle == Swizzle::LINEAR
            || !layout.addressable()
            || !matches!(layout.desc.element_bytes, 4 | 8 | 16)
            || staging.read < MIN_BYTES
            || staging.bytes > MAX_BYTES
            || !staging.read.is_multiple_of(4)
        {
            return None;
        }

        // The CPU staging cache must not already hold a different upload.
        // An offloaded upload leaves its CPU hash unset.
        let cached = staging.last.try_lock().ok()?;
        if cached.hash.is_some() {
            return None;
        }
        let mut total = staging.read;
        for i in 0..staging.footprints.len() {
            let mip = layout.mips.get(staging.part(i).0 as usize)?;
            total = total.checked_add(4 * (mip.width as usize + mip.height as usize))?;
        }
        if total as u64 > MAX_BYTES {
            return None;
        }

        let host = crate::host::host()?;
        if slot
            .as_ref()
            .is_none_or(|s| s.input_size < total || s.output_size < staging.bytes)
        {
            let input_size = total.next_power_of_two();
            let output_size = staging.bytes.next_power_of_two();
            let (input, mapped) = host.upload_buffer(input_size as u64)?;
            *slot = Some(Self {
                input,
                mapped: mapped as usize,
                input_size,
                output: host.uav_buffer(output_size)?,
                output_size,
                parts: Vec::new(),
            });
        }
        let scratch = slot.as_mut()?;
        // Do not block on a remap, or publish an incomplete texture through
        // this path. The CPU refresh can handle holes and retry rules.
        let _mapped = crate::watch::title_memory_reading_now()?;
        // SAFETY: scratch is mapped upload memory and its last fence passed.
        let input = unsafe { std::slice::from_raw_parts_mut(scratch.mapped as *mut u8, total) };
        if !crate::host::read_into(staging.memory, &mut input[..staging.read]) {
            return None;
        }
        drop(_mapped);

        // For this layout offset(x,y) is the sum of independent column
        // and row terms, including mip tails. Compute O(width + height)
        // offsets on the CPU, then let the GPU move all elements.
        scratch.parts.clear();
        let mut end = staging.read;
        for (i, footprint) in staging.footprints.iter().enumerate() {
            let mip = staging.part(i).0;
            let m = &layout.mips[mip as usize];
            let (columns, rows) = append_offsets(layout, mip, input, &mut end)?;
            scratch.parts.push([
                m.width,
                m.height,
                layout.desc.element_bytes / 4,
                columns as u32,
                rows as u32,
                u32::try_from(footprint.Offset).ok()?,
                footprint.Footprint.RowPitch,
                0,
            ]);
        }
        Some(())
    }

    /// Records the decode into a buffer with normal D3D12 copy footprints.
    /// The caller copies it into the texture, then returns it to COMMON.
    ///
    /// # Safety
    /// The direct list is recording; scratch and pass live until completion.
    pub(super) unsafe fn record(&self, list: &ID3D12GraphicsCommandList, pass: &Pass) {
        // SAFETY: live buffers in COMMON/UPLOAD and disjoint output footprints.
        unsafe {
            list.ResourceBarrier(&[transition(
                &self.output,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            list.SetComputeRootSignature(&pass.root);
            list.SetPipelineState(&pass.pipeline);
            list.SetComputeRootShaderResourceView(1, self.input.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, self.output.GetGPUVirtualAddress());
            for params in &self.parts {
                list.SetComputeRoot32BitConstants(
                    0,
                    params.len() as u32,
                    params.as_ptr().cast(),
                    0,
                );
                list.Dispatch(params[0].div_ceil(16), params[1].div_ceil(8), 1);
            }
            list.ResourceBarrier(&[transition(
                &self.output,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
        }
    }
}

/// The address of (x, y) is the sum of its row and column terms, including
/// mip tails. Only the row term includes the mip's origin.
fn append_offsets(
    layout: &Layout,
    mip: u32,
    input: &mut [u8],
    end: &mut usize,
) -> Option<(usize, usize)> {
    let m = layout.mips.get(mip as usize)?;
    let origin = layout.offset(mip, 0, 0, 0).ok()?;
    let columns = *end;
    let rows = columns.checked_add(4 * m.width as usize)?;
    let mut put = |offset: u64| -> Option<()> {
        let offset = u32::try_from(offset).ok()?;
        input
            .get_mut(*end..*end + 4)?
            .copy_from_slice(&offset.to_le_bytes());
        *end += 4;
        Some(())
    };
    for x in 0..m.width {
        put(layout.offset(mip, x, 0, 0).ok()?.checked_sub(origin)?)?;
    }
    for y in 0..m.height {
        put(layout.offset(mip, 0, y, 0).ok()?)?;
    }
    Some((columns, rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::Desc;

    #[test]
    fn shader_address_tables_match_cpu_detiling_including_mip_tails() {
        for swizzle in [Swizzle::SW_256B_S, Swizzle::SW_4KB_D, Swizzle::SW_64KB_S] {
            for (bytes, block) in [(4, 1), (8, 4), (16, 4)] {
                let layout = Layout::new(&Desc {
                    dimension: Dimension::Tex2d,
                    width: 1031,
                    height: 449,
                    depth: 1,
                    mips: 9,
                    element_bytes: bytes,
                    block,
                    samples: 1,
                    swizzle,
                })
                .unwrap();
                let tiled = (0..layout.size)
                    .map(|at| (at ^ (at >> 7) ^ (at >> 15)) as u8)
                    .collect::<Vec<_>>();
                for (i, mip) in layout.mips.iter().enumerate() {
                    let size = bytes as usize;
                    let pitch = (mip.width as usize * size).next_multiple_of(256);
                    let len = (mip.height as usize - 1) * pitch + mip.width as usize * size;
                    let mut cpu = vec![0; len];
                    layout
                        .detile(&tiled, i as u32, 0, &mut cpu, pitch, len)
                        .unwrap();
                    let mut input = vec![0; 4 * (mip.width + mip.height) as usize + 4];
                    let mut end = 4;
                    let (columns, rows) =
                        append_offsets(&layout, i as u32, &mut input, &mut end).unwrap();
                    assert_eq!(end, input.len());
                    let get = |table, i: u32| {
                        let at = table + i as usize * 4;
                        u32::from_le_bytes(input[at..at + 4].try_into().unwrap()) as usize
                    };
                    for y in 0..mip.height {
                        for x in 0..mip.width {
                            let from = get(columns, x) + get(rows, y);
                            let to = y as usize * pitch + x as usize * size;
                            assert_eq!(from % 4, 0);
                            assert_eq!(&tiled[from..from + size], &cpu[to..to + size]);
                        }
                    }
                }
            }
        }
    }
}
