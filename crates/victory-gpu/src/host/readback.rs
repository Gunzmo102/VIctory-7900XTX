//! Plan buffer readbacks independently of their CPU landing order.

pub(super) struct Copy {
    /// Index of any request for this source buffer.
    pub(super) request: usize,
    pub(super) from: usize,
    pub(super) to: usize,
    pub(super) offset: u64,
}

pub(super) struct Fetch {
    pub(super) copies: Vec<Copy>,
    /// Offset of each original request into the readback buffer.
    pub(super) offsets: Vec<u64>,
    pub(super) bytes: u64,
}

/// Combines overlapping or adjacent requests for a single GPU buffer.
/// Requests are (unique buffer ID, start offset, exclusive end offset).
/// Landing still uses the original order, including when separate GPU
/// buffers alias the same title memory. Do not read across gaps.
pub(super) fn fetch(ranges: impl Iterator<Item = (u64, usize, usize)>) -> Fetch {
    let mut ranges: Vec<_> = ranges
        .enumerate()
        .map(|(i, (id, from, to))| (id, from, to, i))
        .collect();
    ranges.sort_unstable();
    let mut copies: Vec<Copy> = Vec::new();
    let mut offsets = vec![0; ranges.len()];
    let mut bytes = 0;
    let mut previous = None;
    for (id, from, to, request) in ranges {
        debug_assert!(from < to);
        if previous == Some(id)
            && let Some(copy) = copies.last_mut()
            && from <= copy.to
        {
            offsets[request] = copy.offset + (from - copy.from) as u64;
            if to > copy.to {
                bytes += (to - copy.to) as u64;
                copy.to = to;
            }
            // Prefer the first request in the original sequence.
            copy.request = copy.request.min(request);
        } else {
            offsets[request] = bytes;
            copies.push(Copy {
                request,
                from,
                to,
                offset: bytes,
            });
            bytes += (to - from) as u64;
        }
        previous = Some(id);
    }
    copies.sort_unstable_by_key(|copy| copy.request);
    Fetch {
        copies,
        offsets,
        bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_and_adjacent_requests_share_transferred_bytes() {
        let requests = [
            (1, 8, 20),
            (2, 0, 12),
            (1, 0, 10),
            (1, 20, 24),
            (1, 27, 32),
            (1, 4, 8),
        ];
        let plan = fetch(requests.into_iter());
        assert_eq!(plan.copies.len(), 3);
        assert_eq!(plan.bytes, 24 + 5 + 12);
        let mut downloaded = vec![0; plan.bytes as usize];
        for copy in plan.copies {
            let (id, _, _) = requests[copy.request];
            for (i, src) in (copy.from..copy.to).enumerate() {
                downloaded[copy.offset as usize + i] = (id as usize * 100 + src) as u8;
            }
        }
        // All requested bytes still land in order and come from their
        // original GPU buffers, including the different copy at offset zero.
        for ((id, from, to), offset) in requests.into_iter().zip(plan.offsets) {
            let expected: Vec<_> = (from..to).map(|i| (id as usize * 100 + i) as u8).collect();
            assert_eq!(&downloaded[offset as usize..][..to - from], expected);
        }
        let empty = fetch(std::iter::empty());
        assert!(empty.copies.is_empty() && empty.offsets.is_empty());
        assert_eq!(empty.bytes, 0);
    }
}
