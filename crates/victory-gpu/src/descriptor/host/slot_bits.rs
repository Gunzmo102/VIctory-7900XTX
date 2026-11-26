//! Occupancy of the dynamic descriptor region. The caller holds its lock.

pub(super) struct Taken {
    words: Vec<u64>,
    len: usize,
}

impl Taken {
    pub(super) fn new(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(64)],
            len,
        }
    }

    #[cfg(any(test, feature = "diagnostics"))]
    pub(super) fn get(&self, at: usize) -> Option<bool> {
        (at < self.len).then(|| self.words[at / 64] & (1 << (at % 64)) != 0)
    }

    /// Search from the cursor to the end, then from the start. The second
    /// search can cross the cursor, but a run cannot wrap around the heap.
    pub(super) fn find(&self, cursor: usize, needed: usize) -> Option<usize> {
        debug_assert!(cursor <= self.len && needed > 0);
        let second = cursor.saturating_add(needed - 1).min(self.len);
        self.find_in(cursor, self.len, needed)
            .or_else(|| self.find_in(0, second, needed))
    }

    fn find_in(&self, from: usize, to: usize, needed: usize) -> Option<usize> {
        if to - from < needed {
            return None;
        }
        let mut at = from;
        let mut carry = 0;
        while at < to {
            let offset = at % 64;
            let len = (64 - offset).min(to - at);
            let mask = u64::MAX >> (64 - len);
            let free = (!self.words[at / 64] >> offset) & mask;
            if free == 0 {
                carry = 0;
            } else if free == mask {
                if carry + len >= needed {
                    return Some(at - carry);
                }
                carry += len;
            } else {
                // A run crossing the previous word comes before any
                // run wholly inside this one.
                if carry + free.trailing_ones() as usize >= needed {
                    return Some(at - carry);
                }
                if needed <= len {
                    let starts = run_starts(free, needed);
                    if starts != 0 {
                        return Some(at + starts.trailing_zeros() as usize);
                    }
                }
                carry = (free << (64 - len)).leading_ones() as usize;
            }
            at += len;
        }
        None
    }

    /// Set bits in a run the allocator just found.
    pub(super) fn reserve(&mut self, from: usize, to: usize) {
        debug_assert!(to <= self.len);
        self.edit(from, to, |word, mask| {
            debug_assert_eq!(*word & mask, 0);
            *word |= mask;
            0
        });
    }

    /// Releases only occupied slots in the range, returning how many.
    pub(super) fn release(&mut self, from: usize, to: usize) -> u32 {
        self.edit(from, to.min(self.len), |word, mask| {
            let released = (*word & mask).count_ones();
            *word &= !mask;
            released
        })
    }

    fn edit(&mut self, mut from: usize, to: usize, mut f: impl FnMut(&mut u64, u64) -> u32) -> u32 {
        let mut changed = 0;
        while from < to {
            let offset = from % 64;
            let len = (64 - offset).min(to - from);
            let mask = (u64::MAX >> (64 - len)) << offset;
            changed += f(&mut self.words[from / 64], mask);
            from += len;
        }
        changed
    }
}

/// For each possible start bit, intersect all the free bits its run needs.
/// Doubling the covered length makes heavily fragmented words cheap too.
fn run_starts(free: u64, needed: usize) -> u64 {
    let mut starts = free;
    let mut covered = 1;
    while covered * 2 <= needed && starts != 0 {
        starts &= starts >> covered;
        covered *= 2;
    }
    if starts != 0 && needed > covered {
        starts &= starts >> (needed - covered);
    }
    starts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear(taken: &[bool], cursor: usize, needed: usize) -> Option<usize> {
        let end = taken.len();
        (cursor..end)
            .chain(0..cursor)
            .find(|&at| at + needed <= end && taken[at..at + needed].iter().all(|&t| !t))
    }

    #[test]
    fn bitmap_matches_the_slot_by_slot_search() {
        for mask in 0..256u64 {
            let mut slots = Taken::new(8);
            let model = std::array::from_fn::<_, 8, _>(|i| mask & (1 << i) != 0);
            for (i, occupied) in model.into_iter().enumerate() {
                if occupied {
                    slots.reserve(i, i + 1);
                }
            }
            for cursor in 0..8 {
                for needed in 1..=8 {
                    assert_eq!(slots.find(cursor, needed), linear(&model, cursor, needed));
                }
            }
        }
    }

    #[test]
    fn multiword_search_matches_fragmented_and_sparse_heaps() {
        let mut seed = 7u64;
        for period in [2, 8, 32, 128] {
            for _ in 0..24 {
                let model: Vec<bool> = (0..195)
                    .map(|i| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        (seed >> 32) as usize % period == 0 || i % (period * 2) == 0
                    })
                    .collect();
                let mut slots = Taken::new(model.len());
                for (i, &taken) in model.iter().enumerate() {
                    if taken {
                        slots.reserve(i, i + 1);
                    }
                }
                for cursor in [0, 1, 62, 63, 64, 65, 127, 190] {
                    for needed in [1, 2, 3, 7, 15, 17, 31, 63, 64, 65, 128, 195] {
                        assert_eq!(slots.find(cursor, needed), linear(&model, cursor, needed));
                    }
                }
            }
        }
    }

    #[test]
    fn reserving_and_releasing_across_words_preserves_neighbors() {
        let mut slots = Taken::new(195);
        slots.reserve(0, 189);
        slots.reserve(192, 195);
        assert_eq!(slots.find(30, 3), Some(189));
        assert_eq!(slots.find(190, 3), Some(189));
        assert_eq!(slots.find(0, 4), None);
        assert_eq!(slots.release(60, 134), 74);
        assert_eq!(slots.release(60, 134), 0);
        assert_eq!(slots.find(62, 74), Some(60));
        assert_eq!(slots.find(62, 75), None);
        slots.reserve(60, 134);
        assert_eq!(slots.release(193, 900), 2);
        assert_eq!(slots.get(192), Some(true));
        assert_eq!(slots.get(194), Some(false));
        assert_eq!(slots.get(195), None);
    }
}
