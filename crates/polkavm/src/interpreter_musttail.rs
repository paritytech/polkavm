use super::{HandlerResult, InterpretedInstance, Target};
use polkavm_common::cast::cast;

#[inline(always)]
pub(super) fn handler_tail<const DEBUG: bool>(visitor: &mut InterpretedInstance, next_off: Target) -> HandlerResult {
    become dispatch::<DEBUG>(visitor, next_off);
}

#[inline(always)]
pub(super) fn dispatch<const DEBUG: bool>(visitor: &mut InterpretedInstance, off: Target) -> HandlerResult {
    if DEBUG {
        visitor.cycle_counter += 1;
    }
    if let Some(&handler) = visitor.compiled_handlers.get(cast(off).to_usize()) {
        become handler(visitor, off);
    } else {
        visitor.interrupt.clone()
    }
}
