use alloc::sync::Arc;
use polkavm_common::program::{InstructionSet, InstructionVisitor, Instructions, RawReg};

#[derive(Clone)]
pub struct CostModelRef {
    pointer: *const CostModel,
    _lifetime: Option<Arc<CostModel>>,
}

// SAFETY: The pointer inside of the struct points to either a value with a static lifetime, or the `Arc` that we keep in the struct.
unsafe impl Send for CostModelRef {}

// SAFETY: The pointer inside of the struct points to either a value with a static lifetime, or the `Arc` that we keep in the struct.
unsafe impl Sync for CostModelRef {}

impl CostModelRef {
    pub const fn from_static(cost_model: &'static CostModel) -> Self {
        CostModelRef {
            pointer: cost_model as *const CostModel,
            _lifetime: None,
        }
    }
}

impl From<&'static CostModel> for CostModelRef {
    fn from(value: &'static CostModel) -> Self {
        Self::from_static(value)
    }
}

impl From<Arc<CostModel>> for CostModelRef {
    fn from(value: Arc<CostModel>) -> Self {
        CostModelRef {
            pointer: Arc::as_ptr(&value),
            _lifetime: Some(value),
        }
    }
}

impl core::ops::Deref for CostModelRef {
    type Target = CostModel;

    fn deref(&self) -> &Self::Target {
        // SAFETY: The pointer points to either a value with a static lifetime, or the `Arc` that we keep in the struct.
        unsafe { &*self.pointer }
    }
}

pub type Cost = u32;

#[allow(clippy::exhaustive_structs)]
#[derive(Hash)]
pub struct CostModel {
    pub add_32: Cost,
    pub add_64: Cost,
    pub add_imm_32: Cost,
    pub add_imm_64: Cost,
    pub and: Cost,
    pub and_imm: Cost,
    pub and_inverted: Cost,
    pub branch_eq: Cost,
    pub branch_eq_imm: Cost,
    pub branch_greater_or_equal_signed: Cost,
    pub branch_greater_or_equal_signed_imm: Cost,
    pub branch_greater_or_equal_unsigned: Cost,
    pub branch_greater_or_equal_unsigned_imm: Cost,
    pub branch_greater_signed_imm: Cost,
    pub branch_greater_unsigned_imm: Cost,
    pub branch_less_or_equal_signed_imm: Cost,
    pub branch_less_or_equal_unsigned_imm: Cost,
    pub branch_less_signed: Cost,
    pub branch_less_signed_imm: Cost,
    pub branch_less_unsigned: Cost,
    pub branch_less_unsigned_imm: Cost,
    pub branch_not_eq: Cost,
    pub branch_not_eq_imm: Cost,
    pub cmov_if_not_zero: Cost,
    pub cmov_if_not_zero_imm: Cost,
    pub cmov_if_zero: Cost,
    pub cmov_if_zero_imm: Cost,
    pub count_leading_zero_bits_32: Cost,
    pub count_leading_zero_bits_64: Cost,
    pub count_set_bits_32: Cost,
    pub count_set_bits_64: Cost,
    pub count_trailing_zero_bits_32: Cost,
    pub count_trailing_zero_bits_64: Cost,
    pub div_signed_32: Cost,
    pub div_signed_64: Cost,
    pub div_unsigned_32: Cost,
    pub div_unsigned_64: Cost,
    pub ecalli: Cost,
    pub fallthrough: Cost,
    pub invalid: Cost,
    pub jump: Cost,
    pub jump_indirect: Cost,
    pub load_i16: Cost,
    pub load_i32: Cost,
    pub load_i8: Cost,
    pub load_imm: Cost,
    pub load_imm64: Cost,
    pub load_imm_and_jump: Cost,
    pub load_imm_and_jump_indirect: Cost,
    pub load_indirect_i16: Cost,
    pub load_indirect_i32: Cost,
    pub load_indirect_i8: Cost,
    pub load_indirect_u16: Cost,
    pub load_indirect_u32: Cost,
    pub load_indirect_u64: Cost,
    pub load_indirect_u8: Cost,
    pub load_u16: Cost,
    pub load_u32: Cost,
    pub load_u64: Cost,
    pub load_u8: Cost,
    pub maximum: Cost,
    pub maximum_unsigned: Cost,
    pub memset: Cost,
    pub minimum: Cost,
    pub minimum_unsigned: Cost,
    pub move_reg: Cost,
    pub mul_32: Cost,
    pub mul_64: Cost,
    pub mul_imm_32: Cost,
    pub mul_imm_64: Cost,
    pub mul_upper_signed_signed: Cost,
    pub mul_upper_signed_unsigned: Cost,
    pub mul_upper_unsigned_unsigned: Cost,
    pub negate_and_add_imm_32: Cost,
    pub negate_and_add_imm_64: Cost,
    pub or: Cost,
    pub or_imm: Cost,
    pub or_inverted: Cost,
    pub rem_signed_32: Cost,
    pub rem_signed_64: Cost,
    pub rem_unsigned_32: Cost,
    pub rem_unsigned_64: Cost,
    pub reverse_byte: Cost,
    pub rotate_left_32: Cost,
    pub rotate_left_64: Cost,
    pub rotate_right_32: Cost,
    pub rotate_right_64: Cost,
    pub rotate_right_imm_32: Cost,
    pub rotate_right_imm_64: Cost,
    pub rotate_right_imm_alt_32: Cost,
    pub rotate_right_imm_alt_64: Cost,
    pub sbrk: Cost,
    pub set_greater_than_signed_imm: Cost,
    pub set_greater_than_unsigned_imm: Cost,
    pub set_less_than_signed: Cost,
    pub set_less_than_signed_imm: Cost,
    pub set_less_than_unsigned: Cost,
    pub set_less_than_unsigned_imm: Cost,
    pub shift_arithmetic_right_32: Cost,
    pub shift_arithmetic_right_64: Cost,
    pub shift_arithmetic_right_imm_32: Cost,
    pub shift_arithmetic_right_imm_64: Cost,
    pub shift_arithmetic_right_imm_alt_32: Cost,
    pub shift_arithmetic_right_imm_alt_64: Cost,
    pub shift_logical_left_32: Cost,
    pub shift_logical_left_64: Cost,
    pub shift_logical_left_imm_32: Cost,
    pub shift_logical_left_imm_64: Cost,
    pub shift_logical_left_imm_alt_32: Cost,
    pub shift_logical_left_imm_alt_64: Cost,
    pub shift_logical_right_32: Cost,
    pub shift_logical_right_64: Cost,
    pub shift_logical_right_imm_32: Cost,
    pub shift_logical_right_imm_64: Cost,
    pub shift_logical_right_imm_alt_32: Cost,
    pub shift_logical_right_imm_alt_64: Cost,
    pub sign_extend_16: Cost,
    pub sign_extend_8: Cost,
    pub store_imm_indirect_u16: Cost,
    pub store_imm_indirect_u32: Cost,
    pub store_imm_indirect_u64: Cost,
    pub store_imm_indirect_u8: Cost,
    pub store_imm_u16: Cost,
    pub store_imm_u32: Cost,
    pub store_imm_u64: Cost,
    pub store_imm_u8: Cost,
    pub store_indirect_u16: Cost,
    pub store_indirect_u32: Cost,
    pub store_indirect_u64: Cost,
    pub store_indirect_u8: Cost,
    pub store_u16: Cost,
    pub store_u32: Cost,
    pub store_u64: Cost,
    pub store_u8: Cost,
    pub sub_32: Cost,
    pub sub_64: Cost,
    pub trap: Cost,
    pub xnor: Cost,
    pub xor: Cost,
    pub xor_imm: Cost,
    pub zero_extend_16: Cost,
}

static NAIVE_COST_MODEL: CostModel = CostModel::naive();

impl CostModel {
    pub fn naive_ref() -> CostModelRef {
        CostModelRef::from_static(&NAIVE_COST_MODEL)
    }

    pub const fn naive() -> Self {
        CostModel {
            add_32: 1,
            add_64: 1,
            add_imm_32: 1,
            add_imm_64: 1,
            and: 1,
            and_imm: 1,
            and_inverted: 1,
            branch_eq: 1,
            branch_eq_imm: 1,
            branch_greater_or_equal_signed: 1,
            branch_greater_or_equal_signed_imm: 1,
            branch_greater_or_equal_unsigned: 1,
            branch_greater_or_equal_unsigned_imm: 1,
            branch_greater_signed_imm: 1,
            branch_greater_unsigned_imm: 1,
            branch_less_or_equal_signed_imm: 1,
            branch_less_or_equal_unsigned_imm: 1,
            branch_less_signed: 1,
            branch_less_signed_imm: 1,
            branch_less_unsigned: 1,
            branch_less_unsigned_imm: 1,
            branch_not_eq: 1,
            branch_not_eq_imm: 1,
            cmov_if_not_zero: 1,
            cmov_if_not_zero_imm: 1,
            cmov_if_zero: 1,
            cmov_if_zero_imm: 1,
            count_leading_zero_bits_32: 1,
            count_leading_zero_bits_64: 1,
            count_set_bits_32: 1,
            count_set_bits_64: 1,
            count_trailing_zero_bits_32: 1,
            count_trailing_zero_bits_64: 1,
            div_signed_32: 1,
            div_signed_64: 1,
            div_unsigned_32: 1,
            div_unsigned_64: 1,
            ecalli: 1,
            fallthrough: 1,
            invalid: 1,
            jump: 1,
            jump_indirect: 1,
            load_i16: 1,
            load_i32: 1,
            load_i8: 1,
            load_imm: 1,
            load_imm64: 1,
            load_imm_and_jump: 1,
            load_imm_and_jump_indirect: 1,
            load_indirect_i16: 1,
            load_indirect_i32: 1,
            load_indirect_i8: 1,
            load_indirect_u16: 1,
            load_indirect_u32: 1,
            load_indirect_u64: 1,
            load_indirect_u8: 1,
            load_u16: 1,
            load_u32: 1,
            load_u64: 1,
            load_u8: 1,
            maximum: 1,
            maximum_unsigned: 1,
            memset: 1,
            minimum: 1,
            minimum_unsigned: 1,
            move_reg: 1,
            mul_32: 1,
            mul_64: 1,
            mul_imm_32: 1,
            mul_imm_64: 1,
            mul_upper_signed_signed: 1,
            mul_upper_signed_unsigned: 1,
            mul_upper_unsigned_unsigned: 1,
            negate_and_add_imm_32: 1,
            negate_and_add_imm_64: 1,
            or: 1,
            or_imm: 1,
            or_inverted: 1,
            rem_signed_32: 1,
            rem_signed_64: 1,
            rem_unsigned_32: 1,
            rem_unsigned_64: 1,
            reverse_byte: 1,
            rotate_left_32: 1,
            rotate_left_64: 1,
            rotate_right_32: 1,
            rotate_right_64: 1,
            rotate_right_imm_32: 1,
            rotate_right_imm_64: 1,
            rotate_right_imm_alt_32: 1,
            rotate_right_imm_alt_64: 1,
            sbrk: 1,
            set_greater_than_signed_imm: 1,
            set_greater_than_unsigned_imm: 1,
            set_less_than_signed: 1,
            set_less_than_signed_imm: 1,
            set_less_than_unsigned: 1,
            set_less_than_unsigned_imm: 1,
            shift_arithmetic_right_32: 1,
            shift_arithmetic_right_64: 1,
            shift_arithmetic_right_imm_32: 1,
            shift_arithmetic_right_imm_64: 1,
            shift_arithmetic_right_imm_alt_32: 1,
            shift_arithmetic_right_imm_alt_64: 1,
            shift_logical_left_32: 1,
            shift_logical_left_64: 1,
            shift_logical_left_imm_32: 1,
            shift_logical_left_imm_64: 1,
            shift_logical_left_imm_alt_32: 1,
            shift_logical_left_imm_alt_64: 1,
            shift_logical_right_32: 1,
            shift_logical_right_64: 1,
            shift_logical_right_imm_32: 1,
            shift_logical_right_imm_64: 1,
            shift_logical_right_imm_alt_32: 1,
            shift_logical_right_imm_alt_64: 1,
            sign_extend_16: 1,
            sign_extend_8: 1,
            store_imm_indirect_u16: 1,
            store_imm_indirect_u32: 1,
            store_imm_indirect_u64: 1,
            store_imm_indirect_u8: 1,
            store_imm_u16: 1,
            store_imm_u32: 1,
            store_imm_u64: 1,
            store_imm_u8: 1,
            store_indirect_u16: 1,
            store_indirect_u32: 1,
            store_indirect_u64: 1,
            store_indirect_u8: 1,
            store_u16: 1,
            store_u32: 1,
            store_u64: 1,
            store_u8: 1,
            sub_32: 1,
            sub_64: 1,
            trap: 1,
            xnor: 1,
            xor: 1,
            xor_imm: 1,
            zero_extend_16: 1,
        }
    }
}

// TODO: Come up with a better cost model.
pub struct GasVisitor {
    cost_model: CostModelRef,
    cost: u32,
    last_block_cost: Option<u32>,
}

impl GasVisitor {
    pub fn new(cost_model: CostModelRef) -> Self {
        Self {
            cost_model,
            cost: 0,
            last_block_cost: None,
        }
    }

    #[inline]
    fn start_new_basic_block(&mut self) {
        self.last_block_cost = Some(self.cost);
        self.cost = 0;
    }

    #[inline]
    pub fn take_block_cost(&mut self) -> Option<u32> {
        self.last_block_cost.take()
    }
}

impl InstructionVisitor for GasVisitor {
    type ReturnTy = ();

    #[cold]
    fn invalid(&mut self) -> Self::ReturnTy {
        self.cost += self.cost_model.invalid;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn trap(&mut self) -> Self::ReturnTy {
        self.cost += self.cost_model.trap;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn fallthrough(&mut self) -> Self::ReturnTy {
        self.cost += self.cost_model.fallthrough;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn sbrk(&mut self, _d: RawReg, _s: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.sbrk;
    }

    #[inline(always)]
    fn memset(&mut self) -> Self::ReturnTy {
        self.cost += self.cost_model.memset;
    }

    #[inline(always)]
    fn ecalli(&mut self, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.ecalli;
    }

    #[inline(always)]
    fn set_less_than_unsigned(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.set_less_than_unsigned;
    }

    #[inline(always)]
    fn set_less_than_signed(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.set_less_than_signed;
    }

    #[inline(always)]
    fn shift_logical_right_32(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.shift_logical_right_32;
    }

    #[inline(always)]
    fn shift_arithmetic_right_32(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.shift_arithmetic_right_32;
    }

    #[inline(always)]
    fn shift_logical_left_32(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.shift_logical_left_32;
    }

    #[inline(always)]
    fn shift_logical_right_64(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.shift_logical_right_64;
    }

    #[inline(always)]
    fn shift_arithmetic_right_64(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.shift_arithmetic_right_64;
    }

    #[inline(always)]
    fn shift_logical_left_64(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.shift_logical_left_64;
    }

    #[inline(always)]
    fn xor(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.xor;
    }

    #[inline(always)]
    fn and(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.and;
    }

    #[inline(always)]
    fn or(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.or;
    }
    #[inline(always)]
    fn add_32(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.add_32;
    }

    #[inline(always)]
    fn add_64(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.add_64;
    }

    #[inline(always)]
    fn sub_32(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.sub_32;
    }

    #[inline(always)]
    fn sub_64(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.sub_64;
    }

    #[inline(always)]
    fn mul_32(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.mul_32;
    }

    #[inline(always)]
    fn mul_64(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.mul_64;
    }

    #[inline(always)]
    fn mul_upper_signed_signed(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.mul_upper_signed_signed;
    }

    #[inline(always)]
    fn mul_upper_unsigned_unsigned(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.mul_upper_unsigned_unsigned;
    }

    #[inline(always)]
    fn mul_upper_signed_unsigned(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.mul_upper_signed_unsigned;
    }

    #[inline(always)]
    fn div_unsigned_32(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.div_unsigned_32;
    }

    #[inline(always)]
    fn div_signed_32(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.div_signed_32;
    }

    #[inline(always)]
    fn rem_unsigned_32(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.rem_unsigned_32;
    }

    #[inline(always)]
    fn rem_signed_32(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.rem_signed_32;
    }

    #[inline(always)]
    fn div_unsigned_64(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.div_unsigned_64;
    }

    #[inline(always)]
    fn div_signed_64(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.div_signed_64;
    }

    #[inline(always)]
    fn rem_unsigned_64(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.rem_unsigned_64;
    }

    #[inline(always)]
    fn rem_signed_64(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.rem_signed_64;
    }

    #[inline(always)]
    fn and_inverted(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.and_inverted;
    }

    #[inline(always)]
    fn or_inverted(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.or_inverted;
    }

    #[inline(always)]
    fn xnor(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.xnor;
    }

    #[inline(always)]
    fn maximum(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.maximum;
    }

    #[inline(always)]
    fn maximum_unsigned(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.maximum_unsigned;
    }

    #[inline(always)]
    fn minimum(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.minimum;
    }

    #[inline(always)]
    fn minimum_unsigned(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.minimum_unsigned;
    }

    #[inline(always)]
    fn rotate_left_32(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.rotate_left_32;
    }

    #[inline(always)]
    fn rotate_left_64(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.rotate_left_64;
    }

    #[inline(always)]
    fn rotate_right_32(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.rotate_right_32;
    }

    #[inline(always)]
    fn rotate_right_64(&mut self, _d: RawReg, _s1: RawReg, _s2: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.rotate_right_64;
    }

    #[inline(always)]
    fn mul_imm_32(&mut self, _d: RawReg, _s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.mul_imm_32;
    }

    #[inline(always)]
    fn mul_imm_64(&mut self, _d: RawReg, _s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.mul_imm_64;
    }

    #[inline(always)]
    fn set_less_than_unsigned_imm(&mut self, _d: RawReg, _s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.set_less_than_unsigned_imm;
    }

    #[inline(always)]
    fn set_less_than_signed_imm(&mut self, _d: RawReg, _s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.set_less_than_signed_imm;
    }

    #[inline(always)]
    fn set_greater_than_unsigned_imm(&mut self, _d: RawReg, _s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.set_greater_than_unsigned_imm;
    }

    #[inline(always)]
    fn set_greater_than_signed_imm(&mut self, _d: RawReg, _s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.set_greater_than_signed_imm;
    }

    #[inline(always)]
    fn shift_logical_right_imm_32(&mut self, _d: RawReg, _s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.shift_logical_right_imm_32;
    }

    #[inline(always)]
    fn shift_arithmetic_right_imm_32(&mut self, _d: RawReg, _s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.shift_arithmetic_right_imm_32;
    }

    #[inline(always)]
    fn shift_logical_left_imm_32(&mut self, _d: RawReg, _s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.shift_logical_left_imm_32;
    }

    #[inline(always)]
    fn shift_logical_right_imm_alt_32(&mut self, _d: RawReg, _s2: RawReg, _s1: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.shift_logical_right_imm_alt_32;
    }

    #[inline(always)]
    fn shift_arithmetic_right_imm_alt_32(&mut self, _d: RawReg, _s2: RawReg, _s1: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.shift_arithmetic_right_imm_alt_32;
    }

    #[inline(always)]
    fn shift_logical_left_imm_alt_32(&mut self, _d: RawReg, _s2: RawReg, _s1: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.shift_logical_left_imm_alt_32;
    }

    #[inline(always)]
    fn shift_logical_right_imm_64(&mut self, _d: RawReg, _s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.shift_logical_right_imm_64;
    }

    #[inline(always)]
    fn shift_arithmetic_right_imm_64(&mut self, _d: RawReg, _s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.shift_arithmetic_right_imm_64;
    }

    #[inline(always)]
    fn shift_logical_left_imm_64(&mut self, _d: RawReg, _s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.shift_logical_left_imm_64;
    }

    #[inline(always)]
    fn shift_logical_right_imm_alt_64(&mut self, _d: RawReg, _s2: RawReg, _s1: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.shift_logical_right_imm_alt_64;
    }

    #[inline(always)]
    fn shift_arithmetic_right_imm_alt_64(&mut self, _d: RawReg, _s2: RawReg, _s1: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.shift_arithmetic_right_imm_alt_64;
    }

    #[inline(always)]
    fn shift_logical_left_imm_alt_64(&mut self, _d: RawReg, _s2: RawReg, _s1: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.shift_logical_left_imm_alt_64;
    }

    #[inline(always)]
    fn or_imm(&mut self, _d: RawReg, _s: RawReg, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.or_imm;
    }

    #[inline(always)]
    fn and_imm(&mut self, _d: RawReg, _s: RawReg, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.and_imm;
    }

    #[inline(always)]
    fn xor_imm(&mut self, _d: RawReg, _s: RawReg, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.xor_imm;
    }

    #[inline(always)]
    fn move_reg(&mut self, _d: RawReg, _s: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.move_reg;
    }

    #[inline(always)]
    fn count_leading_zero_bits_32(&mut self, _d: RawReg, _s: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.count_leading_zero_bits_32;
    }

    #[inline(always)]
    fn count_leading_zero_bits_64(&mut self, _d: RawReg, _s: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.count_leading_zero_bits_64;
    }

    #[inline(always)]
    fn count_trailing_zero_bits_32(&mut self, _d: RawReg, _s: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.count_trailing_zero_bits_32;
    }

    #[inline(always)]
    fn count_trailing_zero_bits_64(&mut self, _d: RawReg, _s: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.count_trailing_zero_bits_64;
    }

    #[inline(always)]
    fn count_set_bits_32(&mut self, _d: RawReg, _s: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.count_set_bits_32;
    }

    #[inline(always)]
    fn count_set_bits_64(&mut self, _d: RawReg, _s: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.count_set_bits_64;
    }

    #[inline(always)]
    fn sign_extend_8(&mut self, _d: RawReg, _s: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.sign_extend_8;
    }

    #[inline(always)]
    fn sign_extend_16(&mut self, _d: RawReg, _s: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.sign_extend_16;
    }

    #[inline(always)]
    fn zero_extend_16(&mut self, _d: RawReg, _s: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.zero_extend_16;
    }

    #[inline(always)]
    fn reverse_byte(&mut self, _d: RawReg, _s: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.reverse_byte;
    }

    #[inline(always)]
    fn cmov_if_zero(&mut self, _d: RawReg, _s: RawReg, _c: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.cmov_if_zero;
    }

    #[inline(always)]
    fn cmov_if_not_zero(&mut self, _d: RawReg, _s: RawReg, _c: RawReg) -> Self::ReturnTy {
        self.cost += self.cost_model.cmov_if_not_zero;
    }

    #[inline(always)]
    fn cmov_if_zero_imm(&mut self, _d: RawReg, _c: RawReg, _s: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.cmov_if_zero_imm;
    }

    #[inline(always)]
    fn cmov_if_not_zero_imm(&mut self, _d: RawReg, _c: RawReg, _s: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.cmov_if_not_zero_imm;
    }

    #[inline(always)]
    fn rotate_right_imm_32(&mut self, _d: RawReg, _s: RawReg, _c: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.rotate_right_imm_32;
    }

    #[inline(always)]
    fn rotate_right_imm_alt_32(&mut self, _d: RawReg, _s: RawReg, _c: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.rotate_right_imm_alt_32;
    }

    #[inline(always)]
    fn rotate_right_imm_64(&mut self, _d: RawReg, _s: RawReg, _c: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.rotate_right_imm_64;
    }

    #[inline(always)]
    fn rotate_right_imm_alt_64(&mut self, _d: RawReg, _s: RawReg, _c: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.rotate_right_imm_alt_64;
    }

    #[inline(always)]
    fn add_imm_32(&mut self, _d: RawReg, _s: RawReg, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.add_imm_32;
    }

    #[inline(always)]
    fn add_imm_64(&mut self, _d: RawReg, _s: RawReg, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.add_imm_64;
    }

    #[inline(always)]
    fn negate_and_add_imm_32(&mut self, _d: RawReg, _s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.negate_and_add_imm_32;
    }

    #[inline(always)]
    fn negate_and_add_imm_64(&mut self, _d: RawReg, _s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.negate_and_add_imm_64;
    }

    #[inline(always)]
    fn store_imm_indirect_u8(&mut self, _base: RawReg, _offset: u32, _value: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.store_imm_indirect_u8;
    }

    #[inline(always)]
    fn store_imm_indirect_u16(&mut self, _base: RawReg, _offset: u32, _value: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.store_imm_indirect_u16;
    }

    #[inline(always)]
    fn store_imm_indirect_u32(&mut self, _base: RawReg, _offset: u32, _value: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.store_imm_indirect_u32;
    }

    #[inline(always)]
    fn store_imm_indirect_u64(&mut self, _base: RawReg, _offset: u32, _value: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.store_imm_indirect_u64;
    }

    #[inline(always)]
    fn store_indirect_u8(&mut self, _src: RawReg, _base: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.store_indirect_u8;
    }

    #[inline(always)]
    fn store_indirect_u16(&mut self, _src: RawReg, _base: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.store_indirect_u16;
    }

    #[inline(always)]
    fn store_indirect_u32(&mut self, _src: RawReg, _base: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.store_indirect_u32;
    }

    #[inline(always)]
    fn store_indirect_u64(&mut self, _src: RawReg, _base: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.store_indirect_u64;
    }

    #[inline(always)]
    fn store_imm_u8(&mut self, _offset: u32, _value: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.store_imm_u8;
    }

    #[inline(always)]
    fn store_imm_u16(&mut self, _offset: u32, _value: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.store_imm_u16;
    }

    #[inline(always)]
    fn store_imm_u32(&mut self, _offset: u32, _value: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.store_imm_u32;
    }

    #[inline(always)]
    fn store_imm_u64(&mut self, _offset: u32, _value: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.store_imm_u64;
    }

    #[inline(always)]
    fn store_u8(&mut self, _src: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.store_u8;
    }

    #[inline(always)]
    fn store_u16(&mut self, _src: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.store_u16;
    }

    #[inline(always)]
    fn store_u32(&mut self, _src: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.store_u32;
    }

    #[inline(always)]
    fn store_u64(&mut self, _src: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.store_u64;
    }

    #[inline(always)]
    fn load_indirect_u8(&mut self, _dst: RawReg, _base: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.load_indirect_u8;
    }

    #[inline(always)]
    fn load_indirect_i8(&mut self, _dst: RawReg, _base: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.load_indirect_i8;
    }

    #[inline(always)]
    fn load_indirect_u16(&mut self, _dst: RawReg, _base: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.load_indirect_u16;
    }

    #[inline(always)]
    fn load_indirect_i16(&mut self, _dst: RawReg, _base: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.load_indirect_i16;
    }

    #[inline(always)]
    fn load_indirect_u32(&mut self, _dst: RawReg, _base: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.load_indirect_u32;
    }

    #[inline(always)]
    fn load_indirect_i32(&mut self, _dst: RawReg, _base: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.load_indirect_i32;
    }

    #[inline(always)]
    fn load_indirect_u64(&mut self, _dst: RawReg, _base: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.load_indirect_u64;
    }

    #[inline(always)]
    fn load_u8(&mut self, _dst: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.load_u8;
    }

    #[inline(always)]
    fn load_i8(&mut self, _dst: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.load_i8;
    }

    #[inline(always)]
    fn load_u16(&mut self, _dst: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.load_u16;
    }

    #[inline(always)]
    fn load_i16(&mut self, _dst: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.load_i16;
    }

    #[inline(always)]
    fn load_u32(&mut self, _dst: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.load_u32;
    }

    #[inline(always)]
    fn load_i32(&mut self, _dst: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.load_i32;
    }

    #[inline(always)]
    fn load_u64(&mut self, _dst: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.load_u64;
    }

    #[inline(always)]
    fn branch_less_unsigned(&mut self, _s1: RawReg, _s2: RawReg, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.branch_less_unsigned;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn branch_less_signed(&mut self, _s1: RawReg, _s2: RawReg, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.branch_less_signed;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn branch_greater_or_equal_unsigned(&mut self, _s1: RawReg, _s2: RawReg, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.branch_greater_or_equal_unsigned;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn branch_greater_or_equal_signed(&mut self, _s1: RawReg, _s2: RawReg, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.branch_greater_or_equal_signed;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn branch_eq(&mut self, _s1: RawReg, _s2: RawReg, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.branch_eq;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn branch_not_eq(&mut self, _s1: RawReg, _s2: RawReg, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.branch_not_eq;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn branch_eq_imm(&mut self, _s1: RawReg, _s2: u32, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.branch_eq_imm;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn branch_not_eq_imm(&mut self, _s1: RawReg, _s2: u32, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.branch_not_eq_imm;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn branch_less_unsigned_imm(&mut self, _s1: RawReg, _s2: u32, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.branch_less_unsigned_imm;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn branch_less_signed_imm(&mut self, _s1: RawReg, _s2: u32, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.branch_less_signed_imm;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn branch_greater_or_equal_unsigned_imm(&mut self, _s1: RawReg, _s2: u32, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.branch_greater_or_equal_unsigned_imm;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn branch_greater_or_equal_signed_imm(&mut self, _s1: RawReg, _s2: u32, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.branch_greater_or_equal_signed_imm;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn branch_less_or_equal_unsigned_imm(&mut self, _s1: RawReg, _s2: u32, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.branch_less_or_equal_unsigned_imm;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn branch_less_or_equal_signed_imm(&mut self, _s1: RawReg, _s2: u32, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.branch_less_or_equal_signed_imm;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn branch_greater_unsigned_imm(&mut self, _s1: RawReg, _s2: u32, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.branch_greater_unsigned_imm;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn branch_greater_signed_imm(&mut self, _s1: RawReg, _s2: u32, _imm: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.branch_greater_signed_imm;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn load_imm(&mut self, _dst: RawReg, _value: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.load_imm;
    }

    #[inline(always)]
    fn load_imm64(&mut self, _dst: RawReg, _value: u64) -> Self::ReturnTy {
        self.cost += self.cost_model.load_imm64;
    }

    #[inline(always)]
    fn load_imm_and_jump(&mut self, _ra: RawReg, _value: u32, _target: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.load_imm_and_jump;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn load_imm_and_jump_indirect(&mut self, _ra: RawReg, _base: RawReg, _value: u32, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.load_imm_and_jump_indirect;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn jump(&mut self, _target: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.jump;
        self.start_new_basic_block();
    }

    #[inline(always)]
    fn jump_indirect(&mut self, _base: RawReg, _offset: u32) -> Self::ReturnTy {
        self.cost += self.cost_model.jump_indirect;
        self.start_new_basic_block();
    }
}

pub fn calculate_for_block<I>(cost_model: CostModelRef, mut instructions: Instructions<I>) -> (u32, bool)
where
    I: InstructionSet,
{
    let mut visitor = GasVisitor::new(cost_model);
    while instructions.visit(&mut visitor).is_some() {
        if let Some(cost) = visitor.last_block_cost {
            return (cost, false);
        }
    }

    if let Some(cost) = visitor.last_block_cost {
        (cost, false)
    } else {
        let started_out_of_bounds = visitor.cost == 0;

        // We've ended out of bounds, so assume there's an implicit trap there.
        visitor.trap();
        (visitor.last_block_cost.unwrap(), started_out_of_bounds)
    }
}

pub fn trap_cost(cost_model: CostModelRef) -> u32 {
    let mut gas_visitor = GasVisitor::new(cost_model);
    gas_visitor.trap();
    gas_visitor.take_block_cost().unwrap()
}
