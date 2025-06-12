use alloc::sync::Arc;
use alloc::vec::Vec;
use polkavm_common::program::{InstructionSet, InstructionVisitor, Instructions, Opcode, RawReg};

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

macro_rules! define_cost_model_struct {
    (@count) => {
        0
    };

    (@count $f0:ident $f1:ident $f2:ident $f3:ident $f4:ident $f5:ident $f6:ident $f7:ident $($rest:ident)*) => {
        8 + define_cost_model_struct!(@count $($rest)*)
    };

    (@count $f0:ident $($rest:ident)*) => {
        1 + define_cost_model_struct!(@count $($rest)*)
    };

    (
        version: $version:expr,
        $($field:ident,)+
    ) => {
        const COST_MODEL_FIELDS: usize = define_cost_model_struct!(@count $($field)+) + 1;

        #[allow(clippy::exhaustive_structs)]
        #[derive(Hash)]
        pub struct CostModel {
            $(
                pub $field: Cost,
            )+

            pub invalid: u32,
        }

        impl CostModel {
            /// A naive gas cost model where every instruction costs one gas.
            pub const fn naive() -> Self {
                CostModel {
                    $(
                        $field: 1,
                    )+

                    invalid: 1,
                }
            }

            /// Serializes the cost model into a byte blob.
            pub fn serialize(&self) -> Vec<u8> {
                let mut output = Vec::with_capacity((COST_MODEL_FIELDS + 2) * 4);
                let version: u32 = $version;
                output.extend_from_slice(&version.to_le_bytes());

                $(
                    output.extend_from_slice(&self.$field.to_le_bytes());
                )+
                output.extend_from_slice(&self.invalid.to_le_bytes());
                output
            }

            /// Deserializes the cost model from a byte blob.
            pub fn deserialize(blob: &[u8]) -> Option<CostModel> {
                if (blob.len() % 4) != 0 || blob.len() / 4 != (COST_MODEL_FIELDS + 2) {
                    return None;
                }

                if u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]) != $version {
                    return None;
                }

                let mut model = CostModel::naive();
                let mut position = 4;
                $(
                    model.$field = u32::from_le_bytes([blob[position], blob[position + 1], blob[position + 2], blob[position + 3]]);
                    position += 4;
                )+

                model.invalid = u32::from_le_bytes([blob[position], blob[position + 1], blob[position + 2], blob[position + 3]]);

                assert_eq!(position, (COST_MODEL_FIELDS + 2) * 4);
                Some(model)
            }

            /// Gets the cost of a given opcode.
            pub fn cost_for_opcode(&self, opcode: Opcode) -> u32 {
                match opcode {
                    $(
                        Opcode::$field => self.$field,
                    )+
                }
            }
        }
    }
}

define_cost_model_struct! {
    version: 1,

    add_32,
    add_64,
    add_imm_32,
    add_imm_64,
    and,
    and_imm,
    and_inverted,
    branch_eq,
    branch_eq_imm,
    branch_greater_or_equal_signed,
    branch_greater_or_equal_signed_imm,
    branch_greater_or_equal_unsigned,
    branch_greater_or_equal_unsigned_imm,
    branch_greater_signed_imm,
    branch_greater_unsigned_imm,
    branch_less_or_equal_signed_imm,
    branch_less_or_equal_unsigned_imm,
    branch_less_signed,
    branch_less_signed_imm,
    branch_less_unsigned,
    branch_less_unsigned_imm,
    branch_not_eq,
    branch_not_eq_imm,
    cmov_if_not_zero,
    cmov_if_not_zero_imm,
    cmov_if_zero,
    cmov_if_zero_imm,
    count_leading_zero_bits_32,
    count_leading_zero_bits_64,
    count_set_bits_32,
    count_set_bits_64,
    count_trailing_zero_bits_32,
    count_trailing_zero_bits_64,
    div_signed_32,
    div_signed_64,
    div_unsigned_32,
    div_unsigned_64,
    ecalli,
    fallthrough,
    jump,
    jump_indirect,
    load_i16,
    load_i32,
    load_i8,
    load_imm,
    load_imm64,
    load_imm_and_jump,
    load_imm_and_jump_indirect,
    load_indirect_i16,
    load_indirect_i32,
    load_indirect_i8,
    load_indirect_u16,
    load_indirect_u32,
    load_indirect_u64,
    load_indirect_u8,
    load_u16,
    load_u32,
    load_u64,
    load_u8,
    maximum,
    maximum_unsigned,
    memset,
    minimum,
    minimum_unsigned,
    move_reg,
    mul_32,
    mul_64,
    mul_imm_32,
    mul_imm_64,
    mul_upper_signed_signed,
    mul_upper_signed_unsigned,
    mul_upper_unsigned_unsigned,
    negate_and_add_imm_32,
    negate_and_add_imm_64,
    or,
    or_imm,
    or_inverted,
    rem_signed_32,
    rem_signed_64,
    rem_unsigned_32,
    rem_unsigned_64,
    reverse_byte,
    rotate_left_32,
    rotate_left_64,
    rotate_right_32,
    rotate_right_64,
    rotate_right_imm_32,
    rotate_right_imm_64,
    rotate_right_imm_alt_32,
    rotate_right_imm_alt_64,
    sbrk,
    set_greater_than_signed_imm,
    set_greater_than_unsigned_imm,
    set_less_than_signed,
    set_less_than_signed_imm,
    set_less_than_unsigned,
    set_less_than_unsigned_imm,
    shift_arithmetic_right_32,
    shift_arithmetic_right_64,
    shift_arithmetic_right_imm_32,
    shift_arithmetic_right_imm_64,
    shift_arithmetic_right_imm_alt_32,
    shift_arithmetic_right_imm_alt_64,
    shift_logical_left_32,
    shift_logical_left_64,
    shift_logical_left_imm_32,
    shift_logical_left_imm_64,
    shift_logical_left_imm_alt_32,
    shift_logical_left_imm_alt_64,
    shift_logical_right_32,
    shift_logical_right_64,
    shift_logical_right_imm_32,
    shift_logical_right_imm_64,
    shift_logical_right_imm_alt_32,
    shift_logical_right_imm_alt_64,
    sign_extend_16,
    sign_extend_8,
    store_imm_indirect_u16,
    store_imm_indirect_u32,
    store_imm_indirect_u64,
    store_imm_indirect_u8,
    store_imm_u16,
    store_imm_u32,
    store_imm_u64,
    store_imm_u8,
    store_indirect_u16,
    store_indirect_u32,
    store_indirect_u64,
    store_indirect_u8,
    store_u16,
    store_u32,
    store_u64,
    store_u8,
    sub_32,
    sub_64,
    trap,
    xnor,
    xor,
    xor_imm,
    zero_extend_16,
}

static NAIVE_COST_MODEL: CostModel = CostModel::naive();

impl CostModel {
    pub fn naive_ref() -> CostModelRef {
        CostModelRef::from_static(&NAIVE_COST_MODEL)
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
