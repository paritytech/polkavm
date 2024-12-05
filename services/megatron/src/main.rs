#![no_std]
#![no_main]

extern crate alloc;


use core::convert::TryInto;
use simplealloc::SimpleAlloc;

#[global_allocator]
static ALLOCATOR: SimpleAlloc<4096> = SimpleAlloc::new();

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        core::arch::asm!("unimp", options(noreturn));
    }
}

#[polkavm_derive::polkavm_import]
extern "C" {
    #[polkavm_import(index = 2)]
    pub fn read(service: u32, key_ptr: *const u8, key_len: u32, out: *mut u8, out_len: u32) -> u32;
    #[polkavm_import(index = 3)]
    pub fn write(key_ptr: *const u8, key_len: u32, value: *const u8, value_len: u32) -> u32;
}

#[polkavm_derive::polkavm_export]
extern "C" fn is_authorized() -> u32 {
    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn refine() -> u32 {
    unsafe {
        core::arch::asm!(
            "li a3, 0xFEFF0004",
            "li a4, 0x24", 
        );
    }
    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate() -> u32 {
    // Define keys and buffers for data storage
    let buffer0 = &mut [0u8; 12];
    let buffer1 = &mut [0u8; 12];
    let key = [0u8; 1];
    let buffer = &mut [0u8; 12]; // output buffer for final accumulation

    // METHOD 2: UNCLEAR HOW THIS WORKS with PREREQS in 0.4.5 -- EVERYONE should research
    unsafe {
        /*
                payload := make([]byte, 4)
                binary.LittleEndian.PutUint32(payload, uint32(megaN))
                payloadM := make([]byte, 8)
                binary.LittleEndian.PutUint32(payloadM[0:4], uint32(service0.ServiceCode))
                binary.LittleEndian.PutUint32(payloadM[4:8], uint32(service1.ServiceCode))
                workPackage := types.WorkPackage{
                        Authorization: []byte("0x"), // TODO: set up null-authorizer                                                                                                                   
                  AuthCodeHost:  serviceM.ServiceCode,
                        Authorizer:    types.Authorizer{},
                        RefineContext: refineContext,
                        WorkItems: []types.WorkItem{
                                {
                                        Service:          service0.ServiceCode,
                                        CodeHash:         service0.CodeHash,
                                        Payload:          payload,
                                        GasLimit:         10000000,
                                        ImportedSegments: importedSegments,
                                        ExportCount:      1,
                                },
                                {
                                        Service:          service1.ServiceCode,
                                        CodeHash:         service1.CodeHash,
                                        Payload:          payload,
                                        GasLimit:         10000000,
                                        ImportedSegments: importedSegments,
                                        ExportCount:      1,
                                },
                                {
                                        Service:          serviceM.ServiceCode,
                                        CodeHash:         serviceM.CodeHash,
                                        Payload:          payloadM,
                                        GasLimit:         10000000,
                                        ImportedSegments: importedSegmentsM,
                                        ExportCount:      0,
                                },
			},
                }
         */
        
        // Service codes: QUESTION - Does this work to get the payload of 4 bytes of service0 + service1 -- if not, we will need the refine results to come with it for Fib + Tribonnaci
        let ptr0 = 0xFEFF0000 as *mut u32;
        let ptr1 = 0xFEFF0004 as *mut u32;
        // QUESTION: how can we ensure that service0+service1 accumulation happens BEFORE this serviceM -- Is it possible to guarantee?
        let service0 = *ptr0;
        let service1 = *ptr1;

        // Read value for key from Service 0 into buffer0
        read(service0, key.as_ptr(), key.len() as u32, buffer0.as_mut_ptr(), buffer0.len() as u32);
        // Read value for key from Service 1 into buffer1
        read(service1, key.as_ptr(), key.len() as u32, buffer1.as_mut_ptr(), buffer1.len() as u32);

        let s0_n = u32::from_le_bytes(buffer0[0..4].try_into().unwrap());
        let s0_vn = u32::from_le_bytes(buffer0[4..8].try_into().unwrap());
        let s0_vnminus1 = u32::from_le_bytes(buffer0[8..12].try_into().unwrap());

        // NOT USED: let s1_n = u32::from_le_bytes(buffer1[0..4].try_into().unwrap());
        let s1_vn = u32::from_le_bytes(buffer1[4..8].try_into().unwrap());
        let s1_vnminus1 = u32::from_le_bytes(buffer1[8..12].try_into().unwrap());

        // Sum the corresponding u32s
        let m_n = s0_n; // Assuming m_n does not change
        let m_vn = s0_vn + s1_vn;
        let m_vnminus1 = s0_vnminus1 + s1_vnminus1;

        buffer[0..4].copy_from_slice(&m_n.to_le_bytes());
        buffer[4..8].copy_from_slice(&m_vn.to_le_bytes());
        buffer[8..12].copy_from_slice(&m_vnminus1.to_le_bytes());

        write(key.as_ptr(), key.len() as u32, buffer.as_ptr(), buffer.len() as u32);
    }

    // METHOD 1: WRANGLING 3 RESULTS (service0, service1, serviceM) and write key 0 and value being 12 bytes Megatron[n] = service0[n] + service1[n]
    unsafe {
        // first 12 bytes 0xFEFF0000..0xFEFF000C
        let service0_result = core::slice::from_raw_parts(0xFEFF0000 as *const u8, 12);
        // next 12 bytes 0xFEFF000C..0xFEFF0018
        let service1_result = core::slice::from_raw_parts(0xFEFF000C as *const u8, 12);

        // Interpret each buffer as 3 u32s and perform accumulation
        let s0_n = u32::from_le_bytes(service0_result[0..4].try_into().unwrap());
        let s0_vn = u32::from_le_bytes(service0_result[4..8].try_into().unwrap());
        let s0_vnminus1 = u32::from_le_bytes(service0_result[4..6].try_into().unwrap());

        // NOT USED: let s1_n = u32::from_le_bytes(service1_result[0..4].try_into().unwrap());
        let s1_vn = u32::from_le_bytes(service1_result[4..8].try_into().unwrap());
        let s1_vnminus1 = u32::from_le_bytes(service1_result[8..12].try_into().unwrap());

        // Sum the corresponding u32s
        let m_n = s0_n; // Assuming m_n does not change
        let m_vn = s0_vn + s1_vn;
        let m_vnminus1 = s0_vnminus1 + s1_vnminus1;

        // Store the result in buffer as 12 bytes (3 u32s)
        buffer[0..4].copy_from_slice(&m_n.to_le_bytes());
        buffer[4..8].copy_from_slice(&m_vn.to_le_bytes());
        buffer[8..12].copy_from_slice(&m_vnminus1.to_le_bytes());

        write(key.as_ptr(), key.len() as u32, buffer.as_ptr(), buffer.len() as u32);

        // FINALLY write out accumulation result which is either buffer or buffer2
        let buffer_addr = buffer.as_ptr() as u32;
        let buffer_len = buffer.len() as u32;
        core::arch::asm!(
                       "mv a3, {0}",
                       "mv a4, {1}",
                       in(reg) buffer_addr,
                       in(reg) buffer_len,
        );
    }

    0 // Return success
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer() -> u32 {
    0
}
