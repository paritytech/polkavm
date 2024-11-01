pub use polkavm::*;

pub struct Module(polkavm::Module);

impl Module {
    pub fn new(engine: &Engine, config: &ModuleConfig, bytes: ArcBytes) -> Result<Self, Error> {
        polkavm::Module::new(engine, config, bytes).map(Module)
    }

    pub fn instantiate(&self) -> Result<RawInstance, Error> {
        self.0.instantiate().map(RawInstance)
    }
}

impl core::ops::Deref for Module {
    type Target = polkavm::Module;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for Module {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}


pub struct RawInstance(polkavm::RawInstance);

impl RawInstance {
    pub fn gas(&self) -> Gas {
        self.0.gas()
    }

    pub fn set_gas(&mut self, gas: Gas) {
        self.0.set_gas(gas)
    }

    pub fn reg(&self, reg: Reg) -> u32 {
        self.0.reg(reg) as u32
    }

    pub fn set_reg(&mut self, reg: Reg, value: u32) {
        self.0.set_reg(reg, value as u64)
    }
}

impl core::ops::Deref for RawInstance {
    type Target = polkavm::RawInstance;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for RawInstance {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
