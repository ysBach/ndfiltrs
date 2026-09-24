use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum KernelError {
    EmptyAxis,
    InvalidConfiguration(&'static str),
    LengthMismatch,
    Memory,
    Overflow,
}

impl fmt::Display for KernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyAxis => formatter.write_str("cannot map an empty axis"),
            Self::InvalidConfiguration(message) => formatter.write_str(message),
            Self::LengthMismatch => {
                formatter.write_str("input length does not match the configured shape")
            }
            Self::Memory => formatter.write_str("unable to allocate filtering workspace"),
            Self::Overflow => formatter.write_str("shape or index arithmetic overflowed"),
        }
    }
}

impl std::error::Error for KernelError {}
