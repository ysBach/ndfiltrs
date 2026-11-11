use std::fmt;

/// Errors returned by filter plans and kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// A boundary mode had to map an index on an axis of length zero.
    EmptyAxis,
    /// The geometry, reduction, or policy is invalid. The text names the
    /// rule.
    InvalidConfiguration(&'static str),
    /// A buffer length differs from the configured shape.
    LengthMismatch,
    /// A workspace allocation failed.
    Memory,
    /// Shape or index arithmetic overflowed.
    Overflow,
}

impl fmt::Display for Error {
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

impl std::error::Error for Error {}
