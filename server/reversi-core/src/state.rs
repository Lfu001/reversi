use common::DiskColor;

/// A trait which provides an extension method for the [`DiskColor`].
pub trait DiskColorExt {
    /// Returns the opposite color.
    fn opposite(&self) -> DiskColor;
}

impl DiskColorExt for DiskColor {
    fn opposite(&self) -> DiskColor {
        match self {
            DiskColor::Light => DiskColor::Dark,
            DiskColor::Dark => DiskColor::Light,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disk_color() {
        assert_eq!(DiskColor::Dark.opposite(), DiskColor::Light);
        assert_eq!(DiskColor::Light.opposite(), DiskColor::Dark);
    }
}
