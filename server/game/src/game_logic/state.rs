use common::{position, Board, DiskColor, Position};
use num_traits::FromPrimitive;

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

const DIRECTIONS: [(i8, i8); 8] = [
    (1, 0),   // right
    (1, -1),  // right up
    (0, -1),  // up
    (-1, -1), // left up
    (-1, 0),  // left
    (-1, 1),  // left down
    (0, 1),   // down
    (1, 1),   // right down
];

/// A trait which provides an extension method for the [`Board`].
pub trait BoardExt {
    /// Sets a disk to the specified position. This operation does not consider reversi rules.
    ///
    /// # Arguments
    ///
    /// * `position` - A position of the square in the board.
    /// * `color` - A color of the disk.
    fn set_disk(&mut self, position: Position, color: DiskColor);
    /// Gets a disk from the specified position.
    ///
    /// # Arguments
    ///
    /// * `position` - A position of the square in the board.
    fn get_disk(&self, position: &Position) -> Option<DiskColor>;
    /// Gets all the rays from the specified position.
    fn get_rays(&self, position: &Position) -> [Vec<(Option<DiskColor>, Position)>; 8];
}

impl BoardExt for Board {
    fn set_disk(&mut self, position: Position, color: DiskColor) {
        self[position.into()] = Some(color);
    }

    fn get_disk(&self, position: &Position) -> Option<DiskColor> {
        self[*position]
    }

    fn get_rays(&self, position: &Position) -> [Vec<(Option<DiskColor>, Position)>; 8] {
        let mut rays = vec![vec![]; 8];
        for (i, &(dx, dy)) in DIRECTIONS.iter().enumerate() {
            let mut column = position.column() as i8 + dx;
            let mut row = position.row() as i8 + dy;
            while (0..8).contains(&column) && (0..8).contains(&row) {
                let position = position!(
                    FromPrimitive::from_i8(row).unwrap(),
                    FromPrimitive::from_i8(column).unwrap()
                );
                let disk = self.get_disk(&position);
                rays[i].push((disk.to_owned(), position));
                column += dx;
                row += dy;
            }
        }
        rays.try_into().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::{Column, Row};

    #[test]
    fn test_disk_color() {
        assert_eq!(DiskColor::Dark.opposite(), DiskColor::Light);
        assert_eq!(DiskColor::Light.opposite(), DiskColor::Dark);
    }

    #[test]
    fn test_get_rays() {
        let board = Board::default();
        let rays = board.get_rays(&position!(Row::Four, Column::C));

        // right
        assert_eq!(
            rays[0],
            vec![
                (Some(DiskColor::Light), position!(Row::Four, Column::D)),
                (Some(DiskColor::Dark), position!(Row::Four, Column::E)),
                (None, position!(Row::Four, Column::F)),
                (None, position!(Row::Four, Column::G)),
                (None, position!(Row::Four, Column::H))
            ]
        );
        // right up
        assert_eq!(
            rays[1],
            vec![
                (None, position!(Row::Three, Column::D)),
                (None, position!(Row::Two, Column::E)),
                (None, position!(Row::One, Column::F)),
            ]
        );
        // up
        assert_eq!(
            rays[2],
            vec![
                (None, position!(Row::Three, Column::C)),
                (None, position!(Row::Two, Column::C)),
                (None, position!(Row::One, Column::C)),
            ]
        );
        // left up
        assert_eq!(
            rays[3],
            vec![
                (None, position!(Row::Three, Column::B)),
                (None, position!(Row::Two, Column::A)),
            ]
        );
        // left
        assert_eq!(
            rays[4],
            vec![
                (None, position!(Row::Four, Column::B)),
                (None, position!(Row::Four, Column::A)),
            ]
        );
        // left down
        assert_eq!(
            rays[5],
            vec![
                (None, position!(Row::Five, Column::B)),
                (None, position!(Row::Six, Column::A)),
            ]
        );
        // down
        assert_eq!(
            rays[6],
            vec![
                (None, position!(Row::Five, Column::C)),
                (None, position!(Row::Six, Column::C)),
                (None, position!(Row::Seven, Column::C)),
                (None, position!(Row::Eight, Column::C)),
            ]
        );
        // right down
        assert_eq!(
            rays[7],
            vec![
                (Some(DiskColor::Dark), position!(Row::Five, Column::D)),
                (None, position!(Row::Six, Column::E)),
                (None, position!(Row::Seven, Column::F)),
                (None, position!(Row::Eight, Column::G)),
            ]
        );
    }
}
