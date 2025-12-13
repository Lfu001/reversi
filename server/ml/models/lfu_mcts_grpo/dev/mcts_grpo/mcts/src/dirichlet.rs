use rand_distr::{Distribution, Gamma};

/// Dirichlet distribution using the Gamma-based method.
///
/// When `X_i ~ Gamma(α_i, 1)`, then `Y_i = X_i / Σ(X_j)` follows Dirichlet(α_1, ..., α_n).
pub struct Dirichlet {
    /// Gamma distribution with alpha parameter.
    gamma: Gamma<f64>,
    /// Buffer for storing samples.
    buffer: Vec<f64>,
}

impl Dirichlet {
    /// Creates a new [`Dirichlet`].
    pub fn new(alpha: f64, buffer_capacity: usize) -> Self {
        Self {
            gamma: Gamma::new(alpha, 1.0).unwrap(),
            buffer: Vec::with_capacity(buffer_capacity),
        }
    }

    /// Samples from the Dirichlet distribution.
    ///
    /// Returns `None` if `size` is 0.
    pub fn sample(&mut self, rng: &mut impl rand::Rng, size: usize) -> Option<&[f64]> {
        if size == 0 {
            return Some(&[]);
        }

        self.buffer.clear();
        let mut l1_norm = 0.0;
        for _ in 0..size {
            let gamma_sample = self.gamma.sample(rng);
            self.buffer.push(gamma_sample);
            l1_norm += gamma_sample;
        }

        if l1_norm > 0.0 {
            for element in self.buffer.iter_mut() {
                *element /= l1_norm;
            }
            Some(&self.buffer)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_sample_dirichlet_empty() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut distribution = Dirichlet::new(1.0, 0);
        let result = distribution.sample(&mut rng, 0);
        let expected = vec![];
        assert_eq!(result, Some(expected.as_slice()));
    }

    #[test]
    fn test_sample_dirichlet_single() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut distribution = Dirichlet::new(1.0, 1);
        let result = distribution.sample(&mut rng, 1);
        assert!(result.is_some());
        let samples = result.unwrap();
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 1.0).abs() < 1e-10); // Should be exactly 1.0
    }

    #[test]
    fn test_sample_dirichlet_multiple() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut distribution = Dirichlet::new(1.0, 5);
        let result = distribution.sample(&mut rng, 5);
        assert!(result.is_some());
        let samples = result.unwrap();
        assert_eq!(samples.len(), 5);

        // Check that all samples are between 0 and 1
        for &sample in samples {
            assert!((0.0..=1.0).contains(&sample));
        }

        // Check that the sum is approximately 1
        let sum: f64 = samples.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_dirichlet_distribution() {
        // Run multiple times to check consistency
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..10 {
            let mut distribution = Dirichlet::new(0.3, 3);
            let result = distribution.sample(&mut rng, 3);
            assert!(result.is_some());
            let samples = result.unwrap();
            assert_eq!(samples.len(), 3);

            let sum: f64 = samples.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }
}
