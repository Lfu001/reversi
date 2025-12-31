use rand_distr::{Distribution, Gamma};

/// Dirichlet distribution using the Gamma-based method.
///
/// When `X_i ~ Gamma(α_i, 1)`, then `Y_i = X_i / Σ(X_j)` follows Dirichlet(α_1, ..., α_n).
pub struct Dirichlet {
    /// Gamma distribution with alpha parameter.
    gamma: Gamma<f64>,
}

impl Dirichlet {
    /// Creates a new [`Dirichlet`] distribution with the given `alpha` parameter.
    pub fn new(alpha: f64) -> Self {
        Self {
            gamma: Gamma::new(alpha, 1.0).unwrap(),
        }
    }

    /// Samples from the Dirichlet distribution.
    ///
    /// Returns `None` if `size` is 0 or if the sum of samples is 0.
    pub fn sample(&self, rng: &mut impl rand::Rng, size: usize) -> Option<Vec<f64>> {
        if size == 0 {
            return Some(vec![]);
        }

        let mut samples: Vec<f64> = (0..size).map(|_| self.gamma.sample(rng)).collect();
        let sum: f64 = samples.iter().sum();

        if sum > 0.0 {
            for sample in samples.iter_mut() {
                *sample /= sum;
            }
            Some(samples)
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
        let distribution = Dirichlet::new(1.0);
        let result = distribution.sample(&mut rng, 0);
        assert_eq!(result, Some(vec![]));
    }

    #[test]
    fn test_sample_dirichlet_single() {
        let mut rng = StdRng::seed_from_u64(42);
        let distribution = Dirichlet::new(1.0);
        let result = distribution.sample(&mut rng, 1);
        assert!(result.is_some());
        let samples = result.unwrap();
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 1.0).abs() < 1e-10); // Should be exactly 1.0
    }

    #[test]
    fn test_sample_dirichlet_multiple() {
        let mut rng = StdRng::seed_from_u64(42);
        let distribution = Dirichlet::new(1.0);
        let result = distribution.sample(&mut rng, 5);
        assert!(result.is_some());
        let samples = result.unwrap();
        assert_eq!(samples.len(), 5);

        // Check that all samples are between 0 and 1
        for sample in &samples {
            assert!((0.0..=1.0).contains(sample));
        }

        // Check that the sum is approximately 1
        let sum: f64 = samples.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_dirichlet_distribution() {
        // Run multiple times to check consistency
        let mut rng = StdRng::seed_from_u64(42);
        let distribution = Dirichlet::new(0.3);
        for _ in 0..10 {
            let result = distribution.sample(&mut rng, 3);
            assert!(result.is_some());
            let samples = result.unwrap();
            assert_eq!(samples.len(), 3);

            let sum: f64 = samples.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }
}
