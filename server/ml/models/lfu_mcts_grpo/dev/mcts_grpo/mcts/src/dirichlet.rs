use rand_distr::{Distribution, Gamma};

/// Samples from a Dirichlet distribution using the Gamma-based method.
///
/// The Dirichlet distribution can be implemented using Gamma distributions:
/// when X_i ~ Gamma(α_i, 1), then Y_i = X_i / Σ(X_j) follows Dirichlet(α_1, ..., α_n).
///
/// Takes a random number generator, an alpha parameter (same for all dimensions),
/// and the dimensionality of the sample. Returns a normalized probability
/// distribution (summing to 1.0) or `None` if sampling fails.
pub fn sample_dirichlet(rng: &mut impl rand::Rng, alpha: f64, size: usize) -> Option<Vec<f64>> {
    if size == 0 {
        return Some(Vec::new());
    }

    let gamma = Gamma::new(alpha, 1.0).ok()?;

    let samples: Vec<f64> = (0..size).map(|_| gamma.sample(rng)).collect();
    let sum: f64 = samples.iter().sum();

    if sum > 0.0 {
        Some(samples.into_iter().map(|x| x / sum).collect())
    } else {
        None
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
        let result = sample_dirichlet(&mut rng, 1.0, 0);
        assert_eq!(result, Some(Vec::new()));
    }

    #[test]
    fn test_sample_dirichlet_single() {
        let mut rng = StdRng::seed_from_u64(42);
        let result = sample_dirichlet(&mut rng, 1.0, 1);
        assert!(result.is_some());
        let samples = result.unwrap();
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 1.0).abs() < 1e-10); // Should be exactly 1.0
    }

    #[test]
    fn test_sample_dirichlet_multiple() {
        let mut rng = StdRng::seed_from_u64(42);
        let result = sample_dirichlet(&mut rng, 1.0, 5);
        assert!(result.is_some());
        let samples = result.unwrap();
        assert_eq!(samples.len(), 5);

        // Check that all samples are between 0 and 1
        for &sample in &samples {
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
            let result = sample_dirichlet(&mut rng, 0.3, 3);
            assert!(result.is_some());
            let samples = result.unwrap();
            assert_eq!(samples.len(), 3);

            let sum: f64 = samples.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }
}
