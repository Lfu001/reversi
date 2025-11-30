use rand_distr::{Distribution, Gamma};

/// Dirichletディストリビューションからサンプリング
///
/// Dirichletディストリビューションは、Gammaディストリビューションを使って実装できます：
/// X_i ~ Gamma(α_i, 1) のとき、Y_i = X_i / Σ(X_j) は Dirichlet(α_1, ..., α_n) に従う
///
/// # Arguments
/// * `rng` - 乱数生成器への可変参照
/// * `alpha` - Dirichletディストリビューションのαパラメータ（各次元で同じ値を使用）
/// * `size` - サンプルのサイズ（次元数）
///
/// # Returns
/// `Some(Vec<f64>)` - 正規化されたサンプル（合計が1になる）
/// `None` - サンプリングに失敗した場合
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
            assert!(sample >= 0.0 && sample <= 1.0);
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
