//! Node selection strategies for MCTS.

/// Configuration for PUCT (Polynomial Upper Confidence Trees) calculation.
#[derive(Debug, Clone, Copy)]
pub struct PuctConfig {
    /// Constant to adjust exploration strength (typically 1.0 to 2.0).
    pub c_puct: f64,
    /// Epsilon parameter for Dirichlet noise mixing at root node.
    pub dirichlet_epsilon: f64,
    /// Alpha parameter for Dirichlet noise distribution.
    pub dirichlet_alpha: f64,
}

/// PUCT (Polynomial Upper Confidence Trees) selection strategy.
pub struct PuctStrategy {
    config: PuctConfig,
}

impl PuctStrategy {
    /// Creates a new [`PuctStrategy`] with the given `config` parameters.
    pub fn new(config: PuctConfig) -> Self {
        Self { config }
    }

    /// Calculates the PUCT score for a child node.
    ///
    /// Formula: PUCT(s,a) = Q(s,a) + c_puct × P(s,a) × √(N(s)) / (1 + N(s,a))
    pub fn calculate_score(
        &self,
        q_value: f64,
        visit_count: u32,
        parent_visit_count: u32,
        prior_probability: f64,
    ) -> f64 {
        let exploration_term =
            self.config.c_puct * prior_probability * (parent_visit_count as f64).sqrt()
                / (1.0 + visit_count as f64);

        q_value + exploration_term
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_puct_score() {
        let config = PuctConfig {
            c_puct: 1.0,
            dirichlet_epsilon: 0.0,
            dirichlet_alpha: 1.0,
        };
        let strategy = PuctStrategy::new(config);

        // Q(s,a) = 0.5, N(s,a) = 10, N(s) = 100, P(s,a) = 0.3
        // exploration = 1.0 * 0.3 * √100 / (1 + 10) ≈ 0.2727
        // PUCT ≈ 0.7727
        let score = strategy.calculate_score(0.5, 10, 100, 0.3);
        assert!((score - 0.7727).abs() < 0.01);
    }
}
