use crate::learning::parameters::{LearnableParameters, TopologicalCognitiveSignature};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Evolutionary optimization system for meta-parameter discovery
/// Replaces magic numbers with evolved, fitness-tested parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionaryOptimizer {
    /// Population of parameter sets
    pub population: Vec<EvolutionaryIndividual>,

    /// Current generation
    pub generation: usize,

    /// Fitness history tracking
    pub fitness_history: Vec<FitnessRecord>,

    /// Evolutionary hyperparameters
    pub evolution_config: EvolutionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionaryIndividual {
    /// Individual's parameter set
    pub parameters: LearnableParameters,

    /// Fitness score across multiple metrics
    pub fitness: FitnessScore,

    /// Individual ID for tracking
    pub id: usize,

    /// Mutation rate (can evolve)
    pub mutation_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessScore {
    /// Task performance (e.g., code analysis accuracy)
    pub task_performance: f32,

    /// Topological elegance (b0=1, low complexity)
    pub topological_elegance: f32,

    /// Cognitive efficiency (low knot complexity)
    pub cognitive_efficiency: f32,

    /// Combined fitness score
    pub combined: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessRecord {
    pub generation: usize,
    pub best_fitness: f32,
    pub average_fitness: f32,
    pub best_individual_id: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    /// Population size
    pub population_size: usize,

    /// Elite individuals to preserve
    pub elite_size: usize,

    /// Mutation rate bounds
    pub mutation_bounds: (f32, f32),

    /// Crossover probability
    pub crossover_rate: f32,

    /// Fitness weights
    pub fitness_weights: FitnessWeights,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessWeights {
    pub task_performance: f32,
    pub topological_elegance: f32,
    pub cognitive_efficiency: f32,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 20,
            elite_size: 4,
            mutation_bounds: (0.01, 0.2),
            crossover_rate: 0.7,
            fitness_weights: FitnessWeights {
                task_performance: 0.5,
                topological_elegance: 0.3,
                cognitive_efficiency: 0.2,
            },
        }
    }
}

impl EvolutionaryOptimizer {
    /// Create new evolutionary optimizer
    pub fn new(config: EvolutionConfig) -> Self {
        let population = LearnableParameters::create_initial_population(config.population_size)
            .into_iter()
            .enumerate()
            .map(|(id, params)| EvolutionaryIndividual {
                parameters: params,
                fitness: FitnessScore::default(),
                id,
                mutation_rate: 0.1,
            })
            .collect();

        Self {
            population,
            generation: 0,
            fitness_history: Vec::new(),
            evolution_config: config,
        }
    }

    /// Evaluate fitness of entire population
    pub fn evaluate_population(&mut self, task_data: &TaskEvaluationData) -> Result<()> {
        let mut fitness_scores = Vec::new();

        // Calculate fitness for each individual without borrowing issues
        for individual in &self.population {
            let fitness = self.evaluate_individual(&individual.parameters, task_data);
            fitness_scores.push(fitness);
        }

        // Apply fitness scores back to population
        for (i, fitness) in fitness_scores.into_iter().enumerate() {
            if let Some(individual) = self.population.get_mut(i) {
                individual.fitness = fitness;
            }
        }

        // Sort by fitness (best first)
        self.population.sort_by(|a, b| {
            b.fitness
                .combined
                .partial_cmp(&a.fitness.combined)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(())
    }

    /// Evaluate single individual's fitness
    fn evaluate_individual(
        &self,
        params: &LearnableParameters,
        task_data: &TaskEvaluationData,
    ) -> FitnessScore {
        // Task Performance: How well parameters work on the actual task
        let task_performance = self.evaluate_task_performance(params, task_data);

        // Topological Elegance: Based on emergent topology metrics
        let topological_elegance = self.evaluate_topological_elegance(params);

        // Cognitive Efficiency: Based on reasoning trajectory complexity
        let cognitive_efficiency = self.evaluate_cognitive_efficiency(params);

        // Combined fitness using weighted sum
        let combined = task_performance * self.evolution_config.fitness_weights.task_performance
            + topological_elegance * self.evolution_config.fitness_weights.topological_elegance
            + cognitive_efficiency * self.evolution_config.fitness_weights.cognitive_efficiency;

        FitnessScore {
            task_performance,
            topological_elegance,
            cognitive_efficiency,
            combined,
        }
    }

    /// Evaluate task performance (e.g., code analysis accuracy)
    fn evaluate_task_performance(
        &self,
        params: &LearnableParameters,
        _task_data: &TaskEvaluationData,
    ) -> f32 {
        // Simulate task performance based on parameters
        // In real implementation, this would run the actual task

        let base_performance = 0.5;

        // Emotional inertia affects consistency
        let inertia_factor = 1.0 - (params.cognitive_dynamics.emotional_inertia - 0.5).abs();

        // Exploration temperature affects discovery rate
        let exploration_factor = if params.cognitive_dynamics.exploration_temperature > 0.3
            && params.cognitive_dynamics.exploration_temperature < 0.8
        {
            1.0
        } else {
            0.7
        };

        // Memory parameters affect recall accuracy
        let memory_factor = if params.memory_parameters.consolidation_threshold > 0.7
            && params.memory_parameters.consolidation_threshold < 0.95
        {
            1.0
        } else {
            0.8
        };

        base_performance * inertia_factor * exploration_factor * memory_factor
    }

    /// Evaluate topological elegance (replaces Torus major radius: 5.0 etc.)
    fn evaluate_topological_elegance(&self, params: &LearnableParameters) -> f32 {
        // Elegance is based on how well parameters promote "good" topology

        let elegance_threshold = params.topology_thresholds.elegance_threshold;
        let complexity_penalty = params.topology_thresholds.complexity_penalty;

        // Prefer moderate elegance threshold (not too strict, not too loose)
        let threshold_score = 1.0 - (elegance_threshold - 1.5).abs() / 2.0;

        // Prefer lower complexity penalty (but not zero)
        let penalty_score = 1.0 - complexity_penalty;

        (threshold_score + penalty_score) / 2.0
    }

    /// Evaluate cognitive efficiency (replaces arbitrary cognitive constants)
    fn evaluate_cognitive_efficiency(&self, params: &LearnableParameters) -> f32 {
        // Efficiency based on cognitive dynamics parameters

        let emotional_inertia = params.cognitive_dynamics.emotional_inertia;
        let threat_threshold = params.cognitive_dynamics.threat_threshold;

        // Prefer balanced emotional inertia (not too rigid, not too chaotic)
        let inertia_score = 1.0 - (emotional_inertia - 0.6).abs();

        // Prefer appropriate threat threshold (sensitive but not paranoid)
        let threat_score = if threat_threshold > 0.02 && threat_threshold < 0.15 {
            1.0
        } else {
            0.5
        };

        (inertia_score + threat_score) / 2.0
    }

    /// Evolve to next generation
    pub fn evolve_generation(&mut self) -> Result<()> {
        let new_population = self.create_next_generation()?;
        self.population = new_population;
        self.generation += 1;

        Ok(())
    }

    /// Create next generation through selection, crossover, and mutation
    fn create_next_generation(&self) -> Result<Vec<EvolutionaryIndividual>> {
        let mut new_population = Vec::with_capacity(self.evolution_config.population_size);

        // Elitism: preserve best individuals
        for i in 0..self.evolution_config.elite_size.min(self.population.len()) {
            let mut elite = self.population[i].clone();
            elite.id = self.generation * 1000 + i; // New ID
            new_population.push(elite);
        }

        // Generate offspring through crossover and mutation
        while new_population.len() < self.evolution_config.population_size {
            let parent1 = self.tournament_selection();
            let parent2 = self.tournament_selection();

            let mut offspring = if rand::random::<f32>() < self.evolution_config.crossover_rate {
                self.crossover(&parent1, &parent2)?
            } else {
                parent1.clone()
            };

            self.mutate(&mut offspring);
            offspring.id = self.generation * 1000 + new_population.len();
            new_population.push(offspring);
        }

        Ok(new_population)
    }

    /// Tournament selection for parent selection
    fn tournament_selection(&self) -> &EvolutionaryIndividual {
        let tournament_size = 3;
        let mut best = &self.population[0];

        for _ in 0..tournament_size {
            let candidate = &self.population[rand::random::<usize>() % self.population.len()];
            if candidate.fitness.combined > best.fitness.combined {
                best = candidate;
            }
        }

        best
    }

    /// Crossover two parents to create offspring
    fn crossover(
        &self,
        parent1: &EvolutionaryIndividual,
        parent2: &EvolutionaryIndividual,
    ) -> Result<EvolutionaryIndividual> {
        let mut offspring_params = parent1.parameters.clone();

        // Simple parameter-wise crossover
        if rand::random() {
            offspring_params.cognitive_dynamics.emotional_inertia =
                parent2.parameters.cognitive_dynamics.emotional_inertia;
        }
        if rand::random() {
            offspring_params.topology_thresholds.elegance_threshold =
                parent2.parameters.topology_thresholds.elegance_threshold;
        }
        if rand::random() {
            offspring_params.evolutionary_genes.dominance_penalty =
                parent2.parameters.evolutionary_genes.dominance_penalty;
        }

        Ok(EvolutionaryIndividual {
            parameters: offspring_params,
            fitness: FitnessScore::default(),
            id: 0, // Will be set later
            mutation_rate: (parent1.mutation_rate + parent2.mutation_rate) / 2.0,
        })
    }

    /// Mutate individual parameters
    fn mutate(&self, individual: &mut EvolutionaryIndividual) {
        let mutation_strength = individual.mutation_rate;

        // Mutate emotional inertia
        if rand::random::<f32>() < 0.3 {
            individual.parameters.cognitive_dynamics.emotional_inertia +=
                (rand::random::<f32>() - 0.5) * mutation_strength;
            individual.parameters.cognitive_dynamics.emotional_inertia = individual
                .parameters
                .cognitive_dynamics
                .emotional_inertia
                .clamp(0.0, 1.0);
        }

        // Mutate topology thresholds
        if rand::random::<f32>() < 0.3 {
            individual.parameters.topology_thresholds.elegance_threshold +=
                (rand::random::<f32>() - 0.5) * mutation_strength;
            individual.parameters.topology_thresholds.elegance_threshold = individual
                .parameters
                .topology_thresholds
                .elegance_threshold
                .clamp(0.1, 5.0);
        }

        // Mutate evolutionary genes
        if rand::random::<f32>() < 0.3 {
            individual.parameters.evolutionary_genes.dominance_penalty +=
                (rand::random::<f32>() - 0.5) * mutation_strength;
            individual.parameters.evolutionary_genes.dominance_penalty = individual
                .parameters
                .evolutionary_genes
                .dominance_penalty
                .clamp(0.0, 1.0);
        }

        // Evolve mutation rate itself
        if rand::random::<f32>() < 0.1 {
            individual.mutation_rate += (rand::random::<f32>() - 0.5) * 0.02;
            individual.mutation_rate = individual.mutation_rate.clamp(
                self.evolution_config.mutation_bounds.0,
                self.evolution_config.mutation_bounds.1,
            );
        }
    }

    /// Get best individual from current population
    pub fn get_best_individual(&self) -> Option<&EvolutionaryIndividual> {
        self.population.first()
    }

    /// Record fitness history
    pub fn record_fitness(&mut self) {
        if let Some(best) = self.get_best_individual() {
            let average_fitness = self
                .population
                .iter()
                .map(|ind| ind.fitness.combined)
                .sum::<f32>()
                / self.population.len() as f32;

            self.fitness_history.push(FitnessRecord {
                generation: self.generation,
                best_fitness: best.fitness.combined,
                average_fitness,
                best_individual_id: best.id,
            });
        }
    }

    /// Check convergence criteria
    pub fn has_converged(&self) -> bool {
        if self.fitness_history.len() < 10 {
            return false;
        }

        // Check if fitness hasn't improved significantly in last 10 generations
        let recent_best: f32 = self
            .fitness_history
            .iter()
            .rev()
            .take(10)
            .map(|record| record.best_fitness)
            .sum::<f32>()
            / 10.0;

        let overall_best = self
            .fitness_history
            .last()
            .map(|record| record.best_fitness)
            .unwrap_or(0.0);

        (overall_best - recent_best).abs() < 0.001
    }
}

impl Default for FitnessScore {
    fn default() -> Self {
        Self {
            task_performance: 0.0,
            topological_elegance: 0.0,
            cognitive_efficiency: 0.0,
            combined: 0.0,
        }
    }
}

/// Data for task evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEvaluationData {
    /// Code analysis accuracy data
    pub analysis_results: Vec<(bool, bool)>, // (predicted, actual)

    /// Topological analysis results
    pub topology_samples: Vec<TopologicalCognitiveSignature>,

    /// Performance metrics
    pub performance_metrics: HashMap<String, f32>,
}

impl Default for TaskEvaluationData {
    fn default() -> Self {
        Self {
            analysis_results: vec![(true, true), (false, false), (true, false), (false, true)],
            topology_samples: vec![TopologicalCognitiveSignature::from_point_cloud(&[])],
            performance_metrics: HashMap::new(),
        }
    }
}
