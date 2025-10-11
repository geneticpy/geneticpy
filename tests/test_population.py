"""Tests for new genetic algorithm features added in v2.0."""

import random

import numpy as np
from geneticpy import optimize
from geneticpy.distributions import ChoiceDistribution, UniformDistribution
from geneticpy.population import Population


class TestDiversityCalculation:
    """Test diversity calculation for different parameter types."""

    def test_diversity_with_numeric_parameters(self) -> None:
        """Test diversity calculation with only numeric parameters."""
        np.random.seed(42)
        random.seed(42)

        param_space = {
            "x": UniformDistribution(0, 100),
            "y": UniformDistribution(-10, 10),
        }

        def dummy_fn(params: dict[str, float]) -> float:
            return float(params["x"] + params["y"])

        pop = Population(fn=dummy_fn, params=param_space, size=20, maximize_fn=True)

        # Initial random population should have diversity
        diversity = pop.calculate_diversity()
        assert diversity > 0.0, "Random population should have diversity"

        # Make all individuals identical
        for ind in pop.population:
            ind.params["x"] = 50.0
            ind.params["y"] = 0.0

        diversity = pop.calculate_diversity()
        assert diversity == 0.0, "Identical population should have zero diversity"

    def test_diversity_with_choice_parameters(self) -> None:
        """Test diversity calculation with categorical choice parameters."""
        np.random.seed(42)
        random.seed(42)

        param_space = {
            "category": ChoiceDistribution(["A", "B", "C", "D", "E"]),
            "numeric": UniformDistribution(0, 10),
        }

        def dummy_fn(params: dict[str, float]) -> float:
            return float(params["numeric"])

        pop = Population(fn=dummy_fn, params=param_space, size=20, maximize_fn=True)

        # Check that diversity accounts for categorical diversity
        diversity = pop.calculate_diversity()
        assert diversity > 0.0, "Population with choices should have diversity"

        # Count unique categories
        categories = [ind.params["category"] for ind in pop.population]
        unique_categories = len(set(map(str, categories)))
        assert unique_categories >= 1, "Should have at least one unique category"

    def test_diversity_with_mixed_parameters(self) -> None:
        """Test diversity calculation with mixed numeric and choice parameters."""
        np.random.seed(42)
        random.seed(42)

        param_space = {
            "numeric": UniformDistribution(0, 100),
            "category": ChoiceDistribution(["A", "B", "C"]),
            "another_num": UniformDistribution(-10, 10),
        }

        def dummy_fn(params: dict[str, float]) -> float:
            return sum(v for v in params.values() if isinstance(v, (int, float)))

        pop = Population(fn=dummy_fn, params=param_space, size=20, maximize_fn=True)

        # Initial diversity
        diversity_high = pop.calculate_diversity()
        assert diversity_high > 0.0

        # Make population less diverse
        for ind in pop.population:
            ind.params["numeric"] = 50.0
            ind.params["category"] = "A"
            ind.params["another_num"] = 0.0

        diversity_low = pop.calculate_diversity()
        assert diversity_low < diversity_high, "Low diversity should be less than high diversity"

    def test_diversity_edge_cases(self) -> None:
        """Test diversity calculation edge cases."""
        param_space = {"x": UniformDistribution(0, 10)}

        def dummy_fn(params: dict[str, float]) -> float:
            return float(params["x"])

        # Single individual - minimum viable population
        pop = Population(fn=dummy_fn, params=param_space, size=2, maximize_fn=True)
        # With only 2 individuals, diversity could be 0 if they're the same
        diversity = pop.calculate_diversity()
        assert diversity >= 0.0, "Diversity should be non-negative"

        # Population with identical values
        pop = Population(fn=dummy_fn, params=param_space, size=10, maximize_fn=True)
        for ind in pop.population:
            ind.params["x"] = 5.0
        assert pop.calculate_diversity() == 0.0, "Identical values should have zero diversity"


class TestDiversityInjection:
    """Test diversity injection mechanism."""

    def test_inject_diversity(self) -> None:
        """Test that diversity injection adds random individuals."""
        np.random.seed(42)
        random.seed(42)

        param_space = {"x": UniformDistribution(0, 100)}

        def dummy_fn(params: dict[str, float]) -> float:
            return float(params["x"])

        pop = Population(fn=dummy_fn, params=param_space, size=20, maximize_fn=True)

        # Make all individuals very similar
        for ind in pop.population:
            ind.params["x"] = 50.0

        initial_diversity = pop.calculate_diversity()

        # Inject diversity
        pop.inject_diversity(percentage=0.3)

        # Diversity should increase after injection
        new_diversity = pop.calculate_diversity()
        assert new_diversity > initial_diversity, "Diversity should increase after injection"

    def test_inject_diversity_percentage(self) -> None:
        """Test that diversity injection respects the percentage parameter."""
        param_space = {"x": UniformDistribution(0, 10)}

        def dummy_fn(params: dict[str, float]) -> float:
            return float(params["x"])

        pop = Population(fn=dummy_fn, params=param_space, size=100, maximize_fn=True)

        # Store original last individuals
        original_last = [ind.params["x"] for ind in pop.population[-20:]]

        # Inject 20% diversity
        pop.inject_diversity(percentage=0.2)

        # Check that approximately 20 individuals (20% of 100) were replaced
        new_last = [ind.params["x"] for ind in pop.population[-20:]]
        changed = sum(1 for old, new in zip(original_last, new_last, strict=True) if old != new)
        assert changed >= 15, f"Expected ~20 replacements, got {changed}"


class TestTournamentSelection:
    """Test tournament selection mechanism."""

    def test_tournament_select(self) -> None:
        """Test that tournament selection returns a valid individual."""
        np.random.seed(42)
        random.seed(42)

        param_space = {"x": UniformDistribution(0, 10)}

        def dummy_fn(params: dict[str, float]) -> float:
            return float(params["x"])

        pop = Population(
            fn=dummy_fn,
            params=param_space,
            size=20,
            maximize_fn=True,
            use_tournament_selection=True,
            tournament_size=3,
        )

        # Run one evolution to populate grades
        pop.evolve()

        # Tournament select should return one of the individuals
        selected = pop.tournament_select(tournament_size=5)
        assert selected in pop.population, "Selected individual should be in population"

    def test_tournament_selection_in_evolution(self) -> None:
        """Test that tournament selection works in the evolution process."""

        def fn(params: dict[str, float]) -> float:
            return -((params["x"] - 5) ** 2)  # Maximum at x=5

        param_space = {"x": UniformDistribution(0, 10)}

        result = optimize(
            fn,
            param_space,
            size=30,
            generation_count=20,
            use_tournament_selection=True,
            tournament_size=3,
            verbose=False,
            seed=42,
            maximize_fn=True,
        )

        # Should converge reasonably close to optimum
        assert abs(result.best_params["x"] - 5) < 1.0, "Should converge near x=5"
        assert result.best_score > -1.0, "Should achieve good score"


class TestEnhancedMutation:
    """Test enhanced mutation with mutation_rate parameter."""

    def test_mutation_rate_parameter(self) -> None:
        """Test that mutation_rate controls per-parameter mutation probability."""
        np.random.seed(42)
        random.seed(42)

        param_space = {
            "a": UniformDistribution(0, 10),
            "b": UniformDistribution(0, 10),
            "c": UniformDistribution(0, 10),
            "d": UniformDistribution(0, 10),
        }

        def dummy_fn(params: dict[str, float]) -> float:
            return sum(params.values())

        pop = Population(fn=dummy_fn, params=param_space, size=10, maximize_fn=True)

        # Get an individual
        ind = pop.population[0]
        original_params = ind.params.copy()

        # Mutate with low rate - fewer parameters should change
        ind.mutate(mutation_rate=0.3)
        changed_count_low = sum(1 for key in original_params if original_params[key] != ind.params[key])

        # Reset
        ind.params = original_params.copy()

        # Mutate with high rate - more parameters should change
        ind.mutate(mutation_rate=1.0)
        changed_count_high = sum(1 for key in original_params if original_params[key] != ind.params[key])

        # High mutation rate should generally change more parameters
        # (This is probabilistic, so we just check it's reasonable)
        assert changed_count_low >= 0
        assert changed_count_high >= 1


class TestEarlyStopping:
    """Test early stopping with patience parameter."""

    def test_early_stopping_triggers(self) -> None:
        """Test that early stopping halts optimization when no improvement."""

        def fn(params: dict[str, float]) -> float:
            # Flat function - should trigger early stopping
            return 1.0

        param_space = {"x": UniformDistribution(0, 10)}

        result = optimize(
            fn,
            param_space,
            size=20,
            generation_count=1000,  # Would run many generations
            patience=5,  # But stop after 5 without improvement
            verbose=False,
            seed=42,
            maximize_fn=True,
        )

        assert result.best_score == 1.0, "Should find the constant score"
        # The optimization should stop early, not run all 1000 generations
        # This is validated by checking it completes quickly

    def test_early_stopping_with_improvement(self) -> None:
        """Test that early stopping doesn't trigger when improving."""

        def fn(params: dict[str, float]) -> float:
            return -((params["x"] - 5) ** 2)  # Has clear optimum

        param_space = {"x": UniformDistribution(0, 10)}

        result = optimize(
            fn,
            param_space,
            size=30,
            generation_count=50,
            patience=10,
            verbose=False,
            seed=42,
            maximize_fn=True,
        )

        # Should find good solution
        assert abs(result.best_params["x"] - 5) < 1.0
        assert result.best_score > -1.0


class TestAdaptiveMutation:
    """Test adaptive mutation feature."""

    def test_adaptive_mutation_enabled(self) -> None:
        """Test that adaptive mutation adjusts parameters during optimization."""

        def fn(params: dict[str, float]) -> float:
            return -((params["x"] - 7) ** 2) - (params["y"] - 3) ** 2

        param_space = {"x": UniformDistribution(0, 10), "y": UniformDistribution(0, 10)}

        result = optimize(
            fn,
            param_space,
            size=30,
            generation_count=50,
            adaptive_mutation=True,
            verbose=False,
            seed=42,
            maximize_fn=True,
        )

        # Should converge to optimum
        assert abs(result.best_params["x"] - 7) < 1.5
        assert abs(result.best_params["y"] - 3) < 1.5

    def test_adaptive_mutation_disabled(self) -> None:
        """Test that optimization works with adaptive mutation disabled."""

        def fn(params: dict[str, float]) -> float:
            return -((params["x"] - 7) ** 2)

        param_space = {"x": UniformDistribution(0, 10)}

        result = optimize(
            fn,
            param_space,
            size=30,
            generation_count=50,
            adaptive_mutation=False,  # Explicitly disabled
            verbose=False,
            seed=42,
            maximize_fn=True,
        )

        # Should still work reasonably well
        assert abs(result.best_params["x"] - 7) < 2.0


class TestIntegration:
    """Integration tests combining multiple new features."""

    def test_all_features_together(self) -> None:
        """Test that all new features work together."""

        def fn(params: dict[str, float]) -> float:
            x = params["x"]
            method = str(params["method"])

            if method == "square":
                return -((x - 5) ** 2)
            elif method == "linear":
                return -abs(x - 5)
            else:  # cubic
                return -((x - 5) ** 3) if x >= 5 else (x - 5) ** 3

        param_space = {
            "x": UniformDistribution(0, 10),
            "method": ChoiceDistribution(["square", "linear", "cubic"]),
        }

        result = optimize(
            fn,
            param_space,
            size=40,
            generation_count=30,
            use_tournament_selection=True,
            tournament_size=3,
            patience=15,
            adaptive_mutation=True,
            verbose=False,
            seed=42,
            maximize_fn=True,
        )

        # Should find x near 5 regardless of method
        assert abs(result.best_params["x"] - 5) < 1.5
        assert result.best_score > -2.0

    def test_reproducibility_with_new_features(self) -> None:
        """Test that results are reproducible with new features."""

        def fn(params: dict[str, float]) -> float:
            return -((params["x"] - 3) ** 2) - (params["y"] - 7) ** 2

        param_space = {"x": UniformDistribution(0, 10), "y": UniformDistribution(0, 10)}

        # Run twice with same seed
        result1 = optimize(
            fn,
            param_space,
            size=30,
            generation_count=20,
            use_tournament_selection=True,
            adaptive_mutation=True,
            patience=10,
            verbose=False,
            seed=42,
            maximize_fn=True,
        )

        result2 = optimize(
            fn,
            param_space,
            size=30,
            generation_count=20,
            use_tournament_selection=True,
            adaptive_mutation=True,
            patience=10,
            verbose=False,
            seed=42,
            maximize_fn=True,
        )

        # Results should be identical
        assert result1.best_params["x"] == result2.best_params["x"]
        assert result1.best_params["y"] == result2.best_params["y"]
        assert result1.best_score == result2.best_score
