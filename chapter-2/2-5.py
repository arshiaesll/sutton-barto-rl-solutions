from scipy.stats import norm
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class StepType(Enum):
    AVG = 0
    CONSTANT = 1


def run_simulation(
    num_bandits: int = 10,
    runs: int = 2000,
    steps: int = 1000,
    seed: int = 1000,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    step_type: StepType = StepType.AVG,
    changing_env: bool = False,
):

    # Need to get runs number of starting true means
    rgn = np.random.default_rng(seed)
    percent_progress = []
    true_means = rgn.normal(0, 1, size=(runs, num_bandits))
    step_counts = np.zeros(shape=(runs, num_bandits))
    q_estimates = np.zeros(shape=(runs, num_bandits))

    for _ in tqdm(range(steps)):
        # If that step is going to be random or not
        # 1 indicate a random action
        if changing_env:
            rand_change = rgn.normal(0, 0.01, size=(runs, num_bandits))
            true_means += rand_change

        epsilon_mask = rgn.random(runs) < epsilon
        explore_actions = rgn.integers(0, num_bandits, size=runs)
        greedy_actions = np.argmax(q_estimates, axis=1)

        # Masking to pick the explore actions with 1 and 0 for greedy actions
        chosen_actions = np.where(epsilon_mask, explore_actions, greedy_actions)
        chosen_means = true_means[np.arange(runs), chosen_actions]
        rewards = rgn.normal(loc=chosen_means, scale=1, size=runs)
        idx = np.arange(runs)

        if step_type == StepType.AVG:
            step_counts[idx, chosen_actions] += 1
            q_estimates[idx, chosen_actions] += (
                rewards - q_estimates[idx, chosen_actions]
            ) / step_counts[idx, chosen_actions]

        if step_type == StepType.CONSTANT:
            q_estimates[idx, chosen_actions] += alpha * (
                rewards - q_estimates[idx, chosen_actions]
            )

        optimal_actions = np.argmax(true_means, axis=1)
        percent_optimal = (optimal_actions == chosen_actions).mean() * 100
        percent_progress.append(percent_optimal)

    return percent_progress


def plot_progress(
    progress_list,
    labels=None,
    title="Average % Optimal Action Over Time",
    save_path="fig.pdf",
):
    """
    progress_list : list of 1D numpy arrays (each length = steps)
    labels        : list of strings (optional)
    """

    plt.figure(figsize=(8, 5))

    # Default labels if none provided
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(progress_list))]

    for progress, label in zip(progress_list, labels):
        steps = len(progress)
        plt.plot(np.arange(1, steps + 1), progress, label=label)

    plt.xlabel("Steps")
    plt.ylabel("Percentage of Optimal Actions")
    plt.title(title)

    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_epsilon_vs_performance(
    epsilons, performances, title="Average Performance vs Epsilon", save_path=None
):

    import numpy as np
    import matplotlib.pyplot as plt

    epsilons = np.array(epsilons)
    performances = np.array(performances)

    plt.figure(figsize=(7, 5))
    plt.plot(epsilons, performances, marker="o")

    plt.xscale("log")

    # Optional: show ticks exactly at your epsilon values
    plt.xticks(epsilons, [str(eps) for eps in epsilons])

    plt.xlabel("Epsilon (ε)")
    plt.ylabel("Average % Optimal Action")
    plt.title(title)

    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    plt.show()
    plt.close()


def main():

    epsilons = [1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4]
    performances = []
    for item in epsilons:
        performances.append(
            run_simulation(
                steps=100000,
                changing_env=True,
                epsilon=item,
                step_type=StepType.CONSTANT,
            )
        )
    performances = np.array(performances)
    half = performances.shape[1] // 2
    performances_last_half = performances[:, half:]
    performances_last_half = np.mean(performances_last_half, axis=1)
    plot_epsilon_vs_performance(
        epsilons=epsilons,
        performances=performances_last_half,
        save_path="non_stationary_epsilon_ablation.pdf",
    )


if __name__ == "__main__":
    main()
