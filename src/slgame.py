import numpy as np
import matplotlib.pyplot as plt


class SnakesLaddersGame:
    """Class for the Snakes and Ladders Game."""

    def __init__(self, steps, snakes, ladders):
        self.steps = steps
        self.snakes = snakes
        self.ladders = ladders

    @staticmethod
    def roll_a_die():
        """Perform a roll of a fair six-sided die and return the output."""
        return np.random.randint(1, 7)

    def get_next_state(self, state=1, prob_take_ladder=1, prob_take_snake=1):
        """Return the new state of the player and the type landed.
        Types: normal, snake and ladder."""
        die_output = self.roll_a_die()
        new_state = die_output + state

        type_landed = "normal"
        if new_state in self.ladders.keys():
            type_landed = "ladder"
            flag_take_ladder = np.random.choice(
                2, p=[1 - prob_take_ladder, prob_take_ladder]
            )
            if flag_take_ladder:
                new_state = self.ladders[new_state]
        elif new_state in self.snakes.keys():
            type_landed = "snake"
            if prob_take_snake:
                new_state_1 = self.snakes[new_state]

        return new_state, type_landed

    def run_2p_simulation(
        self,
        iterations=100,
        prob_take_ladder=1,
        p1_initial_state=1,
        p2_initial_state=1,
        p2_take_first_snake=1,
    ):
        """Run a 2-player simulation of N iterations and get the:
        1. probability that the player who starts the game wins,
        2. average number of ladders landed, and
        3. average number of snakes landed."""
        p1_wins = 0
        types_landed = {}

        for i in range(iterations):
            p1_state, p2_state = p1_initial_state, p2_initial_state
            max_state = 1

            prob_take_snake = p2_take_first_snake

            while max_state < self.steps:
                p1_state, p1_type_landed = self.get_next_state(
                    p1_state, prob_take_ladder
                )
                p2_state, p2_type_landed = self.get_next_state(
                    p2_state, prob_take_ladder, prob_take_snake
                )

                # Logic if the player 2 takes the first snake
                if (prob_take_snake == 0) & (p2_type_landed == "snake"):
                    prob_take_snake += 1
                    for i, j in self.snakes.items():
                        if j == p2_state:
                            p2_state = i
                            break

                max_state = max(p1_state, p2_state)

                types_landed[p1_type_landed] = types_landed.get(p1_type_landed, 0) + 1
                types_landed[p2_type_landed] = types_landed.get(p2_type_landed, 0) + 1

            if p1_state >= self.steps:
                p1_wins += 1

        return {
            "prob_p1_win": np.round(p1_wins / iterations, 4),
            "avg_ladders_landed": np.round(
                types_landed.get("ladder", 0) / iterations, 2
            ),
            "avg_snakes_landed": np.round(types_landed.get("snake", 0) / iterations, 2),
            "avg_rolls": np.round(np.sum(list(types_landed.values())) / iterations, 2),
        }

    def plot_simulations(self, iterations, warmup=0.3):
        """Plot the simulation results and get the odds that player 1 will win."""
        simulations = []
        for i in range(1, iterations + 1):
            simu = self.run_2p_simulation(i)
            simulations.append(simu["prob_p1_win"])

        avg_p1_prob = np.mean(simulations[int(iterations * warmup + 1) :])
        print("Average probability of player 1 winning:", np.round(avg_p1_prob, 4))

        fig, ax = plt.subplots()
        ax.plot(simulations)
        ax.set_xlabel("Number of iterations")
        ax.set_ylabel("Probability that the first player wins")
        plt.show()

    def find_fair_starting_state(self, iterations=100):
        """Find a fair starting state for the second player."""
        p1_probs_win = []
        for i in range(1, self.steps + 1):
            simu = self.run_2p_simulation(iterations=iterations, p2_initial_state=i)
            p1_probs_win.append(simu["prob_p1_win"])
        diff_probs = np.abs(np.array(p1_probs_win) - 0.5)

        return diff_probs.argmin() + 1, diff_probs.min() + 0.5
