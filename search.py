from re import T
from edtrace import text, link, image, note
from typing import Any
from dataclasses import dataclass, field
from graphviz import Digraph
import random


def main():
    text("Last week: **machine learning**")
    text("- Learning algorithm: training data {(input, output)} → predictor")
    text("- Predictor: input → output (regression: number or classification: class)")

    image("images/perceive-reason-act-learn.png", width=400)
    text("A predictor is reflexive directly maps percepts to actions.")
    text("Many problems in the real world require reasoning (thinking, problem solving, planning).")

    text("This week: **search** (one form of reasoning, when the world is deterministic)")
    text("Example applications:")
    image("images/rubiks-cube.jpg", width=200)
    image("images/maps.png", width=200)

    text("Recall: (symbolic) AI started in the 1950s with search")
    text("Is this still relevant today?")

    text("Rich Sutton's *The Bitter Lesson* essay (2019)"), link("http://www.incompleteideas.net/IncIdeas/BitterLesson.html", title="[article]")
    text("- *...general methods that leverage computation are ultimately the most effective, and by a large margin.*")
    text("- *The two methods that seem to scale arbitrarily in this way are **search and learning**.")

    text("Search is increasingly important (e.g., test-time compute in language models)!")

    search_problem()
    introduce_backtracking_search()
    introduce_best_of_n()
    introduce_beam_search()
    searching_language_models()
    introduce_dynamic_programming()

    text("Summary")
    text("- Search problem: formally defines the problem (state, actions, costs, etc.)")
    text("- Objective function: find a solution (sequence of actions) that minimizes the total cost.")
    text("- Backtracking search: find exact solution, but takes exponential time.")
    text("- Best-of-n search, beam search: find approximate solution with a time/tradeoff knob.")
    text("- Dynamic programming: find exact solution, if the number of states is small.")


def search_problem():
    text("Let us formalize search problems using an abstraction called a **search problem**.")
    example1()
    example2()


def example1():
    text("Example problem:")
    text("- Street with blocks numbered to 1 to n.")
    text("- Walking from i to i+1 takes 1 minute.")
    text("- Taking a magic tram from i to 2*i takes 2 minutes.")
    text("- How to travel from 1 to n in the least time?")

    text("Mindset: capture the problem formally before solving it.")
    text("...because we are interested in general methods that can solve **any** search problem.")

    problem = TravelSearchProblem(num_locs=10)
    state = problem.start_state()  # @inspect state
    successors = problem.successors(state)  # @inspect successors
    is_end = problem.is_end(successors[0].state)  # @inspect is_end

    text("A search problem has the following components:")  # @clear 
    text("- `start_state`: the initial state.")
    text("- `successors`: specifies the actions one can take in a state, their costs, and the resulting states.")
    text("- `is_end(state)`: whether `state` is an end state.")

    text("Objective function: find a solution (**sequence of actions**) that minimizes the total cost.")
    solution = Solution(successors=[
        Successor(action="walk", cost=1, state=2),
        Successor(action="tram", cost=2, state=4),
        Successor(action="walk", cost=1, state=5),
        Successor(action="tram", cost=2, state=10),
    ])
    cost = solution.cost  # @inspect cost


@dataclass(frozen=True)
class Successor:
    """Represents taking an `action`, incurring some `cost` and ending up in a new `state`."""
    action: Any
    cost: float
    state: Any


class SearchProblem:
    """Formally and fully represents a search problem."""
    def start_state(self) -> Any:
        raise NotImplementedError

    def successors(self, state: Any) -> list[Successor]:
        raise NotImplementedError

    def is_end(self, state: Any) -> bool:
        raise NotImplementedError


class TravelSearchProblem(SearchProblem):
    """An instance of a `SearchProblem` where you try to go from 1 to n in the least time."""
    def __init__(self, num_locs: int):
        self.num_locs = num_locs

    def start_state(self) -> int:
        return 1

    def successors(self, state: int) -> list[Successor]:
        successors = []
        if state + 1 <= self.num_locs:
            successors.append(Successor(action="walk", cost=1, state=state + 1))
        if 2 * state <= self.num_locs:
            successors.append(Successor(action="tram", cost=2, state=2 * state))
        return successors

    def is_end(self, state: int) -> bool:
        return state == self.num_locs


def example2():
    text("The state can be more complex.")
    text("Suppose we can only take the magic tram a restricted number of times.")
    text("Then we need to somehow track that in the state.")
    problem = LimitedTravelSearchProblem(num_locs=10, starting_tickets=1)
    state = problem.start_state()  # @inspect state
    successors = problem.successors(state)  # @inspect successors
    is_end = problem.is_end(successors[0].state)  # @inspect is_end

    text("In general, the **state** contains any information that's needed to evaluate actions, costs, and successors.")

    text("So far, we have focused on the **modeling** (representing the problem formally).")
    text("How do we actually **solve** the problem?")


@dataclass(frozen=True)
class TravelState:
    """Represents the state of the `LimitedTravelSearchProblem`, where you are at `loc` and have `tickets` left."""
    loc: int
    tickets: int


class LimitedTravelSearchProblem(SearchProblem):
    def __init__(self, num_locs: int, starting_tickets: int):
        self.num_locs = num_locs
        self.starting_tickets = starting_tickets

    def start_state(self) -> TravelState:
        return TravelState(loc=1, tickets=self.starting_tickets)

    def successors(self, state: TravelState) -> list[Successor]:
        successors = []
        if state.loc + 1 <= self.num_locs:
            successors.append(Successor(action="walk", cost=1, state=TravelState(loc=state.loc + 1, tickets=state.tickets)))
        if state.tickets > 0 and 2 * state.loc <= self.num_locs:
            successors.append(Successor(action="tram", cost=2, state=TravelState(loc=2 * state.loc, tickets=state.tickets - 1)))
        return successors

    def is_end(self, state: TravelState) -> bool:
        return state.loc == self.num_locs
    

def introduce_backtracking_search():
    text("Objective: given a search problem, find a sequence of actions that minimizes the total cost.")
    
    text("Backtracking search: simplest way to try all possible sequences of actions.")
    problem = TravelSearchProblem(num_locs=5)  # @stepover
    solution, num_explored = backtracking_search(problem)  # @inspect solution solution.cost num_explored

    text("Summary:")
    text("- Depth-first search (DFS) on the search tree")
    text("- Explore all actions")
    text("- When reach end state, update the best solution so far")

    text("Let's try some larger problems.")

    problem = TravelSearchProblem(num_locs=10)  # @stepover
    solution, num_explored = backtracking_search(problem)  # @stepover @inspect solution solution.cost num_explored

    problem = TravelSearchProblem(num_locs=17)  # @stepover
    solution, num_explored = backtracking_search(problem)  # @stepover @inspect solution solution.cost num_explored

    text("This is getting pretty slow...")
    text("In general, backtracking search is **exponential** in the number of actions.")


@dataclass(frozen=True)
class Solution:
    """Represents a solution to a search problem (sequence of actions that produces a cost)."""
    successors: list[Successor]

    @property
    def cost(self) -> float:
        return sum(successor.cost for successor in self.successors)


def backtracking_search(problem: SearchProblem) -> tuple[Solution | None, int]:
    """
    Perform backtracking search to `problem`.
    Return the best solution and the number of states explored.
    """
    best_solution: Solution | None = None
    num_explored = 0

    def backtrack(state, history: list[Successor]):  # @inspect best_solution history state
        # Allow us to update from inside the function.
        nonlocal best_solution
        nonlocal num_explored

        num_explored += 1

        # Check if we've reached an end state (a valid solution)
        if problem.is_end(state):  # @stepover
            # See if it's better than the best solution we've found so far
            solution = Solution(successors=history)  # @inspect solution.cost
            if best_solution is None or solution.cost < best_solution.cost:  # @inspect best_solution best_solution.cost @stepover
                best_solution = solution  # @inspect best_solution best_solution.cost
            return

        # Explore each successor
        for successor in problem.successors(state): # @stepover @inspect successor
            backtrack(successor.state, history + [successor])

    backtrack(problem.start_state(), history=[])  # @inspect best_solution best_solution.cost
    return best_solution, num_explored


def introduce_best_of_n():
    text("Backtracking search is exact (guaranteed to find the minimum cost path).")
    text("But it is too slow...")
    text("To make it faster, we can heuristically look at only a subset of the actions.")

    text("The simplest idea is to randomly choose actions until we reach the end state.")
    text("We do this n times and take the best solution.")

    text("Let's take the example")
    problem = TravelSearchProblem(num_locs=10)  # @stepover

    text("How we choose actions is determined by a **policy**.")
    text("A **policy** is a function that maps state to action (can be non-deterministic).")
    random.seed(1)
    successor = random_policy(problem, problem.start_state())  # @inspect successor
    successor = random_policy(problem, problem.start_state())  # @inspect successor
    successor = random_policy(problem, problem.start_state())  # @inspect successor

    text("We can iterately apply a policy until we reach the end state to get a solution.")
    solution = rollout_policy(problem, random_policy)  # @inspect rollout_solution rollout_solution.cost

    text("Let's rollout the policy `n` times and take the best solution:")
    solution, num_explored = best_of_n(problem, random_policy, n=10)  # @stepover @inspect solution solution.cost num_explored

    text("Advantage: n paths can be explored in parallel (embarassingly parallel).")

    text("Disadvantage: we can waste time exploring bad paths...")


def best_of_n(problem: SearchProblem, policy, n: int) -> tuple[Solution | None, int]:
    """
    Perform best-of-n search to `problem`.
    Return the best solution and the number of states explored.
    """
    best_solution: Solution | None = None
    num_explored = 0
    for _ in range(n):
        solution = rollout_policy(problem, policy)  # @inspect solution solution.cost
        if best_solution is None or solution.cost < best_solution.cost:
            best_solution = solution
        num_explored += len(solution.successors)

    return best_solution, num_explored


def rollout_policy(problem: SearchProblem, policy) -> Solution:
    """Sample a policy from the start state of `problem`."""
    state = problem.start_state()  # @inspect state
    history = []  # @inspect history

    while not problem.is_end(state):
        successor = policy(problem, state)  # @inspect successor @stepover
        state = successor.state  # @inspect state
        history.append(successor)  # @inspect history

    return Solution(successors=history)


def random_policy(problem: SearchProblem, state: Any) -> Successor:
    """Randomly choose an action from the successors of `state`."""
    return random.choice(problem.successors(state))


def introduce_beam_search():
    text("Beam search: build a number of partial solutions in parallel.")
    
    image("images/beam_car.jpeg", width=200)
    text("Beam: the set of partial solutions at each step.")

    problem = TravelSearchProblem(num_locs=10)  # @stepover
    solution, num_explored = beam_search(problem, beam_width=2, max_steps=4)  # @stepover @inspect solution solution.cost num_explored


def beam_search(problem: SearchProblem, beam_width: int, max_steps: int) -> tuple[Solution | None, int]:
    """Perform beam search on `problem` keeping `beam_width` candidates and `max_steps`."""
    candidates = [Solution(successors=[])]
    for step in range(max_steps):  # @inspect step candidates
        # Given the existing candidates, expand them by one step
        new_candidates = []
        for candidate in candidates:
            state = candidate.successors[-1].state if candidate.successors else problem.start_state()
            if problem.is_end(state):  # If we've alreacy reached the end, just keep
                new_candidates.append(candidate)
            else:
                # Try all possible actions from `state`
                for successor in problem.successors(state):
                    new_candidates.append(Solution(successors=candidate.successors + [successor]))

        # Take the `beam_width` best candidates (lowest cost)
        new_candidates.sort(key=lambda x: x.cost)  # Sort
        candidates = new_candidates[:beam_width]  # Prune

    return candidates[0], len(candidates)
    

def searching_language_models():
    pass


def introduce_dynamic_programming():
    pass


if __name__ == "__main__":
    main()
