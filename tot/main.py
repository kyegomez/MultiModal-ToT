import logging

from shapeless import liquid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@liquid
class MMTot:
    num_thoughts = 3
    max_steps: int = None
    value_of_threshold: float = None
    pruning_threshold = 0.5
    bestracking_threshold = 0.5
    initial_prompt = None
    openai_api_key = None

    def solve(self):
            try:
                self.dfs(self.initial_prompt, 1)

                if not self.output:
                    logger.error("No valid thoughts were generated during DFS")
                    return None
                
                best_state, _ = max(self.output, key=lambda x: x[1])
                solution = self.model.generate_solution(self.initial_prompt, best_state)
                print(f"Solution is {solution}")
                return solution if solution else best_state
            except Exception as error:
                logger.error(f"Error in tot_dfs: {error}")
                raise error

    def dfs(self, state, step):
        if step > self.max_steps:
            thought, value = self.evaluate_thought(state)
            self.output.append((thought, value))
            return

        thoughts = self.generate_and_filter_thoughts(state)
        for next_state in thoughts:
            state_value = self.evaluated_thoughts[next_state]
            if state_value > self.value_threshold:
                child = (state, next_state) if isinstance(state, str) else (*state, next_state)
                self.dfs(child, step + 1)

                #backtracking
                best_value = max([value for _, value in self.output])
                if best_value < self.backtracking_threshold:
                    self.output.pop()
                    continue

    def generate_and_filter_thoughts(self, state):
        thoughts = self.model.generate_thoughts(
            state, 
            self.num_thoughts, 
            self.initial_prompt
        )

        self.evaluated_thoughts = self.model.evaluate_states(
            thoughts, 
            self.initial_prompt
        )

        filtered_thoughts = [
            thought for thought in thoughts if self.evaluated_thoughts[thought] >= self.pruning_threshold
        ]

        print(f"filtered_thoughts: {filtered_thoughts}")
        return filtered_thoughts

    def evaluate_thought(self, state):
        thought = self.model.generate_thoughts(state, 1, self.initial_prompt)
        value = self.model.evaluate_states(
            [state], 
            self.initial_prompt
        )[state]
        print(f"Evaluated thought: {value}")
        return thought, value