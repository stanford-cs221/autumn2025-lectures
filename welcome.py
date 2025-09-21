from edtrace import text, image, link, note

def main():
    text("## Welcome to CS221 (Artificial Intelligence: Principles and Techniques)")
    image("images/course-staff.png", width=600)

    what_is_ai()
    what_is_this_program()  # Interlude
    course_philosophy()
    course_logistics()


def what_is_ai():
    text("We have lots of examples of AI")
    text("- AI assistants (ChatGPT, Claude, Gemini, Grok)")
    text("- Autonomous vehicles (Waymo, Wayve)")
    text("- Game playing (Deep Blue, AlphaGo, AlphaStar)")
    text("- 3D protein structure prediction (AlphaFold)")

    text("But what is AI?")

    text("Artificial intelligence")
    text("- artificial: runs on a computer (or robot)")
    text("- intelligence: ???")

    text("We could define **intelligence** in terms of humans...")
    text("...but we seek a definition from general principles.")

    text("## Ingredients of intelligence")

    text("**Perceive**: process raw inputs from the world")
    text("- visual scene understanding")
    text("- speech recognition")
    text("- natural language understanding")

    text("**Reason**: use knowledge + percepts to draw inferences about the world")
    text("- uniform cost search (in deterministic world)")
    text("- value iteration (decision-making under uncertainty)")
    text("- minimax (for adversial games)")
    text("- probabilistic inference (in Bayesian networks)")

    text("**Act**: do stuff to the world")
    text("- Text/image generation")
    text("- Speech synthesis")
    text("- Robot manipulation")

    text("**Learn**: update state based on experience")
    text("- Gradient descent")
    text("- Q-learning (reinforcement learning)")
    text("- Expectation Maximization (for Bayesian networks)")

    text("...all under **resource constraints**")
    text("- Computation: running time of algorithm, memory, communication")
    text("- Information: data, inputs available")

    text("So what does it do?")
    text("- AI implicitly or explicitly encodes goals / objectives / values / utility functions")
    text("- Alignment: how does these values correspond to what humans want?")
    text("- This is a deep, sociotechnical problem")

    text("## Summary")
    text("- Ingredients of intelligence: perception, reasoning, action, learning")
    text("- Resource constraints: computation, information")
    text("- Alignment: important for AI to actually benefit society")


def course_philosophy():
    text("Course philosophy")
    text("- Timeless foundations (e.g., gradient descent)")
    text("- Modern examples (e.g., GPT-5 solving 12/12 problems at the ICPC)")
    text("- Code-driven: most precise, need it anyway at the end of the day")

    text("Changes this year:")
    text("- Tensor-native: from deep learning to value iteration to Bayesian network inference")
    text("- Cut constraint satisfaction problems :(")
    text("- Deeper dive into societal issues (e.g., copyright)")


def what_is_this_program():
    text("This is an *executable lecture*, a program whose execution delivers the content of a lecture.")
    text("Executable lectures make it possible to:")
    text("- view and run code (since everything is code!),")
    total = 0  # @inspect total
    for x in [1, 2, 3]:  # @inspect x
        total += x  # @inspect total
    text("- see the hierarchical structure of the lecture, and")
    text("- jump to definitions and concepts: "), link(what_is_ai)


def course_logistics():
    text("All information online: "), link("https://stanford-cs221.github.io/autumn2025/")

    text("## Lectures")
    text("## Assignments")
    text("## Exam")
    text("## Project (optional)")

    text("## Office hours")
    text("## Honor code")


if __name__ == "__main__":
    main()
