from edtrace import text, image, link, note

def main():
    text("## Welcome to CS221 (Artificial Intelligence: Principles and Techniques)")
    image("images/course-staff.png", width=600)

    what_is_ai()
    about_this_course()
    what_is_this_program()


def what_is_ai():
    text("We have lots of examples of AI all around us:")
    text("- AI assistants: ChatGPT, Claude, Gemini, Grok")
    text("- Autonomous vehicles: Waymo, Wayve")
    text("- Game playing: [Deep Blue](https://en.wikipedia.org/wiki/Deep_Blue_(chess_computer)), [AlphaGo](https://en.wikipedia.org/wiki/AlphaGo), [AlphaStar](https://en.wikipedia.org/wiki/AlphaStar_(software))")
    text("- Competition math and programming: [IMO](https://x.com/OpenAI/status/1946594928945148246), [IOI](https://codeforces.com/blog/entry/134091), [ICPC](https://venturebeat.com/ai/google-and-openais-coding-wins-at-university-competition-show-enterprise-ai)")
    text("- 3D protein structure prediction: [AlphaFold](https://deepmind.google/science/alphafold/)")

    text("But what exactly **is** AI?")

    text("Artificial intelligence")
    text("- artificial: runs on a computer (or robot)")
    text("- intelligence: ???")

    text("We could define **intelligence** in terms of humans...")
    text("...but we seek a definition from general principles.")

    text("### Ingredients of intelligence")
    text("What kinds of things should an intelligent agent be able to do?")
    image("images/perceive-reason-act-learn.png", width=600)

    text("Motivating example: driving")
    image("images/self-driving-image.png", width=400)

    text("**Perceive**: process raw inputs from the world")
    text("- visual scene understanding")
    text("- speech recognition")
    text("- natural language understanding")

    text("**Reason**: use knowledge + percepts to draw inferences about the world")
    text("- uniform cost search (in a deterministic world)")
    text("- value iteration (decision-making under uncertainty)")
    text("- minimax (for adversarial games)")
    text("- probabilistic inference (in Bayesian networks)")

    text("**Act**: output actions that affect the world")
    text("- Text/image generation")
    text("- Speech synthesis")
    text("- Robot manipulation")

    text("**Learn**: update agent based on experience")
    text("- Gradient descent")
    text("- Q-learning (reinforcement learning)")
    text("- Expectation maximization (for Bayesian networks)")

    text("...all under **resource constraints**")
    text("- Computation: running time (also: memory, communication)")
    text("- Information: data / experience, available inputs in a given situation")

    text("### Goals")
    text("But what does the **developer** want the agent to achieve?")
    text("- An agent explicitly or implicitly encodes values / goals / objectives / utility functions")
    text("- Alignment: how do make these values correspond to what the developer wants?")
    text("- Example: ChatGPT aims to be informative, avoid hallucinations, refuse harmful queries")

    text("Also what impact **we** want the agent to have on society?")
    text("- Issues: privacy, copyright, jobs, inequality, geopolitics")
    text("- This is a deep, **sociotechnical** problem (who's we?)")
    text("- Fundamental tradeoffs between people's values")
    text("- Unintended consequences (social media, impact on education)")

    text("### Summary")
    text("- Ingredients of intelligence: perception, reasoning, action, learning")
    text("- Resource constraints: computation, information"), note("So we'll want to develop algorithms that are compute and data efficient.")
    text("- Developer goals: how do we build AI to accomplish something?")
    text("- Societal goals: how can we develop AI that benefits society?")


def about_this_course():
    text("Course philosophy")
    text("- Timeless foundations (e.g., gradient descent)")
    text("- Modern examples (e.g., GPT-5 solving 12/12 problems at the ICPC)")
    text("- Learn by doing: grounded in building practical applications")

    text("Changes this year:")
    text("- Tensor-native: from deep learning to value iteration to Bayesian network inference")
    text("- Cut constraint satisfaction problems :(")
    text("- Deep dive into societal impact (e.g., copyright, supply chains, policy)")

    text("All course policies, coursework, and schedule are online:")
    link("https://stanford-cs221.github.io/autumn2025/")


def what_is_this_program():
    text("This is an *executable lecture*, a program whose execution delivers the content of a lecture.")
    
    text("We can step through code:")
    total = 0  # @inspect total
    for x in [1, 2, 3]:  # @inspect x
        total += x  # @inspect total

    text("- Lectures inherit the hierarchical structure of code")
    text("- Code is more precise (than English and also than math)")
    text("- Need to write code to build AI at the end of the day")
    text("- Check out the toolbar at the top right (e.g., control display)")


if __name__ == "__main__":
    main()
