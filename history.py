from edtrace import text, image, link
from util import article_link


def main():
    link("https://stanford-cs221.github.io/autumn2023/modules/module.html#include=general%2Fhistory.js&slideId=turing&level=0&mode=print6pp", title="Slides from Autumn 2023")

    turing_test()

    text("The history of AI is the story of three intellectual traditions:")
    image("images/symbolic-neural-statistical.png", width=500)

    symbolic_ai()
    neural_ai()
    statistical_ai()

    foundation_models()
    parting_thoughts()


def turing_test():
    text("In 1950, this man:")
    image("images/alan-turing.jpg", width=150)
    text("published this paper:")
    image("images/turing-1950-paper.jpg", width=300)

    text("Alan Turing asked: \"Can machines think?\"")
    text("More basically: How could you tell?")
    text("His answer: Imitation Game (the Turing Test)")
    image("images/turing-test.jpg", width=200)

    text("Significance: ground philosophical question in objective measurement")
    text("Left open the solution: machine learning? logic?")


def symbolic_ai():
    text("1956: John McCarthy organized workshop at Dartmouth College")
    image("images/dartmouth.jpg", width=100)
    text("- Convened the leading thinkers of the day (Shannon, Minsky, etc.)")
    text("- Goal was to make a \"significant advance\" in 2 months")
    text("- Coined the term \"artificial intelligence\"")

    text("1952: Arthur Samuel\'s chckers playing program")
    text("- Weights were learned")
    text("- Played at strong amateur level")

    text("1955: Newell &amp; Simon\'s Logic Theorist")
    text("- Used search + heuristics")
    text("- Came up with new proof for theorems in Principia Mathematica")

    text("Overwhelming optimism...")
    text("- Herbert Simon: *Machines will be capable, within twenty years, of doing any work a man can do.*")
    text("- Marvin Minsky: *Within 10 years the problems of artificial intelligence will be substantially solved.*")
    text("- Claude Shannon: *I visualize a time when we will be to robots what dogs are to humans, and I\'m rooting for the machines.*")

    text("Underwhelming results...")
    text("Folklore example from machine translation:")
    text("- English: *The spirit is willing but the flesh is weak.*")
    text("- Russian: ...")
    text("- English: *The vodka is good but the meat is rotten.*")
    
    text("1966: ALPAC report cut off government funding for machine translation, first AI winter ❄️")

    text("What went wrong?")
    text("Problems")
    text("- Limited computation: search space grew exponentially, outpacing hardware")
    text("- Limited information: complexity of AI problems (number of words, objects, concepts in the world)")
    text("Silver lining: useful contributions (John McCarthy)")
    text("- Lisp: advanced programming language")
    text("- Garbage collection: don't have to (de)allocate memory")
    text("- Time-sharing: allow multiple people to use the same computer at once")

    text("Knowledge-based systems (70-80s)")
    image("images/knowledge-key.jpg"),
    text("Expert systems: elicit specific domain knowledge from experts in form of rules"),
    image("images/mycin-rule.png")
    text("Systems:")
    text("- DENDRAL: infer molecular structure from mass spectrometry")
    text("- MYCIN: diagnose blood infections, recommend antibiotics")
    text("- XCON: convert customer orders into parts specification")

    text("Wins:")
    text("- Knowledge helped both the information and computation gap")
    text("- First real application that impacted industry")
    text("Shortcomings:")
    text("- Deterministic rules couldn't handle the uncertainty of the real world")
    text("- Rules quickly became too complex to create and maintain")

    text("1987: Collapse of Lisp machines and second AI winter ❄️")


def neural_ai():
    text("Artificial neural networks")
    text("- 1943: artificial neural networks, relate neural circuitry and mathematical logic (McCulloch/Pitts)")
    text("- 1949: \"cells that fire together wire together\" learning rule (Hebb)")
    text("- 1958: Perceptron algorithm for linear classifiers (Rosenblatt)")
    text("- 1959: ADALINE device for linear regression (Widrow/Hoff)")
    text("- 1969: Perceptrons book showed that linear models could not solve XOR, killed neural nets research (Minsky/Papert)")

    text("Revival of connectionism")
    text("- 1980: Neocognitron, a.k.a. convolutional neural networks for images (Fukushima)")
    text("- 1986: popularization of backpropagation for training multi-layer networks (Rumelhardt, Hinton, Williams)")
    text("- 1989: applied convolutional neural networks to recognizing handwritten digits for USPS (LeCun)")

    text("Neural networks were hard to train and were unpopular in the 2000s")

    text("Deep learning")
    text("- 2006: unsupervised layerwise pre-training of deep networks (Hinton et al.)")
    text("- 2009: neural networks outperform Hidden Markov Models in speech recognition, transformed speech community")
    text("- 2012: AlexNet obtains huge gains in object recognition; transformed computer vision community")
    text("- 2014: sequence-to-sequence modeling (for machine translation) "), link("https://arxiv.org/abs/1409.3215")
    text("- 2014: Adam optimizer "), link("https://arxiv.org/abs/1412.6980")
    text("- 2015: Attention mechanism (for machine translation) "), link("https://arxiv.org/abs/1409.0473")
    text("- 2016: AlphaGo uses deep reinforcement learning, defeat world champion Lee Sedol in Go")
    text("- 2017: Transformer architecture (for machine translation) "), link("https://arxiv.org/abs/1706.03762")


def statistical_ai():
    text("Early ideas outside AI")
    text("- 1801: linear regression (Gauss, Legendre)")
    text("- 1936: linear classification (Fisher)")
    text("- 1951: stochastic gradient descent (Robbins/Monro)")
    text("- 1956: Uniform cost search for shortest paths (Dijkstra)")
    text("- 1957: Markov decision processes (Bellman)")

    text("Statistical machine learning")
    text("- 1985: Bayesian networks enabled reasoning under uncertainty (Pearl)")
    text("- 1995: support vector machines (Cortes/Vapnik) became popular in ML: easier to train, rooted in statistical learning theory")
    text("- 1999: variational inference was popularized by Jordan/Jaakkola")
    text("- 2001: conditional random fields allowed for predicting structures (Lafferty/McCallum/Pereira)")
    text("- 2003: topic modeling allowed for hierarchies and uncertainty over parameters (Blei/Ng/Jordan)")


def foundation_models():
    text("### Pretrained language models")
    text("- ELMo: pretraining with LSTMs, fine-tuning helps downstream tasks "), link("https://arxiv.org/abs/1802.05365")
    text("- BERT: pretraining with Transformer, fine-tuning helps downstream tasks "), link("https://arxiv.org/abs/1810.04805")
    text("- Google's T5 (11B): cast everything as text-to-text "), link("https://arxiv.org/abs/1910.10683")
    image("images/t5.png", width=400)

    text("### Scaling up")
    text("- GPT-2: fluent text, first signs of zero-shot capabilities "), link("https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", title="[Radford+ 2019]")
    text("- Scaling laws: provide hope / predictability for scaling "), link("https://arxiv.org/abs/2001.08361")
    text("- GPT-3: in-context learning, closed "), link("https://arxiv.org/abs/2005.14165")
    text("- Chinchilla: compute-optimal scaling laws "), link("https://arxiv.org/abs/2005.14165")
    text("- Llama 3 "), link("https://arxiv.org/abs/2407.21783")
    text("- DeepSeek v3 "), link("https://arxiv.org/abs/2412.19437")

    text("### Reasoning")
    text("- Answering hard questions requires thinking")
    text("- Language models produce \"thoughts\" before producing a response")
    text("- Models: OpenAI's o1-o4, DeepSeek's r1")

    text("### Industrialization of AI")
    image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Industrialisation.jpg/440px-Industrialisation.jpg", width=400)

    text("GPT-4 supposedly has 1.8T parameters. "), article_link("https://www.hpcwire.com/2024/03/19/the-generative-ai-future-is-now-nvidias-huang-says")
    text("GPT-4 supposedly cost $100M to train. "), article_link("https://www.wired.com/story/openai-ceo-sam-altman-the-age-of-giant-ai-models-is-already-over/")
    text("xAI builds cluster with 200,000 H100s to train Grok. "), article_link("https://www.tomshardware.com/pc-components/gpus/elon-musk-is-doubling-the-worlds-largest-ai-gpu-cluster-expanding-colossus-gpu-cluster-to-200-000-soon-has-floated-300-000-in-the-past")
    text("Stargate (OpenAI, NVIDIA, Oracle) invests $500B over 4 years. "), article_link("https://openai.com/index/announcing-the-stargate-project/")

    text("There are no public details on how frontier models are built.")
    text("From the GPT-4 technical report "), link("https://arxiv.org/abs/2303.08774"), text(":")
    image("images/gpt4-no-details.png", width=600)

    text("AI has emerged from research and now is shaping businesses and public policy.")
    text("The research is still far from done...")


def parting_thoughts():
    text("Fierce battles between the traditions")
    text("- Minsky/Papert promoted symbolic AI and killed neural networks research")
    text("- Statistical ML in the 2000s thought neural networks were dead")

    text("Deeper connections")
    text("- McCulloch/Pitts introduced artificial neural networks, but paper is about how to implement logical operations")
    text("- Go is defined purely using symbols, but deep neural networks are key to playing the game")
    text("- Deep learning was initially all about perception, but now turn to reasoning (goals of symbolic AI)")

    text("AI is a melting pot:")
    text("- Symbolic AI: provided the vision and ambition")
    text("- Neural AI: provided the model architectures")
    text("- Statistical AI: provided the rigor (e.g., optimization, generalization)")

    text("This class: we will see elements of all three traditions")


if __name__ == "__main__":
    main()
