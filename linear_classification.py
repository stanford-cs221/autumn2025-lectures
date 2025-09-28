import numpy as np
from typing import Callable
from dataclasses import dataclass
from edtrace import text, link, plot, image
from altair import Chart, Data
from einops import reduce
import tiktoken

def main():
    text("Last unit: linear regression")
    text("- Hypothesis class: linear functions")
    text("- Prediction task: input → output (number)")

    text("This unit: linear classification")
    text("- Hypothesis class: (thresholded) linear functions")
    text("- Prediction task: input → output (one of K choices)")

    link("https://stanford-cs221.github.io/autumn2023/modules/module.html#include=machine-learning%2Flinear-classification.js&mode=print6pp", title="[Autumn 2023 lecture]")

    text("Let's walk through the same steps as for linear regression and see what changes...")

    prediction_task()
    machine_learning_problem()

    hypothesis_class()
    zero_one_loss_function()
    optimization_algorithm()
    hinge_loss_function()
    optimization_algorithm_take_2()

    multiclass_classification()
    handling_text()

    text("Summary")
    text("- Linear classification: linear functions → one of K choices")

    
def prediction_task():
    text("Example task: image classification")
    text("- **Input**: an image; e.g.")
    image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Felis_catus-cat_on_snow.jpg/1920px-Felis_catus-cat_on_snow.jpg", width=200)
    text("- **Output**: what kind of object it is (e.g., cat)")

    text("What's the type of the **input**?")
    text("- Image: width x height x 3 (RGB) tensor")
    text("- Text: a string (hmm, not a tensor...we'll come back to this later)")

    text("What's the type of the **output**?")
    text("- Binary classification (two choices): usually {-1, 1}")
    text("- Multiclass classification (K choices): usually {0, 1, ..., K-1}")

    text("A **predictor** is a function that takes an input and produces an output.")
    text("Here's an example predictor for binary classification:")
    def simple_binary_classifier(x: np.ndarray) -> int:  # @inspect x
        logit = x[0] - x[1] - 1  # @inspect logit
        if logit > 0:
            predicted_y = 1  # @inspect predicted_y
        else:
            predicted_y = -1  # @inspect predicted_y
        return predicted_y

    text("Given an input `x`, we can get a prediction `predicted_y` by calling the predictor:")
    x = np.array([1, 2])  # @inspect x
    predicted_y = simple_binary_classifier(x)  # @inspect predicted_y
    x = np.array([2, 0])  #  @inspect x  @clear predicted_y
    predicted_y = simple_binary_classifier(x)  # @inspect predicted_y

    text("The points where logit = x[0] - x[1] - 1 = 0 is the **decision boundary**.")
    values = [{"x0": x, "x1": x - 1} for x in np.linspace(-3, 3, 30)]
    plot(Chart(Data(values=values)).mark_line().encode(x="x0:Q", y="x1:Q").to_dict())  # @clear values

    text("But how do we get the predictor?")


def machine_learning_problem():
    text("The **training data** is a set of examples that demonstrate the task.")
    text("Each **example** consists of an (input x, target output y) pair.")
    training_data = get_training_data()  # @inspect training_data

    text("A **learning algorithm** takes the training data and produces a predictor.")

    text("Key questions:")
    text("1. Which predictors are possible? **hypothesis class**")
    text("2. How good is a predictor? **loss function**")
    text("3. How do we compute the best predictor? **optimization algorithm**")


@dataclass(frozen=True)
class Example:
    x: np.ndarray
    target_y: float


def get_training_data():
    return [
        Example(x=np.array([1, 2]), target_y=-1),
        Example(x=np.array([2, 0]), target_y=1),
        Example(x=np.array([0, 0]), target_y=-1),
    ]


def hypothesis_class():
    text("Which predictors (classifiers) are possible?")

    text("For linear classifiers, each set of parameters has a **weight vector** and a **bias**.")
    params = Parameters(weight=np.array([1, -1]), bias=-1)
    x = np.array([1, -1])  #  @inspect x
    predicted_y = binary_classifier(params, x)  # @inspect y

    text("Here's another predictor:")  # @clear params x predicted_y
    params = Parameters(weight=np.array([1, -1]), bias=1)
    x = np.array([1, 1])  #  @inspect x
    predicted_y = binary_classifier(params, x)  # @inspect predicted_y

    text("The **hypothesis class** is the set of all predictors you can get by choosing parameters (weight, bias).")


@dataclass(frozen=True)
class Parameters:
    weight: np.ndarray
    bias: float


def binary_classifier(params: Parameters, x: np.ndarray) -> float:  # @inspect params x
    """Applies the linear predictor given by `params` to input `x`."""
    logit = params.weight @ x + params.bias  # @inspect logit
    predicted_y = 1 if logit > 0 else -1  # @inspect predicted_y
    return predicted_y


def zero_one_loss_function():
    text("The next design decision is how to judge each of the many possible predictors.")

    text("Let's consider a predictor:")
    params = Parameters(weight=np.array([1, -1]), bias=-1)  # @inspect params

    text("Recall the training data:")
    training_data = get_training_data()  # @inspect training_data @stepover
    
    text("How well does `params` fit `training_data`?")
    text("We define a loss function that measures how unhappy one point is based on params.")
    text("For regression, we used the squared loss:")
    text("Intuition: how far away the prediction (score) is from the target.")
    loss = squared_loss(Example(x=np.array([2, 0]), target_y=0.5), params)  # @inspect loss
    values = [{"residual": residual, "loss": residual ** 2} for residual in np.linspace(-1, 1, 30)]  # @stepover
    plot(Chart(Data(values=values)).mark_line().encode(x="residual:Q", y="loss:Q").to_dict())

    text("For binary classification, we use the zero-one loss:")
    text("Intuition: whether the prediction has the same sign as the target.")
    loss = zero_one_loss_using_predictor(Example(x=np.array([2, 0]), target_y=1), params)  # @inspect loss
    loss = zero_one_loss_using_predictor(Example(x=np.array([0, -2]), target_y=-1), params)  # @inspect loss
    text("We can write the zero-one loss in terms of the score.")
    loss = zero_one_loss(Example(x=np.array([2, 0]), target_y=1), params)  # @inspect loss
    values = [{"margin": margin, "loss": int(margin <= 0)} for margin in np.linspace(-1, 1, 30)]  # @stepover
    plot(Chart(Data(values=values)).mark_line().encode(x="margin:Q", y="loss:Q").to_dict())

    text("The training loss is the average of the per-example losses of the training examples.")  # @clear loss
    train_loss = train_zero_one_loss(params, training_data)  # @inspect train_loss


def zero_one_loss_using_predictor(example: Example, params: Parameters) -> float:  # @inspect example params
    predicted_y = binary_classifier(params, example.x)  # @inspect predicted_y
    return int(predicted_y != example.target_y)  # Whether the prediction was wrong


def zero_one_loss(example: Example, params: Parameters) -> float:  # @inspect example params
    logit = example.x @ params.weight + params.bias  # @inspect logit
    margin = logit * example.target_y  # @inspect margin
    return int(margin <= 0)  # Whether the prediction was wrong


def squared_loss(example: Example, params: Parameters) -> float:  # @inspect example params
    predicted_y = example.x @ params.weight + params.bias  # @inspect predicted_y
    residual = predicted_y - example.target_y  # @inspect residual
    return residual ** 2


def train_zero_one_loss(params: Parameters, training_data: list[Example]) -> float:  # @inspect params training_data
    losses = [zero_one_loss(example, params) for example in training_data]  # @inspect losses @stepover
    train_loss = np.mean(losses)  # @inspect train_loss
    return train_loss


def optimization_algorithm():
    text("Recall that for every set of parameters `params`, we can compute the training loss `train_loss`.")

    text("Recall in linear regression we optimized the parameters using gradient descent.")
    text("So let's do the same thing here.")

    params = Parameters(weight=np.array([1, 1]), bias=0)  # @inspect params
    training_data = get_training_data()  # @inspect training_data @stepover
    train_loss = train_zero_one_loss(params, training_data)  # @inspect train_loss

    text("We want to find the parameters that yield the lowest training loss.")
    text("This is an optimization problem as with linear regression.")

    text("Let's take the gradient")
    grad = gradient_zero_one_loss(training_data[0], params)  # @inspect grad
    text("We have a problem: the gradient is zero everywhere!")
    text("So gradient descent won't move!")
    text("Intuition: if example is wrong, moving parameters a bit won't make it right, so no local improvement.")
    text("So what do we do?")


def gradient_zero_one_loss(example: Example, params: Parameters) -> Parameters:  # @inspect example params
    score = example.x @ params.weight + params.bias  # @inspect score
    margin = score * example.target_y  # @inspect margin
    # Zero everywhere except when margin = 0, where it's undefined
    return Parameters(weight=np.zeros_like(params.weight), bias=0)


def logistic_function():
    text("Score is a number between -∞ and +∞")
    text("We want to convert a score into a probability (must be between 0 and 1)")
    text("There are many functions that do this, ")
    text("The logistic function takes a number between -∞ and +∞ and maps it to a number between 0 and 1.")

    plot(make_plot("logistic function", "x", "y", logistic))

    score = 0  # @inspect score
    p = logistic(score)  # @inspect p
    score = 1  # @inspect score
    p = logistic(score)  # @inspect p
    score = 8  # @inspect score
    p = logistic(score)  # @inspect p
    score = -1  # @inspect score
    p = logistic(score)  # @inspect p

    text("Another interpretation: log odds")
    p = 0.2  # @inspect p
    odds = p / (1 - p)  # @inspect odds
    score = np.log(odds)  # @inspect score
    p2 = logistic(score)  # @inspect p2
    assert p == p2

    text("Properties")
    text("- As score → -∞, p → 0")
    text("- As score → +∞, p → 1")
    text("- As score → 0, p → 0.5")

    p1 = logistic(score=3)  # @inspect p
    p2 = logistic(score=-3)  # @inspect p2
    assert p1 + p2 == 1

    text("The derivative of the logistic function is quite simple")
    dp = d_logistic(score=3)  # @inspect dp

    plot(make_plot("derivative of logistic function", "x", "y", d_logistic))


def logistic(score: float) -> float:  # @inspect score
    return 1 / (1 + np.exp(-score))


def d_logistic(score: float) -> float:
    return logistic(score) * (1 - logistic(score))

def logistic_loss_function():
    params = Parameters(weight=np.array([1, -1]), bias=1)  # @inspect params
    example = Example(x=np.array([2, 0]), target_y=1)  # @inspect example
    text("Before, our predictor turns a score into a single prediction")    
    logit = example.x @ params.weight + params.bias  # @inspect score
    predicted_y = 1 if logit > 0 else -1  # @inspect predicted_y
    
    text("But let us turn the scores into a probability")
    logistic_function()

    text("Now we can compute the probability of y")
    prob_pos = logistic(logit)  # @inspect prob_pos
    prob_neg = logistic(-logit)  # @inspect prob_neg
    prob_target = logistic(logit * example.target_y)  # @inspect prob_target

    text("If we have multiple examples, we'd multiply the probabilities.")
    text("Equivalent to summing the log probabilities.")
    log_prob_target = np.log(prob_target)  # @inspect log_prob_target

    text("To turn this into a loss, we can just negate it.")
    log_loss = -log_prob_target  # @inspect log_loss

    text("Maximum likelihood: find parameters that maximize the log probability of the training targets")

    text("Packaging it up:")
    loss = logistic_loss(example, params)  # @inspect loss

    training_data = get_training_data()  # @inspect training_data
    train_loss = train_logistic_loss(params, training_data)  # @inspect train_loss
    grad = gradient_train_logistic_loss(params, training_data)  # @inspect grad


def logistic_loss(example: Example, params: Parameters) -> float:  # @inspect example params
    logit = example.x @ params.weight + params.bias  # @inspect logit
    prob_target = logistic(logit * example.target_y)  # @inspect prob_target
    return -np.log(prob_target)


def gradient_logistic_loss(example: Example, params: Parameters) -> Parameters:  # @inspect example params
    logit = example.x @ params.weight + params.bias  # @inspect logit
    prob_target = logistic(logit * example.target_y)  # @inspect prob_target
    dloss_dlogit = -logistic(-logit * example.target_y)  # @inspect dloss_dlogit
    grad_weight = -example.target_y * example.x * dloss_dlogit  # @inspect grad_weight
    grad_bias = -example.target_y * dloss_dlogit  # @inspect grad_bias
    return Parameters(weight=grad_weight, bias=grad_bias)


def train_logistic_loss(params: Parameters, training_data: list[Example]) -> float:  # @inspect params training_data
    losses = [logistic_loss(example, params) for example in training_data]  # @inspect losses @stepover
    train_loss = np.mean(losses)  # @inspect train_loss
    return train_loss


def gradient_train_logistic_loss(params: Parameters, training_data: list[Example]) -> Parameters:  # @inspect params training_data
    grads = [gradient_logistic_loss(example, params) for example in training_data]  # @inspect grads @stepover
    mean_weight = np.mean([grad.weight for grad in grads])  # @inspect mean_weight
    mean_bias = np.mean([grad.bias for grad in grads])  # @inspect mean_bias
    return Parameters(weight=mean_weight, bias=mean_bias)


def hinge_loss_function():
    text("Let's define a **surrogate** loss function that is easier to optimize.")

    data = make_plot("zero-one loss", "margin", "loss", lambda margin: int(margin <= 0))
    plot(data)  # @stepover

    text("The **hinge loss**")
    data = make_plot("hinge loss", "margin", "loss", lambda margin: max(0, 1 - margin))
    plot(data)  # @stepover

    example = Example(x=np.array([2, 0]), target_y=1)
    params = Parameters(weight=np.array([1, -1]), bias=1)  # @inspect params
    loss = hinge_loss(example, params)  # @inspect loss

    training_data = get_training_data()  # @inspect training_data
    train_loss = train_hinge_loss(params, training_data)  # @inspect train_loss


def make_plot(title: str, xlabel: str, ylabel: str, f: Callable[[float], float]) -> dict:
    values = [{xlabel: x, ylabel: f(x)} for x in np.linspace(-1, 1, 30)]
    return Chart(Data(values=values)).mark_line().encode(x=f"{xlabel}:Q", y=f"{ylabel}:Q").to_dict()


def hinge_loss(example: Example, params: Parameters) -> float:  # @inspect example params
    score = example.x @ params.weight + params.bias  # @inspect score
    margin = score * example.target_y  # @inspect margin
    loss = max(0, 1 - margin)  # @inspect loss
    return loss


def train_hinge_loss(params: Parameters, training_data: list[Example]) -> float:  # @inspect params training_data
    losses = [hinge_loss(example, params) for example in training_data]  # @inspect losses @stepover
    train_loss = np.mean(losses)  # @inspect train_loss
    return train_loss


def optimization_algorithm_take_2():
    params = Parameters(weight=np.array([0, 0]), bias=0)  # @inspect params
    training_data = get_training_data()  # @inspect training_data

    text("Let us compute the gradient.")
    grad = gradient_train_hinge_loss(params, training_data)  # @inspect grad

    text("Now we can do gradient descent.")
    gradient_descent()


def gradient_hinge_loss(example: Example, params: Parameters) -> Parameters:  # @inspect example params
    score = example.x @ params.weight + params.bias  # @inspect score
    margin = score * example.target_y  # @inspect margin
    if 1 - margin >= 0:  # Incorrectly classified
        grad_weight = -example.target_y * example.x  # @inspect grad_weight
        grad_bias = -example.target_y  # @inspect grad_bias
    else:  # Correctly classified
        grad_weight = np.zeros_like(params.weight)  # @inspect grad_weight
        grad_bias = 0  # @inspect grad_bias
    return Parameters(weight=grad_weight, bias=grad_bias)


def gradient_train_hinge_loss(params: Parameters, training_data: list[Example]) -> Parameters:  # @inspect params training_data
    grads = [gradient_hinge_loss(example, params) for example in training_data]  # @inspect grads @stepover
    mean_weight = np.mean([grad.weight for grad in grads])  # @inspect mean_weight
    mean_bias = np.mean([grad.bias for grad in grads])  # @inspect mean_bias
    return Parameters(weight=mean_weight, bias=mean_bias)


def gradient_descent():
    training_data = get_training_data()  # @stepover
    params = Parameters(weight=np.array([0, 0]), bias=0)  # @inspect params
    learning_rate = 0.01
    losses = []
    for step in range(10):  # @inspect step
        train_loss = train_hinge_loss(params, training_data)  # @inspect train_loss @stepover
        grad = gradient_train_hinge_loss(params, training_data)  # @inspect grad @stepover
        params = Parameters(  # @inspect params
            weight=params.weight - learning_rate * grad.weight,
            bias=params.bias - learning_rate * grad.bias,
        )
        losses.append(train_loss)

    plot(Chart(Data(values=[{"step": i, "loss": loss} for i, loss in enumerate(losses)])).mark_line().encode(x="step:Q", y="loss:Q").to_dict())


def multiclass_classification():
    text("Binary classification (output y ∈ {-1, 1})")
    text("Multiclass classification (output y ∈ {0, 1, ..., K-1})")

    text("For binary classification, we compute a single score for an input")
    text("Negative score means -1, positive score means 1")
    x = np.array([1, -1])  # @inspect x
    params = Parameters(weight=np.array([1, -1]), bias=1)  # @inspect params
    predicted_y = binary_classifier(params, x)  # @inspect predicted_y

    text("For multiclass classification")
    text("- Define a weight vector for each class")
    text("- Compute a score for each class")
    text("- Predict a distribution over classes")
    params = Parameters(weight=np.array([[1, -1], [1, -1]]), bias=np.array([1, 1]))  # @inspect params
    x = np.array([1, -1])  # @inspect x
    logits = params.weight @ x + params.bias
    introduce_softmax()
    probs = softmax(logits)  # @inspect probs

    text("Now let us define the cross entropy loss.")
    introduce_cross_entropy()


def introduce_softmax():
    text("Recall: the logistic function maps (-∞, +∞) to (0, 1)")

    text("The softmax function generalizes this to multiple classes")

    logits = np.array([1, -1, 0])  # @inspect logits
    probs = softmax(logits)  # @inspect softmax_logits

    text("Shifting up scores doesn't change the relative probabilities")
    logits1 = np.array([1, -1, 0])  # @inspect logits1
    probs1 = softmax(logits1)  # @inspect probs1
    logits2 = np.array(logits1 + 2)  # @inspect logits2
    probs2 = softmax(logits2)  # @inspect probs2
    assert np.allclose(probs1, probs2)


def softmax(logits: np.ndarray) -> np.ndarray:  # @inspect logits
    exp_logits = np.exp(logits)  # @inspect exp_logits
    return exp_logits / np.sum(exp_logits)  # @inspect softmax


def multiclass_classifier(params: Parameters, x: np.ndarray) -> int:  # @inspect params x
    scores = [x @ params.weight[y] + params.bias[y] for y in range(len(params.weight))]  # @inspect scores
    predicted_y = np.argmax(scores)  # @inspect predicted_y
    return predicted_y


def introduce_cross_entropy():
    text("Cross entropy: measures the difference between a target distribution and a predicted distribution")
    target = np.array([0.5, 0.2, 0.3])  # @inspect target
    predicted = np.array([0.1, 0.5, 0.4])  # @inspect predicted

    terms = target * -np.log(predicted)  # @inspect terms
    cross_entropy = np.sum(terms)  # @inspect cross_entropy

    text("Special case: target is a single label (represented as a one-hot vector)")
    target = np.array([0, 1, 0])  # @inspect target
    predicted = np.array([0.1, 0.5, 0.4])  # @inspect predicted
    terms = target * -np.log(predicted)  # @inspect terms
    cross_entropy = np.sum(terms)  # @inspect cross_entropy
    text("This is the same as the negative log probability of the target class.")


def handling_text():
    tokenization()


def tokenization():
    string = "the cat in the hat"

    text("How do we represent a string as a tensor?")
    text("1. Tokenization: convert a string into a sequence of integers.")
    text("2. Represent each integer as a one-hot vector.")

    text("### Tokenization")
    text("Split a string by space into words and convert them into integers.")
    vocab = Vocabulary()  # @inspect vocab
    words = string.split()  # @inspect words
    indices = [vocab.get_index(word) for word in words]  # @inspect indices vocab

    text("Language models use more sophisticated tokenizers (Byte-Pair Encoding) "), link("https://arxiv.org/pdf/1508.07909")
    text("To get a feel for how tokenizers work, play with this "), link(title="interactive site", url="https://tiktokenizer.vercel.app/?encoder=gpt2")
    tokenzier = tiktoken.get_encoding("gpt2")
    gpt2_indices = tokenzier.encode(string)  # @inspect gpt2_indices

    text("### Interpretation")  # clear @gpt2_indices
    text("Treat each index as a one-hot vector.")
    index = indices[4]  # @inspect index
    vector = np.eye(len(vocab))[index]  # @inspect vector @stepover

    text("So the string is represented as a sequence of vectors, or a matrix:")
    matrix = np.eye(len(vocab))[indices]  # @inspect matrix @stepover @clear index vector

    text("### Operations")

    text("In practice, we store the indices and not the one-hot vectors to save memory.")
    text("We can operate directly using the indices.")
    text("Suppose we want to take the dot product of each position with `w`.")
    w = np.random.randn(len(vocab))  # @inspect w @stepover

    # Use a matrix-vector product (don't do this!)
    y_dot = matrix @ w  # @inspect y_dot

    # Equivalently, index into the weight vectors (do this!)
    y_index = w[indices]  # @inspect y_index

    text("Bag of words representation")
    text("Each word is represented as a (one-hot) vector.")
    text("The representation is the average of the one-hot vectors.")
    bow = reduce(matrix, "pos vocab -> vocab", "mean")  # @inspect b
    # Then operate on it
    y_bow = bow @ w  # @inspect y_bow

    text("Summary")
    text("- Problem: convert strings to tensors for machine learning")
    text("- Solution: tokenization + one-hot encoding")
    text("- Tokenization: split strings into words and build up a vocabulary (string ↔ index)")
    text("- Mathematically one-hot vectors; in code, directly work with indices")


class Vocabulary:
    """Maps strings to integers."""
    def __init__(self):
        self.index_to_string: list[str] = []
        self.string_to_index: dict[str, int] = {}

    def get_index(self, string: str) -> int:  # @inspect string
        index = self.string_to_index.get(string)  # @inspect index
        if index is None:  # New string
            index = len(self.index_to_string)  # @inspect index
            self.index_to_string.append(string)
            self.string_to_index[string] = index
        return index

    def get_string(self, index: int) -> str:
        return self.index_to_string[index]

    def __len__(self):
        return len(self.index_to_string)

    def asdict(self):
        return {
            "index_to_string": self.index_to_string,
            "string_to_index": self.string_to_index,
        }



if __name__ == "__main__":
    main()
