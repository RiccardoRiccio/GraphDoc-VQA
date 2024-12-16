import numpy as np
from metrics import Evaluator

def test_evaluation_metrics():
    # Simulated ground truth and predictions
    gt_answers = [
        ['Paul'],  # Correct
        ['san diego'],  # Correct
        ['The Robert A. Welch Foundation'],  # Partial similarity
        ['itc limited'],  # Correct
        ['university of california'],  # Incorrect
        ['1128 SIXTEENTH ST., N. W., WASHINGTON, D. C. 20036'],  # Partial similarity
        ['aashirvaad'],  # Correct
        ['0.28'],  # Partial similarity
        ['11:14 to 11:39 a.m.']  # Incorrect
    ]

    predictions = [
        'D.',  # Correct
        'San Diego',  # Correct
        'Robert Foundation',  # Partial similarity
        'ITC Ltd.',  # Incorrect
        'University of California',  # Incorrect
        '1128 SIXTEENTH ST., NW, WASHINGTON, DC 20036',  # Partial similarity
        'aashirvaad',  # Correct
        'D.',  # Partial similarity
        '11:14 to 11:40 a.m.'  # Incorrect
    ]

    # Initialize evaluator
    evaluator = Evaluator()

    # Evaluate metrics
    metrics = evaluator.get_metrics(gt_answers, predictions)

    # Calculate mean metrics
    mean_accuracy = np.mean(metrics['accuracy']) if metrics['accuracy'] else 0
    mean_anls = np.mean(metrics['anls']) if metrics['anls'] else 0

    print(f"Predictions: {predictions}")
    print(f"Ground Truth Answers: {gt_answers}")
    print(f"Accuracy (per sample): {metrics['accuracy']}")
    print(f"ANLS (per sample): {metrics['anls']}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean ANLS: {mean_anls:.4f}")

if __name__ == "__main__":
    test_evaluation_metrics()
