from continual_learning.metrics import (
    average_accuracy,
    backward_transfer,
    forward_transfer,
)


def test_continual_learning_metrics():
    # Rows represent training stages: English, Hindi, German.
    # Columns represent evaluation languages: English, Hindi, German.
    performance_matrix = [
        [0.80, 0.40, 0.30],
        [0.70, 0.75, 0.45],
        [0.65, 0.60, 0.80],
    ]

    # Pretrained model accuracy before any language training.
    zero_shot_accuracy = [0.50, 0.35, 0.25]

    acc = average_accuracy(performance_matrix)
    bwt = backward_transfer(performance_matrix)
    fwt = forward_transfer(
        performance_matrix,
        zero_shot_accuracy,
    )

    assert abs(acc - 0.6833333333) < 1e-6
    assert abs(bwt - (-0.15)) < 1e-6
    assert abs(fwt - 0.125) < 1e-6


from continual_learning.metrics import (
    forgetting,
    final_performance,
)



def test_forgetting():
    # Rows represent results after English, Hindi, and German training.
    performance_matrix = [
        [0.80, 0.40, 0.30],
        [0.70, 0.75, 0.45],
        [0.65, 0.60, 0.80],
    ]

    # After Hindi:
    # English forgetting = 0.80 - 0.70 = 0.10
    # Stage-average forgetting = 0.10
    #
    # After German:
    # English forgetting = 0.80 - 0.65 = 0.15
    # Hindi forgetting = 0.75 - 0.60 = 0.15
    # Stage-average forgetting = 0.15
    #
    # Overall forgetting = (0.10 + 0.15) / 2 = 0.125
    assert abs(forgetting(performance_matrix) - 0.125) < 1e-6



def test_final_performance():
    performance_matrix = [
        [0.80, 0.40, 0.30],
        [0.70, 0.75, 0.45],
        [0.65, 0.60, 0.80],
    ]

    # Mean accuracy in the final row:
    # (0.65 + 0.60 + 0.80) / 3 = 0.683333...
    assert abs(final_performance(performance_matrix) - 0.6833333333) < 1e-6
