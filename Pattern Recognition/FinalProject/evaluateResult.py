# Author: Hamidreza Nademi

import numpy as np


def compute_experimental_result(predicted_numbers, classifier, test_or_train):
    confusion_matrix = np.zeros((10, 10))
    recall_matrix = []
    precision_matrix = []
    f1_score_matrix = []
    average_f1_score = None
    avg_accuracy = None
    total_num = 0

    for key in predicted_numbers.keys():
        for predicted_num in predicted_numbers[key]:
            confusion_matrix[key][predicted_num] += 1

    for key in predicted_numbers.keys():
        total_num += sum(confusion_matrix[key])
        # compute recall for each class
        recall_matrix.append(
            round(confusion_matrix[key][key]/sum(confusion_matrix[key]), 2)
        )

        # compute precision for each class
        precision_matrix.append(
            round(confusion_matrix[key][key]/sum(confusion_matrix[::, key]), 2)
        )

        # compute f1 score for each class
        f1_score_matrix.append(
            round((2*(precision_matrix[key]*recall_matrix[key])) /
                  (precision_matrix[key]+recall_matrix[key]), 2)
        )

    # compute average f1 score
    average_f1_score = round(sum(f1_score_matrix)/len(f1_score_matrix), 2)

    # compute average accuracy score
    avg_accuracy = round(sum(np.diag(confusion_matrix)/total_num), 2)

    # output result
    print(
        f"""
classifier: {classifier} , on {test_or_train} dataset
Confusion matrix:
{confusion_matrix} \n
Precision for each class: {precision_matrix} \n
Recall for each class: {recall_matrix} \n
F1 score for each class: {f1_score_matrix} \n
Mean F1 score: {average_f1_score} \n
Mean accuracy: {avg_accuracy} \n\n
        """
    )
