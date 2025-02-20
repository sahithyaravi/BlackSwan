def get_ground_truths(alt_gt, checkboxes, task_gt, task):
    # Based on the checkboxes, pick the appropriate ground truths
    ground_truths = []
    if task == 'task2':
        for i, checkbox in enumerate(checkboxes):
            if checkbox == 'yes':
                ground_truths.append(alt_gt[i])
            else:
                ground_truths.append(task_gt[i])

    if task == 'task3':
        ground_truths.append(task_gt)
        for i, checkbox in enumerate(checkboxes):
            if checkbox == 'yes':
                ground_truths.append(alt_gt[i])

    return ground_truths