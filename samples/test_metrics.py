from metrics import Evaluator  # Assuming evaluator.py is your file

def run_anls_example():
    evaluator = Evaluator()

    # Ground Truths (GT) and Predictions
    ground_truths = [
        ['trimegestone - publication and abstract tracking report', 
         'trimegestone - publication and abstract tracking report', 
         'publication and abstract tracking report'],
        ['a2', 'a2'],
        ['a1', 'a1'],
        ['300-us endometrial, bleeding and safety', 
         '300-us endometrial, bleeding and safety'],
        ['300-us bone mineral density- final data analysis', 
         '300-us bone mineral density- final data analysis'],
        ['$10,000'],
        ['$16,000'],
        ['$175,000'],
        ['$6,000'],
        ['$20,000'],
        ['12'],
        ['dwrite 065775', 'dwrite 065775']
    ]

    predictions = [
        'report',
        '1',
        '1',
        'what is the title of the second article?',
        'what is the title of the third article?',
        '$1000',
        '$1000',
        '$15,000',
        '$15,000',
        '$1000',
        '1',
        ''
    ]

    # Get metrics
    results = evaluator.get_metrics(ground_truths, predictions)

    # Print ANLS results
    print("ANLS Scores (Per Sample):", results['anls'])
    print("Mean ANLS:", sum(results['anls']) / len(results['anls']))

if __name__ == "__main__":
    run_anls_example()
