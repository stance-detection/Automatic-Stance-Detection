import csv

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']

def write_submission(test_d, predicted, filename):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Headline','Body ID','Stance'])
        for stance_row, label in zip(test_d.stances, predicted):
            writer.writerow([stance_row['Headline'], stance_row['Body ID'], LABELS[label]])


