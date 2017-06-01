import csv

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']

def write_submission(test_d, predicted, filename):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Headline','Body ID','Stance'])
        for stance_row, label in zip(test_d.stances, predicted):
            #print(stance_row, label)
            #print(type(stance_row["Headline"]), type(stance_row["Body ID"]))
            writer.writerow([stance_row['Headline'], stance_row['Body ID'], label])


