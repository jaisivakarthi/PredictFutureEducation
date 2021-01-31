#Create dataset
#Import libraries
import pandas as pd
import numpy as np

#Declare variables
nrows = 5000
num_of_std = 10
subjects = ['Tamil', 'English', 'Maths', 'Science', 'Social']
num_subjects = len(subjects)
header = ['Student Name']

#Extend the subjects in header depend on the no of standard(1-10th)
for std in range(1,num_of_std+1):   
    header.extend(subjects)

avg_sub_header = []
avg_title = "Avg "

#Create the avg header for each subjects
for avg_sub in subjects:
    header.append(avg_title+avg_sub)
    avg_sub_header.append(avg_title+avg_sub)

header.append('Max Avg')
header.append('Max Sub')
header.append('Outcome')

data_set = []
mark_range = {'90range':{'min_mark':90,'max_mark':94},'95range':{'min_mark':95,'max_mark':100},'80range':{'min_mark':80,'max_mark':89}}
future_edu = {'Science':'Medical', 'Maths':'Engineering', 'English':'Arts','Tamil':'BA','Social':'Commerce'}

'''Create dynamic rows of student marks, average of each subjects, max of subjects and outcome'''
for row_num in range(1,nrows+1):
    record = {}
    record['Student Name'] = "Student " + str(row_num)

    #Initialize the total for subjects
    for avg_sub in avg_sub_header:
        record[avg_sub] = 0
    
    #Create random marks based on dictionary 
    for std in range(1,num_of_std+1):   
        for sjt in subjects:
            if not row_num % 10 and sjt=="Science":
                min_mark = mark_range['95range']['min_mark']
                max_mark = mark_range['95range']['max_mark']
                # print(row_num,"Science")
            elif not row_num % 11 and sjt=="Maths":
                min_mark = mark_range['90range']['min_mark']
                max_mark = mark_range['90range']['max_mark']
                # print(row_num,"Maths")
            else:
                min_mark = mark_range['80range']['min_mark']
                max_mark = mark_range['80range']['max_mark']

            record[sjt] = np.random.randint(min_mark, max_mark)
            
            #Calculate the total for each subjects in a row
            record[avg_title + sjt] += record[sjt]

    #Find the average and maximum average for each subjects       
    max_avg = []
    for avg_sub in avg_sub_header:
        record[avg_sub] = record[avg_sub] / num_of_std
        max_avg.append(record[avg_sub])

    record['Max Avg'] = max(max_avg)
    record['Max Sub'] = subjects[max_avg.index(record['Max Avg'])]
    out_come = []
    temp_val = ''

    #Create Outcome for the Future education based on Max subject
    if record['Max Sub'] in future_edu:
        temp_val = future_edu[record['Max Sub']]
    record['Outcome'] = temp_val

    #Append the dict with in a list 
    data_set.append(record)

#Make the dataframe
df = pd.DataFrame(columns=header,index=None,data=data_set)
df.index = np.arange(1, len(df)+1)

#Save the dataframe to data_set.csv file
df.to_csv('data_set.csv', index=False)
