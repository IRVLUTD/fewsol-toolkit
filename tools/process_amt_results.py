#!/usr/bin/env python3
import os
import json


bad_worker_ids = ['A23761J8YYFKP2', 'A3GIHD8GJU04MU', 'A1J3ICF1NZYFCR', 'A20EVX30W7PLZK', 'A2A15NSJTRE2KG',
    'A3UWL7Z2LMBIRV', 'A3KWP11UYLJFUU', 'A2TZ0S3XYATYMF', 'A29GLTXKFWGH1Z', 'A28Z98VFBY06YJ', 'A1NSS58MHS3QXY',
    'A2C556VYMG2BXC', 'A3DEZXXGPTA8OQ', 'A3BLJCUEE98FKI', 'A395NSOQ0WFA31', 'A2P7LFZWAQKIKJ', 'A24C6QQIDFZ5NO',
    'A1T73I9X70FX6Z', 'A19FSND0BO2IUT', 'ACQFVUOP54AXB', 'AI110BKVWIN2F', 'A11VIIKMLHTBRG', 'A20VCUTWVO6OQO',
    'A15AGJDQAZP1DI', 'AAQ3D3R3L5AC7', 'A32ZLUYSK3OU5X', 'A2HX13XTRKDD9C', 'AGOV022EHLH59', 'A1EI8Z972FIFJ3',
    'A14TOTP1WC9RUM', 'A2K3HEVVEWLUS4', 'ACKF1WBCKVLY3', 'AQGWZNJI8VWMK', 'A2KWEVSI7P1RGQ', 'A1XXRBBQ6HFGBA', 
    'A7YMQ5QANDW22', 'A2TM29JCFVDPM', 'AYGMTQOAG9Z8V', 'A3K3GK7LVBLHFZ', 'A2RX8ZFMISKO5T', 'AFIK3VBMMX6G6',
    'A27F7YNICYO6LP', 'A2A3IGLOU9F759', 'A1H6DME332958N']


def merge_annotations(annotations, object_id, answers):

    if object_id not in annotations:
        annotations[object_id] = answers
    else:
        # merge
        previous = annotations[object_id]
        for i in range(len(answers)):

            if answers[i] == '':
                continue

            prev_strings = previous[i].split(',')
            prev_strings = [s.strip() for s in prev_strings]

            strings = answers[i].split(',')
            for j in range(len(strings)):
                s = strings[j].strip()
                if s not in prev_strings:
                    if previous[i] == '':
                        previous[i] = s
                    else:
                        previous[i] = previous[i] + ', ' + s
        annotations[object_id] = previous

    return annotations


def clean_answers(answers):

    num = len(answers)
    for i in range(num):
        answers[i] = answers[i].lower().strip()
        answers[i] = answers[i].replace('&amp;', '&')
        answers[i] = answers[i].replace('&#8217;', '\'')
        answers[i] = answers[i].replace('.', '')
        answers[i] = answers[i].replace('eaing', 'eating')

        strings = answers[i].split(',')
        strings = [s.strip() for s in strings]

        strings_new = []
        for s in strings:
            if s != 'not visible' and s != 'can\'t tell':
                strings_new.append(s)
            else:
                print(strings, strings_new)

        if len(strings_new) > 1:
            answers[i] = ', '.join(strings_new)
        elif len(strings_new) == 1:
            answers[i] = strings_new[0]
        else:
            answers[i] = ''
    return answers


if __name__ == '__main__':
    dirname = '../data/AMT/synthetic_objects'
    save_dir = '../data/synthetic_objects'

    # list dir
    filenames = os.listdir(dirname)

    # read files
    data = []
    for name in filenames:
        filename = os.path.join(dirname, name)
        with open(filename, 'r') as f:
            lines = f.readlines()
        for l in lines:
            data.append(json.loads(l))

    # for each HIT
    num = len(data)
    count = 0
    annotations = {}
    for i in range(num):
        result = data[i]

        # skip bad worker
        if result['worker_id'] in bad_worker_ids:
            continue
        count += 1

        # annotations
        output = result['output']
        for j in range(len(output)):
            image_url = output[j]['image_url']
            answers = output[j]['answers']
            answers = clean_answers(answers)

            # get the object id
            paths = image_url[0].split('/')
            object_id = paths[-2]

            # merge annotations
            annotations = merge_annotations(annotations, object_id, answers)

    print(num, count)

    # write results
    object_ids = sorted(annotations.keys())
    for object_id in object_ids:
        print(object_id, annotations[object_id])

        filename = os.path.join(save_dir, object_id, 'questionnaire.txt')
        print(filename)

        f = open(filename, 'w')
        f.write('1. What is the name of the object in these images?')
        f.write('\n')
        f.write(annotations[object_id][0])
        f.write('\n')

        f.write('2. What is the category of the object in these images?')
        f.write('\n')
        f.write(annotations[object_id][1])
        f.write('\n')

        f.write('3. What is the object in these images made of? (list all materials of the object)')
        f.write('\n')
        f.write(annotations[object_id][2])
        f.write('\n')

        f.write('4. What can be the object in these images used for? (list all function of the object)')
        f.write('\n')
        f.write(annotations[object_id][3])
        f.write('\n')

        f.write('5. What is the color of the object in these images? (list all colors of the object)')
        f.write('\n')
        f.write(annotations[object_id][4])
        f.write('\n')
        f.close()

        print('=============================')
