#!/usr/bin/env python3
import os
import time

if __name__ == '__main__':

    set_name = 'synthetic'
    write = True

    # read folder names
    folder_names = []
    filename = '../data/%s_objects_folders.txt' % set_name
    with open(filename) as fp:
        for line in fp:
            folder_names.append(line.strip())
    print(folder_names, len(folder_names))

    # read categories
    categories = []
    filename = '../data/%s_objects_names.txt' % set_name
    with open(filename) as fp:
        for line in fp:
            categories.append(line.strip())
    print(categories, len(categories))

    # process categories
    names = sorted(set(categories))
    print(names, len(names))

    if write:
        # write categories
        filename = '../data/%s_categories.txt' % set_name
        with open(filename, 'w') as fp:
            for x in names:
                fp.write(x + '\n')

        # write to each folder
        for i in range(len(folder_names)):
            folder = folder_names[i]
            filename = os.path.join('../data/%s_objects' % set_name, folder, 'name.txt')
            with open(filename, 'w') as fp:
               fp.write(categories[i])
