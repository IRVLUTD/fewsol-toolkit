import os
import time

if __name__ == '__main__':

    # load category names
    filename = '../data/categories.txt'
    fp = open(filename, 'r')
    names = []
    for line in fp:
        names.append(line.strip())
    print(names, len(names))

    # list subdirs
    folder = '../data/OCID_objects'
    file_list = sorted(os.listdir(folder))
    print(file_list, len(file_list))

    # for each folder
    for subdir in file_list:
        filename = os.path.join(folder, subdir, 'name.txt')

        if not os.path.exists(filename):
            print('%s not exists' % filename)
            break

        # load annotation
        f = open(filename, 'r')
        annotation = f.readline().strip()
        f.close()

        if annotation not in names:
            print('%s not in list for %s' % (annotation, subdir))
            break

        if annotation == 'salt can':
            print('%s in %s' % (annotation, subdir))
