
import os
import time

# my libraries
from generate_scene_descriptions import parse_arguments
from render_scene_descriptions import check_dir

def main():

    args = parse_arguments()

    save_dir = args.save_path + \
                ('training_set/' if args.train_or_test == 'train' else 'test_set/')

    ### Calculate missing scenes ###

    # These are the missing directories. No scene description nor rendered data.
    temp = os.listdir(save_dir)
    temp1 = [int(x.split('_')[1]) for x in temp]
    missing_scenes = set(range(int(args.start_scene), int(args.end_scene))).difference(set(temp1)) 

    # For each directory, calculate whether the description or rendered data is missing
    temp = sorted(os.listdir(save_dir))
    missing_scene_descriptions = []
    missing_scene_renderings = []
    for direct in temp:

        directory_contents = os.listdir(save_dir + direct)
        scene_num = int(direct.split('_')[1])

        if 'scene_description.txt' not in directory_contents:
            missing_scene_descriptions.append(scene_num)

        if not check_dir(directory_contents):
            missing_scene_renderings.append(scene_num)

    missing_scene_descriptions = missing_scene_descriptions + list(missing_scenes)
    missing_scene_descriptions = sorted(missing_scene_descriptions)
    missing_scene_renderings = missing_scene_renderings + list(missing_scenes)
    missing_scene_renderings = sorted(missing_scene_renderings)


    print(f"{args.scenario}:")
    if len(missing_scene_descriptions) > 0:
        print(f"\tMissing scene descriptions: {missing_scene_descriptions}")
    else:
        print("\tScene description generation is complete!")

    if len(missing_scene_renderings) > 0:
        print(f"\tScenes with missing rendered data: {missing_scene_renderings}")
    else:
        print("\tScene rendering is complete!")

    if args.remove:
        for direct in temp:

            directory_contents = os.listdir(save_dir + direct)
            scene_num = int(direct.split('_')[1])

            if (scene_num in missing_scene_renderings or scene_num in
                    missing_scene_descriptions):

                print("removing " + str(direct))
                ff = os.listdir(save_dir + direct)
                time.sleep(0.1)
                for f in ff:
                    os.remove(os.path.join(save_dir, direct, f))
                os.rmdir(os.path.join(save_dir, direct))


if __name__ == '__main__':
    main()
